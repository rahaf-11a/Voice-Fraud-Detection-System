"""Second-stage deepfake-audio fine-tuning with mixed training data.

Loads the first-stage checkpoint, trains on the new Arabic audio dataset,
and mixes 10% of the original training files into the new training split.
Validation and test sets use only the new dataset.
"""

import argparse
import random
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

EPOCHS = 40
PATIENCE = 5
LEARNING_RATE = 5e-6
BATCH_SIZE = 16
SAMPLE_RATE = 16_000
MAX_DURATION_SECONDS = 2
MIX_RATIO = 0.10
LABELS = {"real": 0, "fake": 1}


def class_files(split_dir: Path, class_name: str):
    """Find WAV files, accepting upper- or lowercase class directories."""
    found = []
    for directory in (split_dir / class_name, split_dir / class_name.capitalize()):
        if directory.is_dir():
            found.extend(
                (path, LABELS[class_name])
                for path in sorted(directory.iterdir())
                if path.is_file() and path.suffix.lower() == ".wav"
            )
    return found


class SmartAudioDataset(Dataset):
    def __init__(
        self,
        new_data_dir: Path,
        old_data_dir: Path,
        split: str,
        extractor,
        rng: random.Random,
    ):
        self.extractor = extractor
        self.samples = []
        for class_name in LABELS:
            self.samples.extend(class_files(new_data_dir / split, class_name))

        if split == "train":
            old_samples = []
            for class_name in LABELS:
                old_samples.extend(class_files(old_data_dir / "train", class_name))
            count = int(len(old_samples) * MIX_RATIO)
            self.samples.extend(rng.sample(old_samples, count))
            print(f"Added {count} original training clips to refined training")

        if not self.samples:
            raise ValueError(f"No WAV files were found in split: {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        try:
            import torchaudio
            waveform, sr = torchaudio.load(str(path))
            waveform = waveform.float().mean(dim=0).numpy()
        except Exception:
            audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
            waveform = audio.mean(axis=1)

        if sr != SAMPLE_RATE:
            waveform = librosa.resample(
                waveform, orig_sr=sr, target_sr=SAMPLE_RATE
            )

        target_length = SAMPLE_RATE * MAX_DURATION_SECONDS
        if len(waveform) < target_length:
            waveform = np.pad(waveform, (0, target_length - len(waveform)))
        else:
            waveform = waveform[:target_length]

        features = self.extractor(
            waveform,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding="max_length",
            max_length=target_length,
        )
        return {
            "input_values": features["input_values"][0],
            "labels": torch.tensor(label, dtype=torch.long),
        }


def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input_values"].to(device)
            labels = batch["labels"].to(device)
            predictions = model(inputs).logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    if total == 0:
        raise ValueError("Evaluation dataset contains no files")
    return 100 * correct / total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-dataset-dir", required=True, type=Path)
    parser.add_argument("--original-dataset-dir", required=True, type=Path)
    parser.add_argument("--first-stage-model", required=True, type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/refined_run")
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForAudioClassification.from_pretrained(
        args.first_stage_model
    ).to(device)
    extractor = AutoFeatureExtractor.from_pretrained(args.first_stage_model)
    rng = random.Random(args.seed)

    datasets = {
        split: SmartAudioDataset(
            args.new_dataset_dir, args.original_dataset_dir, split, extractor, rng
        )
        for split in ("train", "val", "test", "final_eval")
    }
    for split, dataset in datasets.items():
        print(f"{split}: {len(dataset)} files")

    train_loader = DataLoader(datasets["train"], batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(datasets["test"], batch_size=BATCH_SIZE, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    best_val_accuracy = -1.0
    patience_counter = 0
    best_model_dir = args.output_dir / "best_model"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        correct = total = 0
        running_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            inputs = batch["input_values"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs, labels=labels)
            outputs.loss.backward()
            optimizer.step()
            predictions = outputs.logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            running_loss += outputs.loss.item()

        train_accuracy = 100 * correct / total
        val_accuracy = accuracy(model, val_loader, device)
        print(
            f"Epoch {epoch + 1}: loss={running_loss / len(train_loader):.4f}, "
            f"train_acc={train_accuracy:.2f}%, val_acc={val_accuracy:.2f}%"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            model.save_pretrained(best_model_dir)
            extractor.save_pretrained(best_model_dir)
            print(f"Saved best model to {best_model_dir}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Evaluate the saved validation-best model rather than the final epoch.
    best_model = AutoModelForAudioClassification.from_pretrained(
        best_model_dir
    ).to(device)
    test_accuracy = accuracy(best_model, test_loader, device)
    print(f"Best model accuracy on test: {test_accuracy:.2f}%")
    print(f"Best model directory: {best_model_dir}")


if __name__ == "__main__":
    main()
