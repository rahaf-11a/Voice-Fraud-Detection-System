"""First-stage deepfake-audio fine-tuning on the project Real/Fake dataset.

Clean, path-configurable adaptation of the original Colab training notebook.
Dataset layout: DATASET_DIR/{train,val,test}/{Real,Fake}/*.wav.
Model and tokenizer/feature-extractor files are written under OUTPUT_DIR.

Dependencies (original environment):
  transformers==4.43.3 datasets==3.0.1 torch torchaudio librosa
  scikit-learn soundfile tqdm numpy

For any gated Hugging Face model, authenticate separately using your own
Hugging Face CLI credentials; never commit tokens to this repository.
"""

import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

BASE_MODEL = "mo-thecreator/Deepfake-audio-detection"
EPOCHS = 50
PATIENCE = 5
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
SAMPLE_RATE = 16_000
MAX_DURATION_SECONDS = 2


class AudioDataset(Dataset):
    """The original two-class dataset: Real=0 and Fake=1."""

    def __init__(self, split_dir: Path, extractor):
        self.extractor = extractor
        self.data = []
        for folder_name, label in (("Real", 0), ("Fake", 1)):
            folder = split_dir / folder_name
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing class folder: {folder}")
            self.data.extend(
                (file, label)
                for file in sorted(folder.iterdir())
                if file.is_file() and file.suffix.lower() == ".wav"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path, label = self.data[index]
        try:
            waveform, sr = __import__("torchaudio").load(str(path))
            waveform = waveform.float()
        except Exception:
            audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
            waveform = torch.from_numpy(audio.T)

        waveform = waveform.mean(dim=0).numpy()
        if sr != SAMPLE_RATE:
            waveform = librosa.resample(
                waveform, orig_sr=sr, target_sr=SAMPLE_RATE
            )
        features = self.extractor(
            waveform,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=SAMPLE_RATE * MAX_DURATION_SECONDS,
        )
        return {
            "input_values": features["input_values"][0],
            "labels": torch.tensor(label, dtype=torch.long),
        }


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            audio = batch["input_values"].to(device)
            labels = batch["labels"].to(device)
            predictions = model(audio).logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    if not total:
        raise ValueError("Evaluation dataset is empty.")
    return 100 * correct / total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/first_run"))
    parser.add_argument("--base-model", default=BASE_MODEL)
    args = parser.parse_args()

    for split in ("train", "val", "test"):
        if not (args.dataset_dir / split).is_dir():
            raise FileNotFoundError(f"Missing dataset split: {split}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForAudioClassification.from_pretrained(args.base_model).to(device)
    extractor = AutoFeatureExtractor.from_pretrained(args.base_model)

    datasets = {
        split: AudioDataset(args.dataset_dir / split, extractor)
        for split in ("train", "val", "test")
    }
    for split, dataset in datasets.items():
        print(f"{split}: {len(dataset)} files")
        if not dataset:
            raise ValueError(f"The {split} dataset has no WAV files.")

    train_loader = DataLoader(datasets["train"], batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(datasets["test"], batch_size=BATCH_SIZE, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = -1.0
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total = correct = 0
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            audio = batch["input_values"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(audio, labels=labels)
            outputs.loss.backward()
            optimizer.step()
            predictions = outputs.logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            running_loss += outputs.loss.item()

        train_acc = 100 * correct / total
        val_acc = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch + 1}: loss={running_loss / len(train_loader):.4f}, "
            f"train_acc={train_acc:.2f}%, val_acc={val_acc:.2f}%"
        )

        # Match the original notebook: save each epoch and retain the best model.
        checkpoint_dir = args.output_dir / f"epoch_{epoch + 1}"
        model.save_pretrained(checkpoint_dir)
        extractor.save_pretrained(checkpoint_dir)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_dir = args.output_dir / "best_model"
            model.save_pretrained(best_dir)
            extractor.save_pretrained(best_dir)
            print(f"Saved best model to {best_dir}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    best_model = AutoModelForAudioClassification.from_pretrained(
        args.output_dir / "best_model"
    ).to(device)
    test_acc = evaluate(best_model, test_loader, device)
    print(f"Best model accuracy on test: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
