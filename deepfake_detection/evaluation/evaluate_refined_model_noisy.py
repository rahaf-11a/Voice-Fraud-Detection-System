"""Evaluate the refined deepfake audio classifier on noisy audio files."""

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac"}
CLASS_LABELS = {"real": 0, "fake": 1}


def collect_noisy_audio(dataset_roots):
    """Collect noisy clips from final_eval/real and final_eval/fake."""
    records = []
    for root in dataset_roots:
        for label, class_id in CLASS_LABELS.items():
            # Datasets may capitalize the Real/Fake folder names differently.
            final_eval = root / "final_eval"
            if not final_eval.is_dir():
                raise FileNotFoundError(f"Missing evaluation directory: {final_eval}")
            label_dir = next(
                (item for item in final_eval.iterdir()
                 if item.is_dir() and item.name.lower() == label),
                None,
            )
            if label_dir is None:
                raise FileNotFoundError(f"Missing {label} folder under {final_eval}")
            for path in sorted(label_dir.iterdir()):
                if (path.is_file() and "noisy" in path.name.lower()
                        and path.suffix.lower() in AUDIO_EXTENSIONS):
                    records.append({"path": path, "true_label": class_id})
    if not records:
        raise ValueError("No files containing 'noisy' were found in the evaluation folders.")
    return records


def prepare_audio(path, extractor):
    audio, sample_rate = sf.read(path, always_2d=True)
    audio = audio.mean(axis=1)
    # Use resampling whenever a recording is not already 16 kHz.
    if sample_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
    inputs = extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    return inputs


def evaluate(model_dir, dataset_roots, output_csv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForAudioClassification.from_pretrained(model_dir).to(device)
    extractor = AutoFeatureExtractor.from_pretrained(model_dir)
    model.eval()

    # These class IDs follow the labels used in the project datasets.
    if model.config.num_labels != 2:
        raise ValueError("Expected a binary Real/Fake classifier.")

    records = collect_noisy_audio(dataset_roots)
    print(f"Total noisy files found: {len(records)}")
    results = []
    for item in records:
        inputs = prepare_audio(item["path"], extractor)
        inputs = {name: values.to(device) for name, values in inputs.items()}
        with torch.inference_mode():
            probabilities = torch.softmax(model(**inputs).logits, dim=-1)[0].cpu().numpy()

        predicted_id = int(np.argmax(probabilities))
        results.append({
            "file": item["path"].name,
            "path": str(item["path"]),
            "true_label": "Real" if item["true_label"] == 0 else "Fake",
            "predicted": "Real" if predicted_id == 0 else "Fake",
            "real_prob": float(probabilities[0]),
            "fake_prob": float(probabilities[1]),
            "true_id": item["true_label"],
            "predicted_id": predicted_id,
        })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.drop(columns=["true_id", "predicted_id", "path"]).to_csv(output_csv, index=False)
    print(f"Accuracy on noisy files: {accuracy_score(df.true_id, df.predicted_id):.2%}")
    print(classification_report(
        df.true_id, df.predicted_id, labels=[0, 1],
        target_names=["Real", "Fake"], zero_division=0,
    ))
    print("Confusion matrix (rows: true Real/Fake; columns: predicted Real/Fake):")
    print(confusion_matrix(df.true_id, df.predicted_id, labels=[0, 1]))
    print(f"Results saved to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Directory containing the refined best_model checkpoint.")
    parser.add_argument("--dataset-dir", type=Path, required=True, nargs="+",
                        help="One or more dataset roots with final_eval/real and final_eval/fake.")
    parser.add_argument("--output-csv", type=Path,
                        default=Path("results/deepfake/evaluation_noisy_results.csv"))
    args = parser.parse_args()
    evaluate(args.model_dir, args.dataset_dir, args.output_csv)


if __name__ == "__main__":
    main()
