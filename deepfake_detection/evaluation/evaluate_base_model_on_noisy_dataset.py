# -*- coding: utf-8 -*-
"""
Evaluate the original deepfake-audio model before fine-tuning.

Clean GitHub version of the original Colab notebook used for the
project deepfake dataset evaluation.

No Google Drive, Colab, or personal paths are used.
"""

import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


MODEL_ID = "mo-thecreator/Deepfake-audio-detection"
SPLITS = ["train", "val", "test", "final_eval"]
CATEGORIES = ["Real", "Fake"]


def load_audio(path: Path):
    try:
        waveform, sr = torchaudio.load(str(path))
    except Exception:
        waveform, sr = sf.read(str(path))
        waveform = torch.tensor(waveform).unsqueeze(0)

    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0)

    waveform = waveform.squeeze().numpy()

    if sr != 16000:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
        sr = 16000

    return waveform, sr


def count_clean_noisy(base_dir: Path):
    summary = {}

    for split in SPLITS:
        summary[split] = {}
        for category in CATEGORIES:
            folder = base_dir / split / category

            if not folder.exists():
                summary[split][category] = (0, 0)
                continue

            files = [
                f for f in folder.iterdir()
                if f.is_file() and f.suffix.lower() == ".wav"
            ]
            clean = [f for f in files if not f.name.lower().endswith("_noisy.wav")]
            noisy = [f for f in files if f.name.lower().endswith("_noisy.wav")]

            summary[split][category] = (len(clean), len(noisy))
            print(
                f"{split:<12} | {category:<5} | clean: {len(clean):>5} | "
                f"noisy: {len(noisy):>5} | total: {len(files):>6}"
            )

    total_clean = sum(summary[s][c][0] for s in SPLITS for c in CATEGORIES)
    total_noisy = sum(summary[s][c][1] for s in SPLITS for c in CATEGORIES)
    total = total_clean + total_noisy

    print("\nDataset summary:")
    print("Clean files:", total_clean)
    print("Noisy files:", total_noisy)
    print("Total files:", total)

    if total:
        print(f"Noisy ratio: {(total_noisy / total) * 100:.2f}%")

    return summary


def calculate_db(audio):
    rms = np.sqrt(np.mean(audio ** 2))
    return 20 * np.log10(rms + 1e-10)


def classify_audio(file_path, model, extractor):
    waveform, _ = load_audio(file_path)

    if len(waveform) < 3200:
        return None, None, None

    db = calculate_db(waveform)
    inputs = extractor(
        waveform,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        pred_id = torch.argmax(logits, dim=-1).item()

    return model.config.id2label[pred_id], db, probabilities


def evaluate(base_dir: Path, output_csv: Path):
    count_clean_noisy(base_dir)

    model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)

    results = []
    skipped_files = []

    for split in SPLITS:
        for category in CATEGORIES:
            subset_path = base_dir / split / category

            if not subset_path.exists():
                continue

            for root, _, files in os.walk(subset_path):
                for filename in files:
                    if not filename.lower().endswith((".wav", ".mp3", ".flac")):
                        continue

                    path = Path(root) / filename
                    label, db, probabilities = classify_audio(path, model, extractor)

                    if label is None:
                        skipped_files.append(filename)
                        continue

                    results.append(
                        {
                            "subset": split,
                            "category": category,
                            "filename": filename,
                            "decibel": db,
                            "prediction": label,
                            "probabilities": probabilities.tolist(),
                        }
                    )

    df = pd.DataFrame(results)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print("Saved predictions to:", output_csv)

    if len(df) == 0:
        return

    def normalize_prediction(prediction):
        prediction = str(prediction).lower()
        if prediction in {"fake", "spoof"}:
            return "fake"
        if prediction in {"real", "bonafide"}:
            return "real"
        return "unknown"

    df["true_label"] = df["category"].str.lower()
    df["normalized_prediction"] = df["prediction"].apply(normalize_prediction)

    correct = (df["true_label"] == df["normalized_prediction"]).sum()
    accuracy = correct / len(df)

    print(f"True accuracy: {accuracy * 100:.2f}%")
    print("Correct predictions:", int(correct))
    print("Incorrect predictions:", int(len(df) - correct))
    print("Skipped files:", len(skipped_files))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the base deepfake-audio model before fine-tuning."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Dataset root containing train/val/test/final_eval with Real and Fake folders.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/deepfake/base_project_dataset_predictions.csv"),
    )

    args = parser.parse_args()
    evaluate(args.dataset_dir, args.output_csv)


if __name__ == "__main__":
    main()
