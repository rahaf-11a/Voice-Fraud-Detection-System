# -*- coding: utf-8 -*-
"""
Evaluate the original deepfake-audio model on the Arabic deepfake dataset
before fine-tuning.

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
CATEGORIES = ["real", "fake"]


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
        waveform = librosa.resample(
            waveform,
            orig_sr=sr,
            target_sr=16000,
        )
        sr = 16000

    return waveform, sr


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

    label = model.config.id2label[pred_id]
    return label, db, probabilities


def normalize_prediction(prediction):
    prediction = str(prediction).lower()

    if prediction in {"fake", "spoof"}:
        return "fake"

    if prediction in {"real", "bonafide"}:
        return "real"

    return "unknown"


def evaluate(dataset_dir: Path, output_csv: Path):
    model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)

    print("Loaded base model:", MODEL_ID)

    results = []
    skipped = []

    for split in SPLITS:
        for category in CATEGORIES:
            subset_path = dataset_dir / split / category

            if not subset_path.exists():
                print(f"Skipping missing folder: {subset_path}")
                continue

            print(f"Evaluating {split}/{category} ...")

            for root, _, files in os.walk(subset_path):
                for filename in files:
                    if not filename.lower().endswith((".wav", ".mp3", ".flac")):
                        continue

                    path = Path(root) / filename

                    prediction, db, probabilities = classify_audio(
                        path,
                        model,
                        extractor,
                    )

                    if prediction is None:
                        skipped.append(str(path))
                        continue

                    results.append(
                        {
                            "split": split,
                            "true_label": category,
                            "filename": filename,
                            "decibel": db,
                            "prediction": prediction,
                            "probabilities": probabilities.tolist(),
                        }
                    )

    df = pd.DataFrame(results)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nSaved predictions to:", output_csv)

    if df.empty:
        print("No valid audio files were evaluated.")
        return

    df["normalized_prediction"] = df["prediction"].apply(normalize_prediction)

    correct = (
        df["true_label"] == df["normalized_prediction"]
    ).sum()

    accuracy = correct / len(df)

    print(f"True accuracy: {accuracy * 100:.2f}%")
    print("Correct predictions:", int(correct))
    print("Incorrect predictions:", int(len(df) - correct))
    print("Skipped files:", len(skipped))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the original deepfake-audio model on the Arabic "
            "deepfake dataset before fine-tuning."
        )
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help=(
            "Dataset root containing train/val/test/final_eval "
            "with real and fake folders."
        ),
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "results/deepfake/base_arabic_dataset_predictions.csv"
        ),
        help="Path for saved predictions.",
    )

    args = parser.parse_args()

    evaluate(
        dataset_dir=args.dataset_dir,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
