# -*- coding: utf-8 -*-
"""
Evaluate the fine-tuned ALLaM fraud/normal classifier.

Clean GitHub version of:
ALlam_Trying_after_fine_tuning.ipynb

This script loads:
1. The base ALLaM model from Hugging Face.
2. The local LoRA/PEFT adapter.
3. A test CSV with a `speech` column and, optionally, `type` and `is_noisy`.

No Google Drive, Colab, or personal paths are used.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


BASE_MODEL_ID = "humain-ai/ALLaM-7B-Instruct-preview"
MAX_LEN = 256
BATCH_SIZE = 64
NUM_WORKERS = 2

ID2LABEL = {0: "fraud", 1: "normal"}
LABEL2ID = {"fraud": 0, "normal": 1}


def load_tokenizer(adapter_dir: Path):
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), use_fast=True)
        print("Loaded tokenizer from adapter directory.")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
        print("Loaded tokenizer from base model.")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"
    return tokenizer


def load_model(adapter_dir: Path, tokenizer):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
    elif torch.cuda.is_available():
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    try:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

        base_model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL_ID,
            num_labels=2,
            quantization_config=bnb_config,
            device_map="auto",
        )
        print("Using 4-bit bitsandbytes.")
    except Exception as exc:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL_ID,
            num_labels=2,
            torch_dtype=compute_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        print("4-bit loading unavailable; using standard precision.")
        print("Reason:", exc)

    base_model.config.id2label = ID2LABEL
    base_model.config.label2id = LABEL2ID

    if base_model.config.pad_token_id is None:
        base_model.config.pad_token_id = tokenizer.pad_token_id

    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()

    if not hasattr(model, "hf_device_map"):
        model.to(device)

    print("Adapter loaded from:", adapter_dir)
    return model, device


def metrics_report(y_true, y_score, title="ALL"):
    report = {
        "title": title,
        "num_samples": int(len(y_true)),
    }

    try:
        # Convert fraud=0 to a positive binary indicator for AUC.
        fraud_true = (y_true == 0).astype(int)
        report["AUC"] = float(roc_auc_score(fraud_true, y_score))
    except Exception:
        report["AUC"] = None

    best = None

    for threshold in np.linspace(0.50, 0.995, 200):
        y_pred = np.where(y_score >= threshold, 0, 1)

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[0, 1],
            zero_division=0,
        )

        macro_f1 = f1.mean()

        if best is None or macro_f1 > best["best_macroF1"]:
            best = {
                "threshold": float(threshold),
                "best_macroF1": float(macro_f1),
                "best_accuracy": float(accuracy),
                "per_class_precision_[fraud(0),normal(1)]": [
                    float(precision[0]),
                    float(precision[1]),
                ],
                "per_class_recall_[fraud(0),normal(1)]": [
                    float(recall[0]),
                    float(recall[1]),
                ],
                "per_class_f1_[fraud(0),normal(1)]": [
                    float(f1[0]),
                    float(f1[1]),
                ],
                "confusion_matrix_rows_true_cols_pred_[0,1]": confusion_matrix(
                    y_true,
                    y_pred,
                    labels=[0, 1],
                ).tolist(),
            }

    report.update(best)
    return report


def evaluate(adapter_dir: Path, test_csv: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(adapter_dir)
    model, device = load_model(adapter_dir, tokenizer)

    df = pd.read_csv(test_csv).dropna(subset=["speech"]).copy()

    has_labels = "type" in df.columns
    has_noise = "is_noisy" in df.columns

    if has_labels:
        df["type"] = df["type"].astype(str).str.lower().str.strip()
        df["label"] = df["type"].map(LABEL2ID)

        if df["label"].isna().any():
            raise ValueError("Column 'type' must contain only fraud/normal labels.")

    print(
        "Rows:",
        len(df),
        "| has_labels:",
        has_labels,
        "| has_noise:",
        has_noise,
    )

    def collate_texts(batch_texts):
        return tokenizer(
            batch_texts,
            truncation=True,
            max_length=MAX_LEN,
            padding=True,
            return_tensors="pt",
        )

    texts = df["speech"].astype(str).tolist()

    loader = DataLoader(
        texts,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_texts,
        num_workers=NUM_WORKERS,
    )

    all_fraud_probs = []
    all_preds = []

    with torch.no_grad():
        progress = tqdm(
            total=len(df),
            desc="Evaluating",
            unit="ex",
        )

        for batch in loader:
            if hasattr(model, "hf_device_map"):
                model_device = next(model.parameters()).device
            else:
                model_device = device

            batch = {
                key: value.to(model_device, non_blocking=True)
                for key, value in batch.items()
            }

            logits = model(**batch).logits
            probabilities = torch.softmax(logits.float(), dim=-1)

            all_fraud_probs.append(
                probabilities[:, 0].detach().cpu().numpy()
            )
            all_preds.append(
                probabilities.argmax(dim=-1).detach().cpu().numpy()
            )

            progress.update(len(batch["input_ids"]))

        progress.close()

    fraud_probs = np.concatenate(all_fraud_probs)
    preds = np.concatenate(all_preds)

    df["prob_fraud"] = fraud_probs
    df["pred_label"] = preds
    df["pred"] = df["pred_label"].map(ID2LABEL)

    report = {
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_model": BASE_MODEL_ID,
        "adapter_dir": str(adapter_dir),
        "test_csv": str(test_csv),
    }

    if has_labels:
        y_true = df["label"].to_numpy().astype(int)

        report["overall"] = metrics_report(
            y_true,
            fraud_probs,
            title="OVERALL",
        )

        if has_noise:
            noise_values = df["is_noisy"].astype(int)

            clean_mask = noise_values == 0
            if clean_mask.any():
                report["clean"] = metrics_report(
                    y_true[clean_mask],
                    fraud_probs[clean_mask],
                    title="CLEAN",
                )

            noisy_mask = noise_values == 1
            if noisy_mask.any():
                report["noisy"] = metrics_report(
                    y_true[noisy_mask],
                    fraud_probs[noisy_mask],
                    title="NOISY",
                )

        overall = report["overall"]

        print(f"\n=== OVERALL ({overall['num_samples']} samples) ===")
        print("AUC:", overall["AUC"])
        print("Best threshold:", overall["threshold"])
        print("Best macro-F1:", overall["best_macroF1"])
        print("Best accuracy:", overall["best_accuracy"])
        print("Confusion matrix [fraud, normal]:")
        print(
            np.array(
                overall[
                    "confusion_matrix_rows_true_cols_pred_[0,1]"
                ]
            )
        )
    else:
        print("No 'type' column found; predictions only will be saved.")

    predictions_path = output_dir / "predictions_24k.csv"
    report_path = output_dir / "report_24k.json"

    df.to_csv(
        predictions_path,
        index=False,
        encoding="utf-8-sig",
    )

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("Saved predictions:", predictions_path)
    print("Saved report:", report_path)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the fine-tuned ALLaM fraud classifier."
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        required=True,
        help="Path to the local LoRA/PEFT adapter directory.",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        required=True,
        help="Path to the fraud/normal test CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/after_finetuning"),
        help="Directory for evaluation outputs.",
    )

    args = parser.parse_args()

    evaluate(
        adapter_dir=args.adapter_dir,
        test_csv=args.test_csv,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
