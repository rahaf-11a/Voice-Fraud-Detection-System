# -*- coding: utf-8 -*-
"""
Evaluate the base ALLaM model before fine-tuning.

Clean GitHub version of:
ALlam_Testing_before_fine_tuning.ipynb

No Google Drive, Colab, or personal paths are used.
"""

import argparse
import re
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "humain-ai/ALLaM-7B-Instruct-preview"
LABELS = {"fraud", "normal"}
PATTERN = re.compile(r"label\s*:\s*(fraud|normal)\b", re.IGNORECASE)

BATCH_SIZE = 8
MAX_NEW_TOKENS = 8
MAX_INPUT_TOKENS = 256


def build_messages(text: str):
    return [
        {
            "role": "user",
            "content": (
                "صنّف الجملة التالية إلى إحدى الفئتين فقط: fraud أو normal.\n"
                "أجب بسطر واحد فقط وبدون أي كلمات إضافية بهذا الشكل: "
                "label: fraud أو label: normal.\n"
                f"الجملة: «{text}»"
            ),
        }
    ]


def load_model(model_id: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
    )
    model.eval()

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer, model, device


@torch.no_grad()
def classify_batch(text_list, tokenizer, model, device):
    chats = [
        tokenizer.apply_chat_template(
            build_messages(text),
            tokenize=False,
            add_generation_prompt=True,
        )
        for text in text_list
    ]

    inputs = tokenizer(
        chats,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
        return_token_type_ids=False,
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=model.config.pad_token_id,
    )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    predictions = []
    for response in decoded:
        match = PATTERN.search(response)

        if match:
            predictions.append(match.group(1).lower())
            continue

        low = response.lower()

        if ("fraud" in low) ^ ("normal" in low):
            predictions.append("fraud" if "fraud" in low else "normal")
        else:
            predictions.append("unknown")

    return predictions


def evaluate(input_csv: Path, output_dir: Path, model_id: str):
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model, device = load_model(model_id)

    df = pd.read_csv(input_csv)

    if not {"speech", "type"}.issubset(df.columns):
        raise ValueError("Input CSV must contain 'speech' and 'type' columns.")

    df = df.dropna(subset=["speech", "type"]).copy()
    df["type"] = df["type"].astype(str).str.lower().str.strip()

    invalid_labels = ~df["type"].isin(LABELS)
    if invalid_labels.any():
        bad = df.loc[invalid_labels, "type"].value_counts()
        raise ValueError(f"Unexpected labels found:\n{bad}")

    demo = [
        "أرسل بيانات بطاقتك البنكية علشان نفعّل الخدمة",
        "مساء الخير، موعدنا بكرة الساعة ٨؟",
    ]
    print("Smoke test predictions:", classify_batch(demo, tokenizer, model, device))

    predictions = []
    unknown_count = 0
    error_count = 0
    printed_error = False

    total = len(df)
    partial_path = output_dir / "allam_preds_partial.csv"

    progress = tqdm(total=total, desc="Classifying (batched)")

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        texts = df.iloc[start:end]["speech"].tolist()

        try:
            batch_predictions = classify_batch(
                texts,
                tokenizer,
                model,
                device,
            )
        except Exception as exc:
            if not printed_error:
                tqdm.write(f"First error: {type(exc).__name__}: {exc}")
                printed_error = True

            batch_predictions = ["unknown"] * (end - start)
            error_count += end - start

        unknown_count += sum(
            1 for prediction in batch_predictions if prediction == "unknown"
        )
        predictions.extend(batch_predictions)

        progress.update(end - start)

        if end % 100 == 0:
            progress.set_postfix(
                unknown=unknown_count,
                errors=error_count,
            )

        if end % 1000 == 0:
            partial = df.iloc[:end].copy()
            partial["pred"] = predictions
            partial.to_csv(partial_path, index=False)

    progress.close()

    df["pred"] = predictions

    unknown_rate = (df["pred"] == "unknown").mean()
    known_mask = df["pred"].isin(LABELS)

    y_true = df.loc[known_mask, "type"].values
    y_pred = df.loc[known_mask, "pred"].values

    accuracy = accuracy_score(y_true, y_pred) if len(y_true) else 0.0

    if len(y_true):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=["fraud", "normal"],
            zero_division=0,
        )
        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=["fraud", "normal"],
        )
    else:
        precision = recall = f1 = [0, 0]
        matrix = [[0, 0], [0, 0]]

    print(f"\nTotal samples: {len(df)}")
    print(
        f"Unknown outputs: {unknown_rate * 100:.2f}% "
        f"(count={int(unknown_rate * len(df))})"
    )
    print(f"Accuracy (known only): {accuracy * 100:.2f}%")

    print("\nPer-class metrics (order: fraud, normal)")
    print(f"Precision: {precision}")
    print(f"Recall:    {recall}")
    print(f"F1:        {f1}")

    print("\nConfusion Matrix (rows=true, cols=pred) [fraud, normal]:")
    print(matrix)

    if len(y_true):
        print("\nClassification report:")
        print(
            classification_report(
                y_true,
                y_pred,
                labels=["fraud", "normal"],
                zero_division=0,
            )
        )

    predictions_path = output_dir / "allam_preds.csv"
    df.to_csv(predictions_path, index=False)
    print(f"\nSaved predictions to: {predictions_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate base ALLaM before fraud-classification fine-tuning."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Path to the fraud/normal testing CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/before_finetuning"),
        help="Directory where predictions will be saved.",
    )
    parser.add_argument(
        "--model-id",
        default=MODEL_ID,
        help="Hugging Face base model ID.",
    )

    args = parser.parse_args()

    evaluate(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        model_id=args.model_id,
    )


if __name__ == "__main__":
    main()
