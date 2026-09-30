# -*- coding: utf-8 -*-
"""
Fine-tune ALLaM-7B for fraud/normal classification using LoRA/PEFT.

Clean GitHub version of:
fine_tuning_ALlam_Classify_Fraud_Normal_1.ipynb

Personal Google Drive / Colab paths have been removed.
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


MODEL_ID = "humain-ai/ALLaM-7B-Instruct-preview"

SEED = 42
MAX_LEN = 256
TRAIN_BS = 8
EVAL_BS = 32
EPOCHS = 2
LR = 1e-4
WARMUP = 0.1
WEIGHT_DEC = 0.05
GRAD_ACC = 4
LABEL_SMOOTH = 0.05
NORMAL_CLASS_WEIGHT = 1.25
USE_NORMALIZE = True
EARLY_STOP_PATIENCE = 1


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_ar(text: str) -> str:
    if not USE_NORMALIZE:
        return str(text)

    text = str(text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[ًٌٍَُِّْ]", "", text)
    text = re.sub(r"ـ", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_data(data_csv: Path):
    df = pd.read_csv(data_csv).dropna(subset=["speech", "type"]).copy()
    df["type"] = df["type"].astype(str).str.lower().str.strip()

    if not df["type"].isin(["fraud", "normal"]).all():
        raise ValueError("Column 'type' must contain only fraud/normal labels.")

    if "is_noisy" not in df.columns:
        df["is_noisy"] = False

    print(
        "Counts:",
        df["type"].value_counts().to_dict(),
        "| Noisy:",
        df["is_noisy"].value_counts().to_dict(),
    )

    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=SEED,
        stratify=df["type"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_df["type"],
    )

    print(
        "Split sizes -> train/val/test:",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    return train_df, val_df, test_df


class WeightedSmoothedCELoss(nn.Module):
    def __init__(self, class_weights=None, smoothing=0.0, num_classes=2):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else torch.ones(num_classes),
        )

    def forward(self, logits, target):
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        log_probs = torch.log_softmax(logits, dim=1)
        weights = self.class_weights[target]
        loss = -(true_dist * log_probs).sum(dim=1) * weights
        return loss.mean()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        labels=[0, 1],
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "precision_fraud": float(precision[0]),
        "recall_fraud": float(recall[0]),
        "f1_fraud": float(f1[0]),
        "precision_normal": float(precision[1]),
        "recall_normal": float(recall[1]),
        "f1_normal": float(f1[1]),
        "macro_f1": float(f1.mean()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune ALLaM for fraud/normal classification."
    )
    parser.add_argument(
        "--data-csv",
        type=Path,
        required=True,
        help="Path to the prepared 100k training CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/allam_finetune"),
        help="Directory where the LoRA adapter and results will be saved.",
    )
    args = parser.parse_args()

    set_seed()

    use_cuda = torch.cuda.is_available()
    print(
        "CUDA:"
        if use_cuda
        else "Running on CPU.",
        torch.cuda.get_device_name(0) if use_cuda else "",
    )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = args.output_dir / f"allam7b_finetune_100k_with_noise-{stamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    print("SAVE_DIR:", save_dir)

    train_df, val_df, test_df = load_data(args.data_csv)

    label2id = {"fraud": 0, "normal": 1}
    id2label = {0: "fraud", 1: "normal"}

    for split in (train_df, val_df, test_df):
        split["label"] = split["type"].map(label2id)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
    elif torch.cuda.is_available():
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    use_bnb = False

    try:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

        base_model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID,
            num_labels=2,
            quantization_config=bnb_config,
            device_map="auto",
        )
        use_bnb = True
        print("Using 4-bit bitsandbytes.")
    except Exception as exc:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID,
            num_labels=2,
            torch_dtype=compute_dtype,
            device_map="auto",
        )
        print("bitsandbytes unavailable; using FP16/BF16/FP32 fallback:", exc)

    base_model.config.id2label = id2label
    base_model.config.label2id = label2id

    if base_model.config.pad_token_id is None:
        base_model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj",
        ],
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    if use_bnb:
        try:
            from transformers import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(model)
        except Exception:
            pass

    def tokenize_batch(batch):
        texts = [normalize_ar(text) for text in batch["speech"]]
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=MAX_LEN,
        )
        tokenized["labels"] = batch["label"]
        return tokenized

    ds_train = Dataset.from_pandas(
        train_df[["speech", "label", "is_noisy"]],
        preserve_index=False,
    )
    ds_val = Dataset.from_pandas(
        val_df[["speech", "label", "is_noisy"]],
        preserve_index=False,
    )
    ds_test = Dataset.from_pandas(
        test_df[["speech", "label", "is_noisy"]],
        preserve_index=False,
    )

    ds_train = ds_train.map(
        tokenize_batch,
        batched=True,
        remove_columns=["speech", "is_noisy"],
    )
    ds_val = ds_val.map(
        tokenize_batch,
        batched=True,
        remove_columns=["speech", "is_noisy"],
    )
    ds_test = ds_test.map(
        tokenize_batch,
        batched=True,
        remove_columns=["speech", "is_noisy"],
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
    )

    class_weights = torch.tensor(
        [1.0, NORMAL_CLASS_WEIGHT],
        dtype=torch.float32,
    ).to(model.device)

    loss_fn = WeightedSmoothedCELoss(
        class_weights=class_weights,
        smoothing=LABEL_SMOOTH,
        num_classes=2,
    )

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            loss = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss

    training_args = TrainingArguments(
        output_dir=str(save_dir),
        seed=SEED,
        per_device_train_batch_size=TRAIN_BS,
        per_device_eval_batch_size=EVAL_BS,
        gradient_accumulation_steps=GRAD_ACC,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        weight_decay=WEIGHT_DEC,
        warmup_ratio=WARMUP,
        lr_scheduler_type="cosine",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=(compute_dtype == torch.float16),
        bf16=(compute_dtype == torch.bfloat16),
        gradient_checkpointing=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
        report_to=[],
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOP_PATIENCE
            )
        ],
    )

    print("Starting training...")
    training_output = trainer.train()
    print(training_output)

    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    print(f"Saved PEFT adapter and tokenizer to: {save_dir}")

    def evaluate_split(dataset, title):
        print(f"\n=== EVAL on {title} ===")

        output = trainer.predict(
            dataset,
            metric_key_prefix=f"eval_{title}",
        )

        logits = output.predictions
        y_true = output.label_ids

        probabilities = torch.softmax(
            torch.tensor(logits),
            dim=-1,
        ).numpy()

        fraud_probabilities = probabilities[:, 0]
        y_pred = logits.argmax(axis=-1)

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[0, 1],
            zero_division=0,
        )
        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        )

        print(f"Accuracy: {accuracy * 100:.2f}%")
        print("Per-class (0=fraud, 1=normal)")
        print("Precision:", np.round(precision, 3))
        print("Recall:", np.round(recall, 3))
        print("F1:", np.round(f1, 3))
        print("Confusion matrix:\n", matrix)

        try:
            auc = roc_auc_score(
                y_true,
                fraud_probabilities,
            )
            print("ROC-AUC:", round(auc, 4))
        except Exception:
            pass

        return y_true, fraud_probabilities

    y_val, p_val = evaluate_split(ds_val, "val")

    best = None

    for threshold in np.linspace(0.50, 0.995, 200):
        pred = np.where(p_val >= threshold, 0, 1)

        f1_fraud = f1_score(
            y_val,
            pred,
            pos_label=0,
            zero_division=0,
        )
        f1_normal = f1_score(
            y_val,
            pred,
            pos_label=1,
            zero_division=0,
        )
        macro = 0.5 * (f1_fraud + f1_normal)
        accuracy = accuracy_score(y_val, pred)
        recall = precision_recall_fscore_support(
            y_val,
            pred,
            labels=[0, 1],
            zero_division=0,
        )[1]

        if best is None or macro > best["macro"]:
            best = {
                "thr": float(threshold),
                "macro": float(macro),
                "acc": float(accuracy),
                "rec": [
                    float(recall[0]),
                    float(recall[1]),
                ],
            }

    print("\nBest threshold on validation set:")
    print(json.dumps(best, indent=2))

    y_test, p_test = evaluate_split(
        ds_test,
        "test_raw_argmax",
    )

    threshold = best["thr"]
    yhat_test = np.where(
        p_test >= threshold,
        0,
        1,
    )

    accuracy = accuracy_score(y_test, yhat_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        yhat_test,
        labels=[0, 1],
        zero_division=0,
    )
    matrix = confusion_matrix(
        y_test,
        yhat_test,
        labels=[0, 1],
    )

    print(f"\n=== TEST @ threshold={threshold:.4f} ===")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Precision:", np.round(precision, 3))
    print("Recall:", np.round(recall, 3))
    print("F1:", np.round(f1, 3))
    print("Confusion matrix:\n", matrix)

    with open(
        save_dir / "val_best_threshold.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            best,
            file,
            ensure_ascii=False,
            indent=2,
        )

    pd.DataFrame(
        {
            "label": y_val.astype(int),
            "prob_fraud": p_val.astype(float),
        }
    ).to_csv(
        save_dir / "val_probs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        {
            "label": y_test.astype(int),
            "prob_fraud": p_test.astype(float),
            "pred_thr": yhat_test.astype(int),
        }
    ).to_csv(
        save_dir / f"test_preds_thr_{threshold:.4f}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Saved adapter and evaluation outputs to:", save_dir)


if __name__ == "__main__":
    main()
