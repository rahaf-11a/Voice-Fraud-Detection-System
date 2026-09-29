"""
Speaker Verification Evaluation
Voice Fraud Detection System

Evaluates the fine-tuned ECAPA-TDNN speaker encoder using cosine similarity.

Supports:
- configurable test dataset
- 1 or more enrollment utterances per speaker
- positive (genuine) and negative (impostor) trials
- AUC, EER, accuracy, F1, macro-F1
- confusion matrix, FPR, and FFR
- JSON result export

The same evaluation procedure can be used before and after fine-tuning
to preserve a fair comparison.
"""

import argparse
import glob
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_curve,
)
from speechbrain.pretrained import EncoderClassifier
from tqdm import tqdm


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000


def load_wav_16k(path: str, target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Load audio as mono waveform and resample to 16 kHz."""
    wav, sr = torchaudio.load(path)

    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)

    wav = wav.squeeze(0)

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    return wav


def load_encoder(checkpoint: Path | None) -> EncoderClassifier:
    """
    Load pretrained SpeechBrain ECAPA-TDNN.

    If checkpoint is provided, replace the embedding-model weights
    with the fine-tuned project checkpoint.
    """
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": DEVICE},
    )

    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location=DEVICE)
        model.mods.embedding_model.load_state_dict(
            ckpt["embedding_model"],
            strict=True,
        )
        model.mods.embedding_model.eval()

    return model


@torch.inference_mode()
def embed_file(model: EncoderClassifier, path: str) -> np.ndarray:
    """Extract one L2-normalized ECAPA speaker embedding."""
    wav = load_wav_16k(path).to(DEVICE).unsqueeze(0)
    emb = model.encode_batch(wav)

    if emb.dim() == 3:
        emb = emb.squeeze(0).squeeze(0)

    emb = F.normalize(emb, p=2, dim=-1)

    return emb.detach().cpu().numpy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for L2-normalized vectors."""
    return float(np.dot(a, b))


def list_speakers_wavs(
    root_dir: Path,
    extensions=(".wav", ".flac", ".mp3"),
) -> Dict[str, List[str]]:
    """Read a dataset organized as one folder per speaker."""
    speaker_to_files = {}

    for speaker_dir in sorted(root_dir.glob("*")):
        if not speaker_dir.is_dir():
            continue

        files = []
        for ext in extensions:
            files.extend(glob.glob(str(speaker_dir / f"*{ext}")))

        files.sort()

        if files:
            speaker_to_files[speaker_dir.name] = files

    return speaker_to_files


def split_enroll_test(
    speaker_to_files: Dict[str, List[str]],
    enroll_per_speaker: int,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Split each speaker's files into enrollment and test utterances."""
    enroll = {}
    test = {}

    for speaker, files in speaker_to_files.items():
        if len(files) <= enroll_per_speaker:
            if len(files) >= 2:
                enroll[speaker] = files[:-1]
                test[speaker] = [files[-1]]
            else:
                enroll[speaker] = []
                test[speaker] = files
        else:
            enroll[speaker] = files[:enroll_per_speaker]
            test[speaker] = files[enroll_per_speaker:]

    return enroll, test


def build_centroids(
    model: EncoderClassifier,
    enroll: Dict[str, List[str]],
) -> Dict[str, np.ndarray]:
    """
    Create one normalized enrollment centroid per speaker.

    With one enrollment utterance, the centroid is simply that embedding.
    With multiple enrollment utterances, embeddings are averaged and normalized.
    """
    speaker_to_centroid = {}

    total_files = sum(len(files) for files in enroll.values())

    with tqdm(
        total=total_files,
        desc="Building enrollment centroids",
        unit="utt",
    ) as progress:
        for speaker, files in enroll.items():
            if not files:
                continue

            embeddings = []

            for file_path in files:
                embeddings.append(
                    embed_file(model, file_path)
                )
                progress.update(1)

            centroid = np.mean(
                embeddings,
                axis=0,
            )

            centroid = centroid / (
                np.linalg.norm(centroid) + 1e-8
            )

            speaker_to_centroid[speaker] = centroid

    return speaker_to_centroid


def calculate_eer(
    fpr: np.ndarray,
    tpr: np.ndarray,
    thresholds: np.ndarray,
) -> Tuple[float, float]:
    """Approximate Equal Error Rate (EER) and its threshold."""
    fnr = 1 - tpr
    index = np.nanargmin(np.abs(fnr - fpr))
    eer = (fnr[index] + fpr[index]) / 2

    return float(eer), float(thresholds[index])


def rates_from_confusion_matrix(cm: np.ndarray) -> Tuple[float, float]:
    """
    Compute:
    FPR = FP / (FP + TN)
    FFR = FN / (FN + TP)

    Rows are true labels and columns are predicted labels:
    [[TN, FP],
     [FN, TP]]
    """
    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    ffr = fn / (fn + tp) if (fn + tp) else 0.0

    return float(fpr), float(ffr)


def evaluate_verification(
    model: EncoderClassifier,
    speaker_to_centroid: Dict[str, np.ndarray],
    test: Dict[str, List[str]],
    max_negative_per_positive: int = 2,
):
    """
    Score genuine and impostor trials.

    For every test utterance:
    - one positive score is computed against the correct speaker centroid
    - up to N negative scores are computed against other speaker centroids
    """
    scores = []
    labels = []

    valid_speakers = [
        speaker
        for speaker in test
        if speaker in speaker_to_centroid
    ]

    if len(valid_speakers) < 2:
        raise RuntimeError(
            "At least two enrolled speakers are required."
        )

    positive_trials = sum(
        len(test[speaker])
        for speaker in valid_speakers
    )

    negative_trials = positive_trials * min(
        max_negative_per_positive,
        len(valid_speakers) - 1,
    )

    total_trials = (
        positive_trials + negative_trials
    )

    with tqdm(
        total=total_trials,
        desc="Verification scoring",
        unit="trial",
    ) as progress:

        for speaker in valid_speakers:
            positive_centroid = (
                speaker_to_centroid[speaker]
            )

            other_speakers = [
                other
                for other in valid_speakers
                if other != speaker
            ]

            for file_path in test[speaker]:
                embedding = embed_file(
                    model,
                    file_path,
                )

                # Genuine trial
                scores.append(
                    cosine_similarity(
                        embedding,
                        positive_centroid,
                    )
                )
                labels.append(1)
                progress.update(1)

                # Impostor trials for the same test utterance
                negative_sample = random.sample(
                    other_speakers,
                    k=min(
                        max_negative_per_positive,
                        len(other_speakers),
                    ),
                )

                for negative_speaker in negative_sample:
                    scores.append(
                        cosine_similarity(
                            embedding,
                            speaker_to_centroid[
                                negative_speaker
                            ],
                        )
                    )
                    labels.append(0)
                    progress.update(1)

    scores = np.asarray(scores)
    labels = np.asarray(labels)

    fpr_curve, tpr_curve, thresholds = roc_curve(
        labels,
        scores,
        pos_label=1,
    )

    auc_value = float(
        auc(
            fpr_curve,
            tpr_curve,
        )
    )

    eer, eer_threshold = calculate_eer(
        fpr_curve,
        tpr_curve,
        thresholds,
    )

    best_accuracy = -1.0
    best_accuracy_threshold = 0.0
    best_f1 = -1.0
    best_f1_threshold = 0.0
    best_macro_f1 = -1.0
    best_macro_f1_threshold = 0.0

    for threshold in np.linspace(
        scores.min(),
        scores.max(),
        200,
    ):
        predictions = (
            scores >= threshold
        ).astype(int)

        accuracy = accuracy_score(
            labels,
            predictions,
        )

        f1 = f1_score(
            labels,
            predictions,
            zero_division=0,
        )

        _, _, f1_values, _ = (
            precision_recall_fscore_support(
                labels,
                predictions,
                average=None,
                zero_division=0,
            )
        )

        macro_f1 = float(
            f1_values.mean()
        )

        if accuracy > best_accuracy:
            best_accuracy = float(accuracy)
            best_accuracy_threshold = float(
                threshold
            )

        if f1 > best_f1:
            best_f1 = float(f1)
            best_f1_threshold = float(
                threshold
            )

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_macro_f1_threshold = float(
                threshold
            )

    final_predictions = (
        scores >= best_macro_f1_threshold
    ).astype(int)

    cm = confusion_matrix(
        labels,
        final_predictions,
        labels=[0, 1],
    )

    fpr_value, ffr_value = (
        rates_from_confusion_matrix(cm)
    )

    return {
        "AUC": auc_value,
        "EER": eer,
        "EER_threshold": eer_threshold,
        "best_accuracy": best_accuracy,
        "best_threshold_by_accuracy": (
            best_accuracy_threshold
        ),
        "best_f1": best_f1,
        "best_threshold_by_f1": (
            best_f1_threshold
        ),
        "best_macroF1": best_macro_f1,
        "best_threshold_by_macroF1": (
            best_macro_f1_threshold
        ),
        "FPR": fpr_value,
        "FFR": ffr_value,
        "confusion_matrix_rows_true_cols_pred_[0,1]": (
            cm.tolist()
        ),
        "num_positive_trials": int(
            positive_trials
        ),
        "num_negative_trials": int(
            negative_trials
        ),
        "num_trials": int(
            total_trials
        ),
    }


def evaluate_dataset(
    model: EncoderClassifier,
    dataset_root: Path,
    enroll_per_speaker: int,
    max_negative_per_positive: int,
):
    """Run complete speaker-verification evaluation."""
    speaker_to_files = list_speakers_wavs(
        dataset_root
    )

    if not speaker_to_files:
        raise RuntimeError(
            f"No speaker audio found in {dataset_root}"
        )

    enroll, test = split_enroll_test(
        speaker_to_files,
        enroll_per_speaker,
    )

    test = {
        speaker: files
        for speaker, files in test.items()
        if files
    }

    speaker_to_centroid = build_centroids(
        model,
        enroll,
    )

    results = evaluate_verification(
        model,
        speaker_to_centroid,
        test,
        max_negative_per_positive,
    )

    results["dataset"] = str(dataset_root)
    results["num_speakers"] = len(
        speaker_to_files
    )
    results["num_utterances"] = sum(
        len(files)
        for files in speaker_to_files.values()
    )
    results["enroll_per_speaker"] = (
        enroll_per_speaker
    )
    results["max_negative_per_positive"] = (
        max_negative_per_positive
    )

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ECAPA-TDNN "
            "speaker verification."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help=(
            "Test dataset root "
            "(one folder per speaker)."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "checkpoints/voiceprint/"
            "aam_finetuning/"
            "ecapa_aam_best.pt"
        ),
        help=(
            "Fine-tuned ECAPA checkpoint. "
            "Use --pretrained-only to test "
            "the original VoxCeleb model."
        ),
    )

    parser.add_argument(
        "--pretrained-only",
        action="store_true",
        help=(
            "Evaluate the original pretrained "
            "ECAPA-VoxCeleb encoder without "
            "loading a project checkpoint."
        ),
    )

    parser.add_argument(
        "--enroll-per-speaker",
        type=int,
        default=1,
        help=(
            "Number of enrollment utterances "
            "used to build each speaker centroid."
        ),
    )

    parser.add_argument(
        "--max-negative-per-positive",
        type=int,
        default=2,
        help=(
            "Number of impostor comparisons "
            "per test utterance."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/voiceprint/"
            "speaker_verification_results.json"
        ),
        help="JSON output path.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint = (
        None
        if args.pretrained_only
        else args.checkpoint
    )

    model = load_encoder(
        checkpoint
    )

    results = evaluate_dataset(
        model=model,
        dataset_root=args.dataset,
        enroll_per_speaker=(
            args.enroll_per_speaker
        ),
        max_negative_per_positive=(
            args.max_negative_per_positive
        ),
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    print("\n--- Speaker Verification Results ---")
    print(f"AUC: {results['AUC']:.6f}")
    print(f"EER: {results['EER'] * 100:.2f}%")
    print(
        "Accuracy: "
        f"{results['best_accuracy'] * 100:.2f}%"
    )
    print(
        "Macro-F1: "
        f"{results['best_macroF1']:.6f}"
    )
    print(
        "FPR: "
        f"{results['FPR'] * 100:.2f}%"
    )
    print(
        "FFR: "
        f"{results['FFR'] * 100:.2f}%"
    )
    print(
        "Confusion Matrix: "
        f"{results['confusion_matrix_rows_true_cols_pred_[0,1]']}"
    )
    print(
        f"Saved results to: {args.output}"
    )


if __name__ == "__main__":
    main()
