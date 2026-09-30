# -*- coding: utf-8 -*-
"""
Dataset preprocessing for fraudulent-speech classification.

Clean GitHub version of:
arranging_Dataset_normal_fraud.ipynb

It performs two tasks from the original notebook:
1. Shuffle and interleave the 100k training dataset by label.
2. Add simulated Arabic transcription noise to the 24k testing dataset.

No Google Drive, Colab, or personal paths are used.
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils import shuffle as sk_shuffle
from tqdm.auto import tqdm


SEED = 42


def validate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Keep valid rows and normalize the fraud/normal label."""
    df = df.dropna(subset=["speech", "type"]).copy()
    df["type"] = df["type"].astype(str).str.lower().str.strip()

    valid = df["type"].isin(["fraud", "normal"])
    if not valid.all():
        bad = df.loc[~valid, "type"].value_counts()
        raise ValueError(f"Unexpected values in 'type':\n{bad}")

    return df


def interleave_by_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Alternate fraud and normal rows.

    This preserves the intended behavior of the original notebook while keeping
    complete rows aligned across all columns.
    """
    fraud = df[df["type"] == "fraud"].reset_index(drop=True)
    normal = df[df["type"] == "normal"].reset_index(drop=True)

    n = min(len(fraud), len(normal))
    rows = []

    for i in range(n):
        rows.append(fraud.iloc[i])
        rows.append(normal.iloc[i])

    if len(fraud) > n:
        rows.extend(fraud.iloc[n:].itertuples(index=False, name=None))
    if len(normal) > n:
        rows.extend(normal.iloc[n:].itertuples(index=False, name=None))

    if not rows:
        return df.iloc[0:0].copy()

    # First 2*n entries are Series; any tail entries may be tuples.
    records = []
    columns = list(df.columns)
    for row in rows:
        if isinstance(row, pd.Series):
            records.append(row.to_dict())
        else:
            records.append(dict(zip(columns, row)))

    return pd.DataFrame(records, columns=columns)


def prepare_training_dataset(input_csv: Path, output_dir: Path):
    """Shuffle the training dataset and create an interleaved version."""
    df = pd.read_csv(input_csv)
    df = validate_labels(df)

    print("Training counts before shuffle:", df["type"].value_counts().to_dict())

    shuffled = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    interleaved = interleave_by_label(shuffled)

    shuffled_path = output_dir / "calls_100k_shuffled_utf8.csv"
    interleaved_path = output_dir / "calls_100k_shuffled_interleaved_utf8.csv"

    shuffled.to_csv(
        shuffled_path,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_MINIMAL,
    )
    interleaved.to_csv(
        interleaved_path,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_MINIMAL,
    )

    print(f"Saved: {shuffled_path}")
    print(f"Saved: {interleaved_path}")


def _noise_params(level="med"):
    levels = {
        "mild": dict(del_p=0.05, sub_p=0.08, ins_p=0.03, space_p=0.04, drop_al_p=0.10),
        "med": dict(del_p=0.08, sub_p=0.12, ins_p=0.05, space_p=0.07, drop_al_p=0.15),
        "strong": dict(del_p=0.12, sub_p=0.18, ins_p=0.08, space_p=0.10, drop_al_p=0.22),
    }
    return levels[level]


_SUBS_MAP = {
    "أ": "ا", "إ": "ا", "آ": "ا", "ة": "ه", "ى": "ي", "ؤ": "و", "ئ": "ي",
    "ق": "ك", "ث": "ت", "ذ": "د", "ظ": "ض", "ح": "ه", "ط": "ت",
    "ص": "س", "ض": "ظ", "غ": "ق", "؟": "", "،": "", "؛": "", "!": "",
}

_DROP_VOWELS = {"ا", "و", "ي"}


def noisify_ar(text, rng, del_p, sub_p, ins_p, space_p, drop_al_p):
    """Simulate character-level Arabic transcription noise."""
    s = re.sub(r"\s+", " ", str(text)).strip()

    def drop_al(match):
        word = match.group(0)
        return word[2:] if rng.rand() < drop_al_p else word

    s = re.sub(r"\bال(?=\w)", drop_al, s)

    out = []
    for ch in s:
        if ch == " ":
            if rng.rand() < space_p:
                continue
            out.append(ch)
            continue

        r = rng.rand()

        if r < del_p or (ch in _DROP_VOWELS and rng.rand() < del_p):
            continue
        elif r < del_p + sub_p and ch in _SUBS_MAP:
            out.append(_SUBS_MAP[ch])
        else:
            out.append(ch)
            if rng.rand() < ins_p:
                out.append(ch if rng.rand() < 0.6 else ("ه" if rng.rand() < 0.5 else "و"))

    noisy = re.sub(r"\s+", " ", "".join(out)).strip()
    if not noisy:
        noisy = s[: max(1, len(s) // 2)]

    return noisy


def make_noisy_series(texts, level="med", seed=123):
    params = _noise_params(level)
    rng = np.random.RandomState(seed)

    noisy = []
    for text in tqdm(texts, desc=f"Noisifying [{level}]"):
        noisy.append(noisify_ar(text, rng, **params))

    return noisy


def add_noise_per_class(
    df_clean: pd.DataFrame,
    frac=0.5,
    mode="replace",
    level="mix",
    seed=SEED,
):
    """Apply the original clean/mild/medium/strong noise strategy per class."""
    if not 0.0 <= frac <= 1.0:
        raise ValueError("frac must be between 0 and 1.")

    rng_global = np.random.RandomState(seed)
    out = df_clean.copy()
    out["is_noisy"] = False
    frames = [out]

    if str(level).lower() == "mix":
        for label in ["fraud", "normal"]:
            idx = out.index[out["type"] == label].to_numpy()
            if len(idx) == 0:
                continue

            q = len(idx) // 4

            chosen_mild = (
                rng_global.choice(idx, size=q, replace=False)
                if q > 0 else np.array([], dtype=int)
            )
            rem1 = np.setdiff1d(idx, chosen_mild, assume_unique=False)

            chosen_med = (
                rng_global.choice(rem1, size=min(q, len(rem1)), replace=False)
                if len(rem1) > 0 else np.array([], dtype=int)
            )
            rem2 = np.setdiff1d(rem1, chosen_med, assume_unique=False)

            chosen_strong = (
                rng_global.choice(rem2, size=min(q, len(rem2)), replace=False)
                if len(rem2) > 0 else np.array([], dtype=int)
            )

            def apply_noise(indices, noise_level, seed_offset):
                if len(indices) == 0:
                    return

                noisy_texts = make_noisy_series(
                    out.loc[indices, "speech"].astype(str).tolist(),
                    level=noise_level,
                    seed=seed + seed_offset + (0 if label == "fraud" else 1),
                )

                if mode == "replace":
                    out.loc[indices, "speech"] = noisy_texts
                    out.loc[indices, "is_noisy"] = True
                elif mode == "augment":
                    add = out.loc[indices].copy()
                    add["speech"] = noisy_texts
                    add["is_noisy"] = True
                    frames.append(add)
                else:
                    raise ValueError("mode must be 'replace' or 'augment'.")

            apply_noise(chosen_mild, "mild", 10)
            apply_noise(chosen_med, "med", 20)
            apply_noise(chosen_strong, "strong", 30)

        if mode == "augment":
            out = pd.concat(frames, ignore_index=True)
            out = sk_shuffle(out, random_state=seed).reset_index(drop=True)

        return out

    for label in ["fraud", "normal"]:
        idx = out.index[out["type"] == label].to_numpy()
        if len(idx) == 0:
            continue

        k = int(np.floor(len(idx) * frac))
        chosen = (
            rng_global.choice(idx, size=k, replace=False)
            if k > 0 else np.array([], dtype=int)
        )

        if k == 0:
            continue

        noisy_texts = make_noisy_series(
            out.loc[chosen, "speech"].astype(str).tolist(),
            level=level,
            seed=seed + (0 if label == "fraud" else 1),
        )

        if mode == "replace":
            out.loc[chosen, "speech"] = noisy_texts
            out.loc[chosen, "is_noisy"] = True
        elif mode == "augment":
            add = out.loc[chosen].copy()
            add["speech"] = noisy_texts
            add["is_noisy"] = True
            frames.append(add)
        else:
            raise ValueError("mode must be 'replace' or 'augment'.")

    if mode == "augment":
        out = pd.concat(frames, ignore_index=True)
        out = sk_shuffle(out, random_state=seed).reset_index(drop=True)

    return out


def prepare_testing_dataset(input_csv: Path, output_dir: Path):
    """Create the mixed-noise 24k testing dataset."""
    df = pd.read_csv(input_csv)
    df = validate_labels(df)

    print("Testing input counts:", df["type"].value_counts().to_dict())

    mixed = add_noise_per_class(
        df,
        frac=0.5,
        mode="replace",
        level="mix",
    )

    output_path = output_dir / "calls_24k_mixed_with_noise.csv"
    mixed.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Saved: {output_path}")
    print("Final counts:", mixed["type"].value_counts().to_dict())
    print("Noise distribution:", mixed["is_noisy"].value_counts().to_dict())


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training and testing datasets for ALLaM fraud classification."
    )
    parser.add_argument(
        "--training-csv",
        type=Path,
        help="Path to the 100k fraud/normal training CSV.",
    )
    parser.add_argument(
        "--testing-csv",
        type=Path,
        help="Path to the clean 24k fraud/normal testing CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for processed output files.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.training_csv:
        prepare_training_dataset(args.training_csv, args.output_dir)

    if args.testing_csv:
        prepare_testing_dataset(args.testing_csv, args.output_dir)

    if not args.training_csv and not args.testing_csv:
        parser.error("Provide --training-csv and/or --testing-csv.")


if __name__ == "__main__":
    main()
