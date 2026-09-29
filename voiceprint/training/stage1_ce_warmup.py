"""
Stage 1 — ECAPA-TDNN Cross-Entropy Warm-up
Voice Fraud Detection System

Purpose:
Adapt a pretrained SpeechBrain ECAPA-TDNN speaker encoder to the project
speaker dataset using a temporary speaker-classification head and
cross-entropy loss.

The classification head is used only as a training objective. The final
speaker-verification stage uses ECAPA embeddings and similarity scoring.
"""

import argparse
import glob
import random
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from speechbrain.pretrained import EncoderClassifier


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------
# Default training configuration
# ---------------------------------------------------------------------
SR = 16000
TRAIN_CROP_SEC = 2.0
VAL_CROP_SEC = 2.0

BATCH_TRAIN = 8
BATCH_VAL = 32
EPOCHS = 40
LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 5

FREEZE_FIRST_N_BLOCKS = 2


# ---------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------
def list_speakers(root):
    """Return {speaker_name: [audio_files]} from a speaker-folder dataset."""
    root = Path(root)
    spk_dirs = sorted([p for p in root.iterdir() if p.is_dir()])

    spk2files = {}
    for directory in spk_dirs:
        files = []
        for ext in (".wav", ".flac", ".mp3"):
            files.extend(glob.glob(str(directory / f"*{ext}")))
        files.sort()

        if files:
            spk2files[directory.name] = files

    return spk2files


def build_items(spk2files):
    """Create speaker IDs and (audio_path, speaker_id) training items."""
    spk2id = {speaker: i for i, speaker in enumerate(sorted(spk2files))}
    items = []

    for speaker, files in spk2files.items():
        speaker_id = spk2id[speaker]
        for file_path in files:
            items.append((file_path, speaker_id))

    return spk2id, items


def load_wav_16k(path, target_sr=SR):
    """Load mono audio and resample it to 16 kHz."""
    wav, sr = torchaudio.load(path)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    return wav


def random_crop(wav, crop_sec):
    """Randomly crop or pad audio to the requested duration."""
    required = int(crop_sec * SR)
    total = wav.shape[-1]

    if total == required:
        return wav

    if total > required:
        start = random.randint(0, total - required)
        return wav[:, start : start + required]

    return F.pad(wav, (0, required - total))


def center_crop_or_pad(wav, crop_sec):
    """Center-crop or pad audio for deterministic validation."""
    required = int(crop_sec * SR)
    total = wav.shape[-1]

    if total >= required:
        start = (total - required) // 2
        return wav[:, start : start + required]

    pad = required - total
    return F.pad(wav, (pad // 2, pad - pad // 2))


def add_noise_snr(wav, snr_db):
    """Add Gaussian noise at a target SNR."""
    signal_power = wav.pow(2).mean().clamp_min(1e-9)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(wav) * torch.sqrt(noise_power)
    return wav + noise


def random_gain(wav, low_db=-3.0, high_db=3.0):
    """Apply random gain augmentation."""
    gain = 10.0 ** (random.uniform(low_db, high_db) / 20.0)
    return wav * gain


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
class SpeakerDataset(Dataset):
    def __init__(self, items, split="train", crop_sec=2.0, augment=False):
        self.items = items
        self.split = split
        self.crop_sec = crop_sec
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        wav = load_wav_16k(path)

        if self.split == "train":
            wav = random_crop(wav, self.crop_sec)

            if self.augment:
                if random.random() < 0.6:
                    wav = add_noise_snr(wav, random.uniform(7.0, 12.0))

                if random.random() < 0.5:
                    wav = random_gain(wav, -3.0, 3.0)
        else:
            wav = center_crop_or_pad(wav, self.crop_sec)

        return wav.squeeze(0), label


def collate_fn(batch):
    wavs, labels = zip(*batch)

    lengths = torch.tensor([w.shape[-1] for w in wavs], dtype=torch.long)
    max_len = int(lengths.max().item())

    padded = [
        w if w.shape[-1] == max_len else F.pad(w, (0, max_len - w.shape[-1]))
        for w in wavs
    ]

    wavs = torch.stack(padded, dim=0).unsqueeze(1)
    relative_lengths = (lengths.float() / max_len).clamp(0, 1)
    labels = torch.tensor(labels, dtype=torch.long)

    return wavs, relative_lengths, labels


# ---------------------------------------------------------------------
# Temporary classification head used for CE warm-up
# ---------------------------------------------------------------------
class Classifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(self.dropout(self.norm(x)))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 1 ECAPA-TDNN cross-entropy warm-up."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/voiceprint/train"),
        help="Speaker-folder training dataset.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/voiceprint/ce_warmup"),
        help="Directory for saved checkpoints.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # Load pretrained ECAPA-TDNN
    # -------------------------------------------------------------
    sb_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": DEVICE},
    )

    compute_features = sb_model.mods.compute_features.to(DEVICE)
    mean_var_norm = sb_model.mods.mean_var_norm.to(DEVICE)
    embedding_model = sb_model.mods.embedding_model.to(DEVICE)
    embedding_dim = getattr(embedding_model, "out_channels", 192)

    # -------------------------------------------------------------
    # Build file-level train/validation split
    # -------------------------------------------------------------
    spk2files = list_speakers(args.data_root)
    num_speakers = len(spk2files)
    assert num_speakers >= 2, "At least two speakers are required."

    spk2id, all_items = build_items(spk2files)

    n_items = len(all_items)
    val_ratio = 0.1
    val_size = max(1, int(n_items * val_ratio))

    indices = list(range(n_items))
    random.shuffle(indices)

    val_idx = set(indices[:val_size])
    train_items = [all_items[i] for i in range(n_items) if i not in val_idx]
    val_items = [all_items[i] for i in range(n_items) if i in val_idx]

    train_ds = SpeakerDataset(
        train_items,
        split="train",
        crop_sec=TRAIN_CROP_SEC,
        augment=True,
    )
    val_ds = SpeakerDataset(
        val_items,
        split="val",
        crop_sec=VAL_CROP_SEC,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_TRAIN,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_VAL,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    # -------------------------------------------------------------
    # Classification head and partial ECAPA freezing
    # -------------------------------------------------------------
    head = Classifier(embedding_dim, num_speakers).to(DEVICE)

    frozen = 0
    if hasattr(embedding_model, "blocks") and isinstance(
        embedding_model.blocks, nn.ModuleList
    ):
        for i, block in enumerate(embedding_model.blocks):
            if i < FREEZE_FIRST_N_BLOCKS:
                for parameter in block.parameters():
                    parameter.requires_grad = False
                frozen += 1
    else:
        # Fallback kept to preserve the original experiment behavior.
        params = list(embedding_model.parameters())
        k = int(0.3 * len(params))
        for parameter in params[:k]:
            parameter.requires_grad = False
        frozen = -1

    backbone_params = [
        parameter
        for parameter in embedding_model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": LR * 0.5},
            {"params": head.parameters(), "lr": LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    ce_loss = nn.CrossEntropyLoss()

    def forward_to_embedding(wavs, relative_lengths):
        features = compute_features(wavs.squeeze(1))
        features = mean_var_norm(features, relative_lengths)

        embedding = embedding_model(features, relative_lengths)

        if embedding.dim() == 3 and embedding.size(1) == 1:
            embedding = embedding.squeeze(1)

        return F.normalize(embedding, p=2, dim=-1)

    def run_epoch(loader, train=True):
        if train:
            embedding_model.train()
            head.train()
        else:
            embedding_model.eval()
            head.eval()

        total_loss = 0.0
        total_correct = 0
        total = 0

        for wavs, relative_lengths, labels in tqdm(loader, disable=False):
            wavs = wavs.to(DEVICE, non_blocking=True)
            relative_lengths = relative_lengths.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True).view(-1).long()

            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(train):
                embeddings = forward_to_embedding(wavs, relative_lengths)
                logits = head(embeddings)
                loss = ce_loss(logits, labels)

                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(embedding_model.parameters()) + list(head.parameters()),
                        5.0,
                    )
                    optimizer.step()

            total_loss += float(loss.item()) * labels.size(0)
            total_correct += int(
                (logits.argmax(dim=-1) == labels).sum().item()
            )
            total += labels.size(0)

        return total_loss / total, total_correct / total

    # -------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------
    best_val = float("inf")
    no_improve = 0
    best_path = args.checkpoint_dir / "ecapa_ce_warmup_best.pt"

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(train_loader, train=True)
        val_loss, val_acc = run_epoch(val_loader, train=False)

        scheduler.step(val_loss)

        print(
            f"[{epoch:02d}/{EPOCHS}] "
            f"train_loss={train_loss:.4f} "
            f"acc={train_acc * 100:.2f}% | "
            f"val_loss={val_loss:.4f} "
            f"acc={val_acc * 100:.2f}%"
        )

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            no_improve = 0

            checkpoint = {
                "epoch": epoch,
                "embedding_model": embedding_model.state_dict(),
                "head": head.state_dict(),
                "num_speakers": num_speakers,
                "sr": SR,
                "embedding_dim": embedding_dim,
                "spk2id": spk2id,
                "config": {
                    "train_crop_sec": TRAIN_CROP_SEC,
                    "val_crop_sec": VAL_CROP_SEC,
                    "frozen_blocks": frozen,
                    "lr": LR,
                    "batch_train": BATCH_TRAIN,
                    "batch_val": BATCH_VAL,
                },
            }

            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint to: {best_path}")
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch} "
                f"(no improvement for {PATIENCE} epochs)."
            )
            break

    print("Stage 1 CE warm-up completed.")


if __name__ == "__main__":
    main()
