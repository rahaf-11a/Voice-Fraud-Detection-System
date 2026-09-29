"""
Stage 2 — ECAPA-TDNN AAM-Softmax Fine-Tuning
Voice Fraud Detection System

Continues fine-tuning from the Stage 1 CE warm-up checkpoint using
Additive Angular Margin (AAM) loss to improve separation between
speaker embeddings.
"""

import argparse
import glob
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from speechbrain.pretrained import EncoderClassifier

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

SR = 16000
TRAIN_CROP_SEC = 2.0
VAL_CROP_SEC = 2.0
BATCH_TRAIN = 8
BATCH_VAL = 32
EPOCHS = 30
PATIENCE = 5

LR_WARMUP_HEAD = 1e-3
LR_BACKBONE = 5e-5
LR_HEAD = 5e-4
WEIGHT_DECAY = 1e-4

FREEZE_FIRST_N_BLOCKS = 2
AAM_SCALE = 30.0
AAM_MARGIN = 0.2


def list_speakers(root):
    root = Path(root)
    speaker_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    speaker_to_files = {}
    for d in speaker_dirs:
        files = []
        for ext in (".wav", ".flac", ".mp3"):
            files.extend(glob.glob(str(d / f"*{ext}")))
        files.sort()
        if files:
            speaker_to_files[d.name] = files
    return speaker_to_files


def build_items(speaker_to_files):
    speaker_to_id = {
        speaker: i for i, speaker in enumerate(sorted(speaker_to_files))
    }
    items = []
    for speaker, files in speaker_to_files.items():
        sid = speaker_to_id[speaker]
        for f in files:
            items.append((f, sid))
    return speaker_to_id, items


def load_wav_16k(path, target_sr=SR):
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav.T)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


def random_crop(wav, crop_sec):
    need = int(crop_sec * SR)
    total = wav.shape[-1]
    if total == need:
        return wav
    if total > need:
        start = random.randint(0, total - need)
        return wav[:, start:start + need]
    return F.pad(wav, (0, need - total))


def center_crop_or_pad(wav, crop_sec):
    need = int(crop_sec * SR)
    total = wav.shape[-1]
    if total >= need:
        start = (total - need) // 2
        return wav[:, start:start + need]
    pad = need - total
    return F.pad(wav, (pad // 2, pad - pad // 2))


def add_noise_snr(wav, snr_db):
    signal_power = wav.pow(2).mean().clamp_min(1e-9)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(wav) * torch.sqrt(noise_power)
    return wav + noise


def random_gain(wav, low_db=-3.0, high_db=3.0):
    gain = 10.0 ** (random.uniform(low_db, high_db) / 20.0)
    return wav * gain


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


class AAMSoftmax(nn.Module):
    def __init__(self, in_dim, num_classes, scale=30.0, margin=0.2):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.randn(num_classes, in_dim))
        nn.init.xavier_normal_(self.weight)

    def forward(self, embeddings, labels=None):
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        weights = F.normalize(self.weight, p=2, dim=-1)
        cosine = torch.matmul(embeddings, weights.t())

        if labels is None:
            return self.scale * cosine

        theta = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))
        target_cosine = torch.cos(theta + self.margin)
        one_hot = F.one_hot(labels, num_classes=weights.size(0)).float().to(
            embeddings.device
        )
        logits = cosine * (1 - one_hot) + target_cosine * one_hot
        return self.scale * logits


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 2 ECAPA-TDNN AAM-Softmax fine-tuning."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/voiceprint/train"),
    )
    parser.add_argument(
        "--ce-checkpoint",
        type=Path,
        default=Path(
            "checkpoints/voiceprint/ce_warmup/ecapa_ce_warmup_best.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/voiceprint/aam_finetuning"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "ecapa_aam_best.pt"

    sb_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": DEVICE},
    )

    compute_features = sb_model.mods.compute_features.to(DEVICE)
    mean_var_norm = sb_model.mods.mean_var_norm.to(DEVICE)
    embedding_model = sb_model.mods.embedding_model.to(DEVICE)
    embedding_dim = getattr(embedding_model, "out_channels", 192)

    speaker_to_files = list_speakers(args.data_root)
    num_speakers = len(speaker_to_files)
    assert num_speakers >= 2, "At least two speakers are required."

    speaker_to_id, all_items = build_items(speaker_to_files)

    n_items = len(all_items)
    val_size = max(1, int(n_items * 0.1))
    indices = list(range(n_items))
    random.shuffle(indices)

    val_indices = set(indices[:val_size])
    train_items = [all_items[i] for i in range(n_items) if i not in val_indices]
    val_items = [all_items[i] for i in range(n_items) if i in val_indices]

    train_loader = DataLoader(
        SpeakerDataset(train_items, "train", TRAIN_CROP_SEC, True),
        batch_size=BATCH_TRAIN,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    val_loader = DataLoader(
        SpeakerDataset(val_items, "val", VAL_CROP_SEC, False),
        batch_size=BATCH_VAL,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    ce_checkpoint = torch.load(args.ce_checkpoint, map_location=DEVICE)
    embedding_model.load_state_dict(
        ce_checkpoint["embedding_model"],
        strict=True,
    )

    head = AAMSoftmax(
        embedding_dim,
        num_speakers,
        scale=AAM_SCALE,
        margin=AAM_MARGIN,
    ).to(DEVICE)

    use_amp = DEVICE == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def forward_to_embedding(wavs, relative_lengths):
        features = compute_features(wavs.squeeze(1))
        features = mean_var_norm(features, relative_lengths)
        embeddings = embedding_model(features, relative_lengths)

        if embeddings.dim() == 3 and embeddings.size(1) == 1:
            embeddings = embeddings.squeeze(1)

        return F.normalize(embeddings, p=2, dim=-1)

    optimizer = None

    def run_epoch(loader, train=True, description=""):
        nonlocal optimizer

        if train:
            embedding_model.train()
            head.train()
        else:
            embedding_model.eval()
            head.eval()

        total_loss = 0.0
        total_correct = 0
        total = 0

        for wavs, relative_lengths, labels in tqdm(
            loader,
            desc=description,
            leave=False,
        ):
            wavs = wavs.to(DEVICE, non_blocking=True)
            relative_lengths = relative_lengths.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True).view(-1).long()

            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(train):
                with torch.cuda.amp.autocast(enabled=use_amp):
                    embeddings = forward_to_embedding(
                        wavs,
                        relative_lengths,
                    )
                    logits = head(embeddings, labels)
                    loss = F.cross_entropy(logits, labels)

            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(embedding_model.parameters()) + list(head.parameters()),
                    5.0,
                )
                scaler.step(optimizer)
                scaler.update()

            total_loss += float(loss.item()) * labels.size(0)
            total_correct += int(
                (logits.argmax(dim=-1) == labels).sum().item()
            )
            total += labels.size(0)

        return total_loss / total, total_correct / total

    # Head warm-up for one epoch.
    for parameter in embedding_model.parameters():
        parameter.requires_grad = False

    optimizer = torch.optim.AdamW(
        [{"params": head.parameters(), "lr": LR_WARMUP_HEAD}],
        weight_decay=WEIGHT_DECAY,
    )

    warmup_train_loss, warmup_train_acc = run_epoch(
        train_loader,
        train=True,
        description="warmup(head)",
    )

    warmup_val_loss, warmup_val_acc = run_epoch(
        val_loader,
        train=False,
        description="val(warmup)",
    )

    tqdm.write(
        "[WARMUP] "
        f"train_loss={warmup_train_loss:.4f} "
        f"acc={warmup_train_acc * 100:.2f}% | "
        f"val_loss={warmup_val_loss:.4f} "
        f"acc={warmup_val_acc * 100:.2f}%"
    )

    # Unfreeze backbone and retain the original partial-freezing strategy.
    for parameter in embedding_model.parameters():
        parameter.requires_grad = True

    frozen = 0
    if hasattr(embedding_model, "blocks") and isinstance(
        embedding_model.blocks,
        nn.ModuleList,
    ):
        for i, block in enumerate(embedding_model.blocks):
            if i < FREEZE_FIRST_N_BLOCKS:
                for parameter in block.parameters():
                    parameter.requires_grad = False
                frozen += 1
    else:
        frozen = -1

    backbone_params = [
        p for p in embedding_model.parameters() if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": LR_BACKBONE},
            {"params": head.parameters(), "lr": LR_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    best_val = float("inf")
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(
            train_loader,
            train=True,
            description=f"train e{epoch:02d}",
        )

        val_loss, val_acc = run_epoch(
            val_loader,
            train=False,
            description=f"val e{epoch:02d}",
        )

        scheduler.step(val_loss)

        tqdm.write(
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
                "spk2id": speaker_to_id,
                "config": {
                    "train_crop_sec": TRAIN_CROP_SEC,
                    "val_crop_sec": VAL_CROP_SEC,
                    "frozen_blocks": frozen,
                    "lr_backbone": LR_BACKBONE,
                    "lr_head": LR_HEAD,
                    "batch_train": BATCH_TRAIN,
                    "batch_val": BATCH_VAL,
                    "aam_scale": head.scale,
                    "aam_margin": head.margin,
                },
            }

            torch.save(checkpoint, best_path)
            tqdm.write(f"Saved best checkpoint to: {best_path}")
        else:
            no_improve += 1

            if no_improve >= PATIENCE:
                tqdm.write(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {PATIENCE} epochs)."
                )
                break

    print("Stage 2 AAM fine-tuning completed.")


if __name__ == "__main__":
    main()
