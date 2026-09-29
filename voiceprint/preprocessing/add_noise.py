"""
Add Synthetic Noise to Voiceprint Audio
Voice Fraud Detection System

Creates noisy speaker-verification datasets by mixing clean speech with
synthetic music-like, street-like, and cafe-like noise at a configurable
SNR range.

This script is adapted from the project's original preprocessing notebook.
It supports both 2-second and longer clips and can be reused for the
3–7 dB and 8–12 dB evaluation conditions.
"""

import argparse
import math
import random
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, resample_poly
from tqdm import tqdm


SAMPLE_RATE = 16000
RANDOM_SEED = 42
MIN_VALID_SEC = 0.5


def numeric_key_from_name(name: str):
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else 10**9


def list_speakers_numeric(root: Path):
    speakers = [path for path in root.iterdir() if path.is_dir()]
    return sorted(
        speakers,
        key=lambda path: (
            numeric_key_from_name(path.name),
            path.name.lower(),
        ),
    )


def list_audio_files_recursive(folder: Path):
    files = [
        path
        for path in folder.rglob("*")
        if path.is_file()
        and not path.name.startswith(".")
    ]

    def file_index(path: Path):
        match = re.search(
            r"(\d+)(?=\.[^.]+$)",
            path.name,
        )
        return int(match.group(1)) if match else 10**9

    return sorted(
        files,
        key=lambda path: (
            file_index(path),
            path.name.lower(),
        ),
    )


def resample_if_needed(
    signal: np.ndarray,
    input_sr: int,
    output_sr: int = SAMPLE_RATE,
):
    if input_sr == output_sr:
        return signal.astype(np.float32)

    return resample_poly(
        signal.astype(np.float32),
        up=output_sr,
        down=input_sr,
    ).astype(np.float32)


def load_audio_any(
    path: Path,
    sr: int = SAMPLE_RATE,
):
    """
    Load audio with soundfile when possible.
    Falls back to ffmpeg for unsupported formats.
    """
    try:
        signal, input_sr = sf.read(
            str(path),
            dtype="float32",
            always_2d=False,
        )

        if signal.ndim > 1:
            signal = signal.mean(axis=1)

        signal = resample_if_needed(
            signal,
            input_sr,
            sr,
        )

        return signal, sr

    except Exception:
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as tmp:
            tmp_path = tmp.name

        try:
            command = [
                "ffmpeg",
                "-y",
                "-i",
                str(path),
                "-ac",
                "1",
                "-ar",
                str(sr),
                "-f",
                "wav",
                tmp_path,
            ]

            subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )

            signal, _ = sf.read(
                tmp_path,
                dtype="float32",
            )

            if signal.ndim > 1:
                signal = signal.mean(axis=1)

            return (
                signal.astype(np.float32),
                sr,
            )

        finally:
            Path(tmp_path).unlink(
                missing_ok=True
            )


def save_wav(
    path: Path,
    data: np.ndarray,
    sr: int = SAMPLE_RATE,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sf.write(
        str(path),
        data.astype(np.float32),
        sr,
        subtype="PCM_16",
    )


def rms(signal: np.ndarray):
    return float(
        np.sqrt(
            np.mean(
                signal.astype(np.float32) ** 2
            )
            + 1e-12
        )
    )


def mix_at_snr(
    speech: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
):
    """
    Mix speech and noise at the requested SNR.
    """
    noise_rms = rms(noise)

    if noise_rms > 0:
        noise = noise / noise_rms

    speech_power = (
        np.mean(speech ** 2) + 1e-12
    )

    noise_power = (
        np.mean(noise ** 2) + 1e-12
    )

    alpha = math.sqrt(
        speech_power
        / (
            noise_power
            * (10 ** (snr_db / 10.0))
        )
    )

    mixed = speech + alpha * noise

    peak = max(
        1e-6,
        float(np.max(np.abs(mixed))),
    )

    if peak > 1.0:
        mixed = mixed / peak

    return mixed.astype(np.float32)


def butter_bandpass(
    low_hz: float,
    high_hz: float,
    sr: int,
    order: int = 4,
):
    nyquist = 0.5 * sr

    return butter(
        order,
        [
            low_hz / nyquist,
            high_hz / nyquist,
        ],
        btype="band",
    )


def bandpass(
    signal: np.ndarray,
    low_hz: float,
    high_hz: float,
    sr: int,
):
    b, a = butter_bandpass(
        low_hz,
        high_hz,
        sr,
    )

    return lfilter(
        b,
        a,
        signal,
    ).astype(np.float32)


def synth_music_like(
    length: int,
    sr: int = SAMPLE_RATE,
):
    noise = (
        np.random.randn(length)
        .astype(np.float32)
        * 0.3
    )

    time_axis = np.arange(length) / sr

    modulation = 0.5 * (
        1
        + np.sin(
            2
            * np.pi
            * np.random.uniform(1.5, 3.5)
            * time_axis
        )
    )

    signal = (
        noise * modulation
    ).astype(np.float32)

    return bandpass(
        signal,
        100,
        4000,
        sr,
    )


def synth_street_like(
    length: int,
    sr: int = SAMPLE_RATE,
):
    signal = (
        np.random.randn(length)
        .astype(np.float32)
        * 0.25
    )

    signal = bandpass(
        signal,
        20,
        6000,
        sr,
    )

    for _ in range(
        np.random.randint(1, 3)
    ):
        position = np.random.randint(
            0,
            max(1, length // 2),
        )

        burst_length = np.random.randint(
            int(0.05 * sr),
            int(0.15 * sr),
        )

        end = min(
            length,
            position + burst_length,
        )

        signal[position:end] += (
            np.hanning(end - position)
            * 0.12
        ).astype(np.float32)

    return np.clip(
        signal,
        -1.0,
        1.0,
    )


def synth_cafe_like(
    length: int,
    sr: int = SAMPLE_RATE,
):
    signal = (
        np.random.randn(length)
        .astype(np.float32)
        * 0.15
    )

    signal = bandpass(
        signal,
        200,
        800,
        sr,
    )

    return np.clip(
        signal,
        -1.0,
        1.0,
    )


NOISE_GENERATORS = {
    "music": synth_music_like,
    "street": synth_street_like,
    "cafe": synth_cafe_like,
}


def process_dataset(
    source_root: Path,
    output_root: Path,
    snr_low: float,
    snr_high: float,
    skip_if_exists: bool,
):
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    speakers = list_speakers_numeric(
        source_root
    )

    if not speakers:
        raise RuntimeError(
            f"No speaker folders found in "
            f"{source_root}"
        )

    total_written = 0

    for speaker_dir in tqdm(
        speakers,
        desc="Speakers",
        unit="speaker",
    ):
        speaker_name = speaker_dir.name
        files = list_audio_files_recursive(
            speaker_dir
        )

        if not files:
            continue

        for audio_path in tqdm(
            files,
            leave=False,
            desc=speaker_name,
            unit="file",
        ):
            try:
                speech, sr = load_audio_any(
                    audio_path
                )
            except Exception:
                continue

            if (
                len(speech)
                < int(
                    MIN_VALID_SEC
                    * SAMPLE_RATE
                )
            ):
                continue

            for noise_name, generator in (
                NOISE_GENERATORS.items()
            ):
                noise = generator(
                    len(speech),
                    sr,
                )

                snr_db = float(
                    np.random.uniform(
                        snr_low,
                        snr_high,
                    )
                )

                mixed = mix_at_snr(
                    speech,
                    noise,
                    snr_db,
                )

                output_path = (
                    output_root
                    / noise_name
                    / speaker_name
                    / (
                        f"{audio_path.stem}_"
                        f"{noise_name}.wav"
                    )
                )

                if (
                    skip_if_exists
                    and output_path.exists()
                ):
                    continue

                save_wav(
                    output_path,
                    mixed,
                    sr,
                )

                total_written += 1

    print(
        f"\n[done] Wrote "
        f"{total_written} noisy files."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Add synthetic noise to "
            "voiceprint audio at a "
            "configurable SNR range."
        )
    )

    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help=(
            "Clean speaker dataset "
            "(one folder per speaker)."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help=(
            "Output root for noisy "
            "speaker datasets."
        ),
    )

    parser.add_argument(
        "--snr-low",
        type=float,
        required=True,
        help="Lower SNR bound in dB.",
    )

    parser.add_argument(
        "--snr-high",
        type=float,
        required=True,
        help="Upper SNR bound in dB.",
    )

    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        help=(
            "Skip files that already exist."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    process_dataset(
        source_root=args.source_root,
        output_root=args.output_root,
        snr_low=args.snr_low,
        snr_high=args.snr_high,
        skip_if_exists=(
            args.skip_if_exists
        ),
    )


if __name__ == "__main__":
    main()
