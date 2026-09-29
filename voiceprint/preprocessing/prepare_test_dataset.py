"""
Prepare Clean Voiceprint Test Dataset
Voice Fraud Detection System

Creates fixed-length clean speech clips (2 s or 5 s) from source audio/video
files for speaker-verification evaluation.

The script:
- converts media to 16 kHz mono PCM audio
- segments each source into fixed-length clips
- uses Faster-Whisper + VAD to keep speech-containing segments
- optionally uses an overlapping second pass if more clips are needed
- stores metadata for every saved clip
"""

import argparse
import io
import json
import os
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from pydub import AudioSegment


SUPPORTED_MEDIA = {
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg",
    ".opus", ".wma", ".mp4", ".mkv", ".mov", ".webm",
}

SAMPLE_RATE = 16000
LANGUAGE = "ar"
MODEL_SIZE = "small"
COMPUTE_TYPE = "int8"

MIN_TEXT_CHARS = 2
MIN_AVG_PROB = 0.40
USE_VAD_FILTER = True


def load_media_to_audio16k_mono(path: Path) -> AudioSegment:
    """Load audio/video and normalize it to 16 kHz mono 16-bit audio."""
    audio = AudioSegment.from_file(path)
    return (
        audio
        .set_frame_rate(SAMPLE_RATE)
        .set_channels(1)
        .set_sample_width(2)
    )


def segment_starts(total_ms: int, window_ms: int, hop_ms: int) -> List[int]:
    """Return valid segment start positions."""
    return list(
        range(
            0,
            max(0, total_ms - window_ms + 1),
            hop_ms,
        )
    )


def export_segment_to_wav_bytes(segment: AudioSegment) -> bytes:
    """Export an AudioSegment to in-memory WAV bytes."""
    buffer = io.BytesIO()
    samples = np.array(
        segment.get_array_of_samples(),
        dtype=np.int16,
    )
    sf.write(
        buffer,
        samples,
        segment.frame_rate,
        format="WAV",
        subtype="PCM_16",
    )
    return buffer.getvalue()


def transcribe_chunk_bytes(
    asr_model: WhisperModel,
    wav_bytes: bytes,
) -> Tuple[str, float]:
    """Transcribe one clip and return text plus approximate confidence."""
    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False,
    ) as tmp:
        tmp.write(wav_bytes)
        tmp.flush()
        tmp_path = tmp.name

    try:
        segments, _ = asr_model.transcribe(
            tmp_path,
            language=LANGUAGE,
            task="transcribe",
            vad_filter=USE_VAD_FILTER,
            vad_parameters={
                "min_silence_duration_ms": 300,
            },
            beam_size=1,
            best_of=1,
            word_timestamps=False,
        )

        text_parts = []
        probabilities = []

        for segment in segments:
            if segment.text:
                text_parts.append(
                    segment.text.strip()
                )

            if (
                hasattr(segment, "avg_logprob")
                and segment.avg_logprob is not None
            ):
                probability = max(
                    0.0,
                    min(
                        1.0,
                        1.0 + (
                            segment.avg_logprob / 5.0
                        ),
                    ),
                )
                probabilities.append(probability)

        text = " ".join(
            part
            for part in text_parts
            if part
        )

        average_probability = (
            sum(probabilities) / len(probabilities)
            if probabilities
            else (1.0 if text else 0.0)
        )

        return text, float(average_probability)

    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def looks_like_speech(text: str) -> bool:
    """Check that the ASR output contains enough alphabetic speech content."""
    cleaned = "".join(
        char
        for char in text
        if char.isalpha() or char.isspace()
    )
    return len(cleaned.strip()) >= MIN_TEXT_CHARS


def process_one_file(
    media_path: Path,
    output_root: Path,
    asr_model: WhisperModel,
    segment_seconds: int,
    target_per_file: int,
) -> int:
    """Create accepted speech clips from one input media file."""
    segment_ms = segment_seconds * 1000
    primary_hop_ms = segment_ms
    secondary_hop_ms = segment_ms // 2

    print(f"\n[+] Processing: {media_path.name}")

    audio = load_media_to_audio16k_mono(
        media_path
    )
    total_ms = len(audio)

    output_dir = (
        output_root / media_path.stem
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = []
    saved = 0

    def try_save(start_ms: int) -> bool:
        nonlocal saved

        segment = audio[
            start_ms:start_ms + segment_ms
        ]

        wav_bytes = (
            export_segment_to_wav_bytes(
                segment
            )
        )

        text, avg_prob = (
            transcribe_chunk_bytes(
                asr_model,
                wav_bytes,
            )
        )

        if (
            not looks_like_speech(text)
            or avg_prob < MIN_AVG_PROB
        ):
            return False

        output_path = (
            output_dir
            / (
                f"{media_path.stem}_"
                f"{segment_seconds}s_"
                f"{saved + 1:03d}.wav"
            )
        )

        with output_path.open("wb") as file:
            file.write(wav_bytes)

        metadata.append(
            {
                "file": output_path.name,
                "start_ms": start_ms,
                "duration_ms": segment_ms,
                "text": text,
                "avg_prob": round(
                    avg_prob,
                    3,
                ),
            }
        )

        saved += 1
        return True

    # First pass: no overlap.
    primary_starts = segment_starts(
        total_ms,
        segment_ms,
        primary_hop_ms,
    )

    for start_ms in primary_starts:
        if saved >= target_per_file:
            break
        try_save(start_ms)

    # Second pass: 50% overlap if more clips are needed.
    if saved < target_per_file:
        secondary_starts = segment_starts(
            total_ms,
            segment_ms,
            secondary_hop_ms,
        )

        primary_set = set(primary_starts)

        for start_ms in secondary_starts:
            if saved >= target_per_file:
                break

            if start_ms in primary_set:
                continue

            try_save(start_ms)

    metadata_path = (
        output_dir
        / f"{media_path.stem}_meta.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "source": str(media_path),
                "sample_rate": SAMPLE_RATE,
                "segment_seconds": (
                    segment_seconds
                ),
                "target_count": (
                    target_per_file
                ),
                "saved_count": saved,
                "clips": metadata,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"   Saved {saved}/"
        f"{target_per_file} clips -> "
        f"{output_dir}"
    )

    return saved


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare clean voiceprint "
            "speaker-verification test clips."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help=(
            "Folder containing source "
            "audio/video files."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Output root; one speaker folder "
            "is created per source file."
        ),
    )

    parser.add_argument(
        "--segment-seconds",
        type=int,
        choices=[2, 5],
        default=2,
        help=(
            "Clip duration used by the "
            "evaluation dataset."
        ),
    )

    parser.add_argument(
        "--target-per-file",
        type=int,
        default=50,
        help=(
            "Maximum number of accepted "
            "speech clips per source."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    asr_model = WhisperModel(
        MODEL_SIZE,
        device="auto",
        compute_type=COMPUTE_TYPE,
    )

    media_files = [
        path
        for path in args.input_dir.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_MEDIA
        )
    ]

    if not media_files:
        raise RuntimeError(
            f"No supported media files "
            f"found in {args.input_dir}"
        )

    total_saved = 0

    for media_path in sorted(
        media_files
    ):
        total_saved += process_one_file(
            media_path=media_path,
            output_root=args.output_dir,
            asr_model=asr_model,
            segment_seconds=(
                args.segment_seconds
            ),
            target_per_file=(
                args.target_per_file
            ),
        )

    print(
        "\n[done] Total saved clips: "
        f"{total_saved}"
    )


if __name__ == "__main__":
    main()
