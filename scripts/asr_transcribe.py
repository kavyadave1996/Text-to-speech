#!/usr/bin/env python3
"""
ASR transcription script using faster-whisper.

Usage examples:

Single audio:
    python tools/asr_transcribe.py --audio "workspace/docs/Noisy_dataset/fishspeech_output_noisy/audio_no_clean_audio_fishspeech_hormuz.wav"

Single audio with output folder:
    python tools/asr_transcribe.py --audio "path/to/audio.wav" --out_dir "workspace/asr_outputs"

All wav files in a folder:
    python tools/asr_transcribe.py --audio_dir "workspace/docs/Noisy_dataset/fishspeech_output_noisy" --out_dir "workspace/asr_outputs"

GPU, int8:
    python tools/asr_transcribe.py --audio "path/to/audio.wav" --device cuda --compute_type int8

CPU, int8:
    python tools/asr_transcribe.py --audio "path/to/audio.wav" --device cpu --compute_type int8
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable, List, Dict, Optional

from faster_whisper import WhisperModel


AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma"}


def clean_filename(text: str) -> str:
    """Create a safe filename stem."""
    text = re.sub(r"[^\w\-]+", "_", text, flags=re.UNICODE)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:120] if text else "audio"


def collect_audio_files(audio: Optional[str], audio_dir: Optional[str]) -> List[Path]:
    """Collect one audio file or all audio files from a folder."""
    files: List[Path] = []

    if audio:
        p = Path(audio)
        if not p.exists():
            raise FileNotFoundError(f"Audio file not found: {p}")
        if not p.is_file():
            raise ValueError(f"--audio must point to a file: {p}")
        files.append(p)

    if audio_dir:
        d = Path(audio_dir)
        if not d.exists():
            raise FileNotFoundError(f"Audio folder not found: {d}")
        if not d.is_dir():
            raise ValueError(f"--audio_dir must point to a folder: {d}")

        for p in sorted(d.rglob("*")):
            if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS:
                files.append(p)

    if not files:
        raise ValueError("Provide either --audio or --audio_dir.")

    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for f in files:
        resolved = f.resolve()
        if resolved not in seen:
            unique_files.append(f)
            seen.add(resolved)

    return unique_files


def transcribe_file(
    model: WhisperModel,
    audio_path: Path,
    language: Optional[str],
    vad_filter: bool,
    beam_size: int,
) -> Dict[str, object]:
    """Transcribe one audio file and return transcript metadata."""
    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=beam_size,
        vad_filter=vad_filter,
        word_timestamps=False,
    )

    segment_rows = []
    transcript_parts = []

    for seg in segments:
        text = seg.text.strip()
        if text:
            transcript_parts.append(text)
        segment_rows.append(
            {
                "start": round(float(seg.start), 3),
                "end": round(float(seg.end), 3),
                "text": text,
            }
        )

    transcript = " ".join(transcript_parts).strip()

    return {
        "audio_path": str(audio_path),
        "language": getattr(info, "language", ""),
        "language_probability": round(float(getattr(info, "language_probability", 0.0)), 4),
        "duration_sec": round(float(getattr(info, "duration", 0.0)), 3),
        "transcript": transcript,
        "segments": segment_rows,
    }


def save_outputs(result: Dict[str, object], out_dir: Path) -> None:
    """Save transcript as TXT and segment-level CSV."""
    audio_path = Path(str(result["audio_path"]))
    safe_stem = clean_filename(audio_path.stem)

    txt_path = out_dir / f"{safe_stem}.txt"
    seg_csv_path = out_dir / f"{safe_stem}_segments.csv"

    txt_path.write_text(str(result["transcript"]) + "\n", encoding="utf-8")

    with seg_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["start", "end", "text"])
        writer.writeheader()
        writer.writerows(result["segments"])  # type: ignore[arg-type]


def write_master_csv(results: List[Dict[str, object]], out_dir: Path) -> Path:
    """Write one CSV containing one row per audio file."""
    master_csv = out_dir / "asr_transcripts_summary.csv"

    fieldnames = [
        "audio_path",
        "language",
        "language_probability",
        "duration_sec",
        "generated_text",
    ]

    with master_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            writer.writerow(
                {
                    "audio_path": r["audio_path"],
                    "language": r["language"],
                    "language_probability": r["language_probability"],
                    "duration_sec": r["duration_sec"],
                    "generated_text": r["transcript"],
                }
            )

    return master_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using faster-whisper and save TXT/CSV outputs."
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--audio", type=str, help="Path to one audio file.")
    input_group.add_argument("--audio_dir", type=str, help="Path to a folder containing audio files.")

    parser.add_argument(
        "--out_dir",
        type=str,
        default="workspace/asr_outputs",
        help="Output folder for transcripts. Default: workspace/asr_outputs",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="large-v2",
        help='Model name or local model path. Example: "large-v2" or "models/faster-whisper-large-v2".',
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "auto"],
        help="Device to use. Default: cuda",
    )
    parser.add_argument(
        "--compute_type",
        type=str,
        default="int8",
        help='Compute type. Common options: "int8", "int8_float16", "float16", "float32". Default: int8',
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help='Language code. Use "en" for English. Use "auto" for auto-detection.',
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=5,
        help="Beam size for decoding. Default: 5",
    )
    parser.add_argument(
        "--no_vad",
        action="store_true",
        help="Disable VAD filtering. By default VAD filtering is enabled.",
    )

    args = parser.parse_args()

    language = None if args.language.lower() == "auto" else args.language
    vad_filter = not args.no_vad

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_files = collect_audio_files(args.audio, args.audio_dir)

    print(f"Loading faster-whisper model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Compute type: {args.compute_type}")
    print(f"VAD filter: {vad_filter}")
    print(f"Number of audio files: {len(audio_files)}")

    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
    )

    results: List[Dict[str, object]] = []

    for i, audio_path in enumerate(audio_files, start=1):
        print(f"\n[{i}/{len(audio_files)}] Transcribing: {audio_path}")

        try:
            result = transcribe_file(
                model=model,
                audio_path=audio_path,
                language=language,
                vad_filter=vad_filter,
                beam_size=args.beam_size,
            )
            results.append(result)
            save_outputs(result, out_dir)

            print("Transcript:")
            print(result["transcript"])

        except Exception as e:
            print(f"ERROR while transcribing {audio_path}: {e}")

    master_csv = write_master_csv(results, out_dir)

    print("\nDone.")
    print(f"Transcript folder: {out_dir}")
    print(f"Summary CSV: {master_csv}")


if __name__ == "__main__":
    main()
