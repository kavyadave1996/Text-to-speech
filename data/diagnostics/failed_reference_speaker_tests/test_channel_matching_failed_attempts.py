#!/usr/bin/env python3
"""
Channel-matching diagnostic for failed maritime TTS speaker-verification attempts.

Run from the Chatterbox repository:
    /data/kavya/conda_envs/chatterbox

Purpose
-------
The failed TTS attempts are often much cleaner than the real noisy/VHF reference
recordings. This script tests whether that channel mismatch is contributing to
speaker-verification failures.

For every failed generated attempt it:
  1. Reads the failed attempt from failed_attempt_tests_2_3.csv.
  2. Finds the manually checked reference Chunk A and Chunk B.
  3. Re-scores the ORIGINAL generated attempt against A, B and the full reference.
  4. Creates a temporary radio-like band-limited copy of the generated attempt.
  5. Re-scores the FILTERED copy against A, B and the full reference.
  6. Records whether any decision changes from DIFFERENT -> SAME.

The original generated WAV is NEVER overwritten.

Default diagnostic filter
-------------------------
A simple 300-3400 Hz band-pass is used as a controlled first test. This is not
claimed to be a full VHF-channel simulation; it is only a diagnostic to reduce
one obvious spectral mismatch between clean TTS and radio-like references.

Inputs
------
workspace/data/diagnostics/failed_reference_speaker_tests/
    failed_attempt_tests_2_3.csv
    reference_test1_summary.csv
    references/<reference_slug>/chunk_A.wav
    references/<reference_slug>/chunk_B.wav

Existing backend
----------------
workspace/models/speaker_backend_3way.pkl

Outputs
-------
workspace/data/diagnostics/channel_matching/
    channel_matching_results.csv
    channel_matching_summary.csv
    filtered_audio/*.wav
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import pandas as pd
import torch
import torchaudio as ta


REPO_ROOT = Path(__file__).resolve().parent

# Allow the script to work when project Python files are either in repository
# root (data-station layout) or under workspace/docs/tools (Windows Git layout).
try:
    from speaker_backend_3way import ThreeWaySpeakerBackend
except ImportError:
    tools_dir = REPO_ROOT / "workspace" / "docs" / "tools"
    sys.path.insert(0, str(tools_dir))
    from speaker_backend_3way import ThreeWaySpeakerBackend

try:
    from run_pilot_generation_dialogue import SpeakerEmbeddingVerifier
except ImportError:
    tools_dir = REPO_ROOT / "workspace" / "docs" / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    from run_pilot_generation_dialogue import SpeakerEmbeddingVerifier


SCORE_KEYS = [
    "raw_cosine",
    "raw_plda_score",
    "raw_cohort_snorm",
    "calibrated_cosine_llr",
    "calibrated_plda_llr",
    "calibrated_cohort_llr",
    "calibrated_cosine_same",
    "plda_same",
    "cohort_same",
    "same_votes",
    "decision_same",
]


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def resolve_repo_path(value) -> Path | None:
    value = clean_text(value)
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def repo_relative(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def safe_slug(path: Path) -> str:
    stem = Path(path).stem
    slug = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").lower()
    return slug[:100] or "reference"


def flatten_score(prefix: str, result: dict) -> dict:
    return {f"{prefix}_{key}": result.get(key) for key in SCORE_KEYS}


def require_file(path: Path, description: str = "file") -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}:\n{path}")


def load_mono(path: Path):
    wav, sr = ta.load(str(path))
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    elif wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.to(torch.float32), int(sr)


def radio_bandpass(wav: torch.Tensor, sr: int, low_hz: float, high_hz: float):
    """Apply a simple diagnostic band-pass without modifying the source file."""
    nyquist = sr / 2.0

    # Keep cutoff safely below Nyquist for low-sample-rate files.
    effective_high = min(float(high_hz), nyquist * 0.90)
    effective_low = max(20.0, float(low_hz))

    if effective_low >= effective_high:
        raise ValueError(
            f"Invalid effective band for sample rate {sr}: "
            f"{effective_low:.1f}-{effective_high:.1f} Hz"
        )

    filtered = ta.functional.highpass_biquad(
        wav,
        sample_rate=sr,
        cutoff_freq=effective_low,
    )
    filtered = ta.functional.lowpass_biquad(
        filtered,
        sample_rate=sr,
        cutoff_freq=effective_high,
    )

    # Filtering can create a small overshoot. Prevent clipping only if needed.
    peak = float(filtered.abs().max()) if filtered.numel() else 0.0
    if peak > 0.999:
        filtered = filtered * (0.98 / peak)

    return filtered, effective_low, effective_high


def find_reference_chunks(
    reference_audio: Path,
    reference_map: dict[str, tuple[Path, Path]],
    failed_diag_dir: Path,
) -> tuple[Path, Path, str]:
    """
    Prefer paths recorded in reference_test1_summary.csv.
    Fall back to the deterministic folder naming used by diagnose_failed_samples.py.
    """
    key_variants = {
        str(reference_audio),
        str(reference_audio.resolve()),
        repo_relative(reference_audio),
    }

    for key in key_variants:
        if key in reference_map:
            a, b = reference_map[key]
            return a, b, "reference_test1_summary.csv"

    chunk_dir = failed_diag_dir / "references" / safe_slug(reference_audio)
    return chunk_dir / "chunk_A.wav", chunk_dir / "chunk_B.wav", "derived_from_reference_slug"


def ab_label(decision_a: bool, decision_b: bool) -> str:
    if decision_a and decision_b:
        return "MATCH_BOTH"
    if decision_a or decision_b:
        return "MATCH_ONE_BORDERLINE"
    return "DIFFERENT_BOTH"


def transition(before: bool, after: bool) -> str:
    if (not before) and after:
        return "DIFFERENT_TO_SAME"
    if before and (not after):
        return "SAME_TO_DIFFERENT"
    if before and after:
        return "SAME_TO_SAME"
    return "DIFFERENT_TO_DIFFERENT"


def stable_filtered_name(row: pd.Series, attempt_audio: Path) -> str:
    sample_id = re.sub(r"[^A-Za-z0-9_-]+", "_", clean_text(row.get("sample_id", "sample")))
    speaker = re.sub(r"[^A-Za-z0-9_-]+", "_", clean_text(row.get("speaker", "speaker")))
    turn_id = clean_text(row.get("turn_id", "x")) or "x"
    attempt_number = clean_text(row.get("attempt_number", "x")) or "x"

    # Prevent collisions if multiple rows happen to share the same visible IDs.
    digest = hashlib.sha1(str(attempt_audio).encode("utf-8")).hexdigest()[:8]
    return f"{sample_id}_{speaker}_turn{turn_id}_attempt{attempt_number}_{digest}_radio.wav"


def build_reference_map(reference_summary_csv: Path) -> dict[str, tuple[Path, Path]]:
    if not reference_summary_csv.exists():
        return {}

    df = pd.read_csv(reference_summary_csv)
    required = {"reference_audio", "chunk_a", "chunk_b"}
    if not required.issubset(df.columns):
        print(
            f"[WARN] {reference_summary_csv} does not contain "
            "reference_audio/chunk_a/chunk_b; using folder-name fallback."
        )
        return {}

    mapping: dict[str, tuple[Path, Path]] = {}
    for _, row in df.iterrows():
        ref = resolve_repo_path(row["reference_audio"])
        a = resolve_repo_path(row["chunk_a"])
        b = resolve_repo_path(row["chunk_b"])
        if ref is None or a is None or b is None:
            continue

        value = (a, b)
        mapping[str(ref)] = value
        mapping[str(ref.resolve())] = value
        mapping[repo_relative(ref)] = value

    return mapping


def main():
    parser = argparse.ArgumentParser(
        description="Test whether radio/channel mismatch contributes to failed speaker-QC attempts."
    )
    parser.add_argument(
        "--input-csv",
        default=(
            "workspace/data/diagnostics/failed_reference_speaker_tests/"
            "failed_attempt_tests_2_3.csv"
        ),
        help="Failed-attempt diagnostic CSV produced by diagnose_failed_samples.py.",
    )
    parser.add_argument(
        "--reference-summary",
        default=(
            "workspace/data/diagnostics/failed_reference_speaker_tests/"
            "reference_test1_summary.csv"
        ),
        help="Reference A/B chunk summary produced by diagnose_failed_samples.py.",
    )
    parser.add_argument(
        "--backend-model",
        default="workspace/models/speaker_backend_3way.pkl",
        help="Existing trained 3-way speaker backend.",
    )
    parser.add_argument(
        "--out-dir",
        default="workspace/data/diagnostics/channel_matching",
        help="Directory for filtered WAVs and CSV results.",
    )
    parser.add_argument(
        "--low-hz",
        type=float,
        default=300.0,
        help="Diagnostic high-pass cutoff in Hz (default: 300).",
    )
    parser.add_argument(
        "--high-hz",
        type=float,
        default=3400.0,
        help="Diagnostic low-pass cutoff in Hz (default: 3400).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional number of rows to test first; 0 means all rows.",
    )
    args = parser.parse_args()

    input_csv = resolve_repo_path(args.input_csv)
    reference_summary_csv = resolve_repo_path(args.reference_summary)
    backend_model = resolve_repo_path(args.backend_model)
    out_dir = resolve_repo_path(args.out_dir)

    require_file(input_csv, "failed-attempt CSV")
    require_file(backend_model, "3-way speaker backend model")

    out_dir.mkdir(parents=True, exist_ok=True)
    filtered_dir = out_dir / "filtered_audio"
    filtered_dir.mkdir(parents=True, exist_ok=True)

    rows_in = pd.read_csv(input_csv)
    if args.limit and args.limit > 0:
        rows_in = rows_in.head(args.limit).copy()

    required_columns = {"attempt_audio", "reference_audio"}
    missing = required_columns - set(rows_in.columns)
    if missing:
        raise ValueError(
            f"Input CSV is missing required column(s): {sorted(missing)}\n"
            f"Available columns: {list(rows_in.columns)}"
        )

    reference_map = build_reference_map(reference_summary_csv)

    print("=" * 78)
    print("CHANNEL-MATCHING DIAGNOSTIC")
    print("=" * 78)
    print(f"Input rows:       {len(rows_in)}")
    print(f"Input CSV:        {input_csv}")
    print(f"Backend model:    {backend_model}")
    print(f"Filter requested: {args.low_hz:.1f}-{args.high_hz:.1f} Hz")
    print(f"Output directory: {out_dir}")

    print("\nLoading ECAPA verifier...")
    verifier = SpeakerEmbeddingVerifier()

    print("Loading existing 3-way backend...")
    backend = ThreeWaySpeakerBackend(
        verifier=verifier,
        model_path=backend_model,
        rebuild=False,
    )

    embedding_cache: dict[str, object] = {}

    def embedding(path: Path):
        key = str(path.resolve())
        if key not in embedding_cache:
            print(f"  Embedding: {repo_relative(path)}")
            embedding_cache[key] = verifier.embedding(path)
        return embedding_cache[key]

    results = []
    warnings = []

    for idx, row in rows_in.iterrows():
        attempt_audio = resolve_repo_path(row.get("attempt_audio", ""))
        reference_audio = resolve_repo_path(row.get("reference_audio", ""))

        label = clean_text(row.get("sample_id", "")) or f"row_{idx}"
        print(f"\n[{idx + 1}/{len(rows_in)}] {label}")

        if attempt_audio is None or not attempt_audio.exists():
            message = f"Attempt audio missing: {attempt_audio}"
            print(f"  [WARN] {message}")
            warnings.append({"row_index": idx, "sample_id": label, "warning": message})
            continue

        if reference_audio is None or not reference_audio.exists():
            message = f"Reference audio missing: {reference_audio}"
            print(f"  [WARN] {message}")
            warnings.append({"row_index": idx, "sample_id": label, "warning": message})
            continue

        chunk_a, chunk_b, chunk_lookup_mode = find_reference_chunks(
            reference_audio,
            reference_map,
            input_csv.parent,
        )

        if not chunk_a.exists() or not chunk_b.exists():
            message = (
                "Reference chunks missing: "
                f"A={chunk_a}, B={chunk_b}"
            )
            print(f"  [WARN] {message}")
            warnings.append({"row_index": idx, "sample_id": label, "warning": message})
            continue

        # Create temporary filtered copy.
        wav, sr = load_mono(attempt_audio)
        filtered, effective_low, effective_high = radio_bandpass(
            wav,
            sr,
            low_hz=args.low_hz,
            high_hz=args.high_hz,
        )
        filtered_path = filtered_dir / stable_filtered_name(row, attempt_audio)
        ta.save(str(filtered_path), filtered.cpu(), sr)

        # Embeddings.
        emb_a = embedding(chunk_a)
        emb_b = embedding(chunk_b)
        emb_full = embedding(reference_audio)
        emb_original = embedding(attempt_audio)
        emb_filtered = embedding(filtered_path)

        target_speaker_id = backend.speaker_id_for_path(reference_audio)
        exclude_ids = [target_speaker_id] if target_speaker_id else []

        # BEFORE: original generated WAV.
        before_a = backend.score_pair(emb_a, emb_original, exclude_speaker_ids=exclude_ids)
        before_b = backend.score_pair(emb_b, emb_original, exclude_speaker_ids=exclude_ids)
        before_full = backend.score_pair(
            emb_full,
            emb_original,
            exclude_speaker_ids=exclude_ids,
        )

        # AFTER: same generated WAV after diagnostic radio-like filtering.
        after_a = backend.score_pair(emb_a, emb_filtered, exclude_speaker_ids=exclude_ids)
        after_b = backend.score_pair(emb_b, emb_filtered, exclude_speaker_ids=exclude_ids)
        after_full = backend.score_pair(
            emb_full,
            emb_filtered,
            exclude_speaker_ids=exclude_ids,
        )

        before_a_dec = bool(before_a.get("decision_same"))
        before_b_dec = bool(before_b.get("decision_same"))
        before_full_dec = bool(before_full.get("decision_same"))
        after_a_dec = bool(after_a.get("decision_same"))
        after_b_dec = bool(after_b.get("decision_same"))
        after_full_dec = bool(after_full.get("decision_same"))

        result_row = {
            "source_row_index": idx,
            "sample_id": clean_text(row.get("sample_id", "")),
            "model": clean_text(row.get("model", "")),
            "condition": clean_text(row.get("condition", "")),
            "text_id": clean_text(row.get("text_id", "")),
            "speaker": clean_text(row.get("speaker", "")),
            "turn_id": row.get("turn_id", ""),
            "attempt_number": row.get("attempt_number", ""),
            "qc_failure_reason": clean_text(row.get("qc_failure_reason", "")),
            "attempt_audio": repo_relative(attempt_audio),
            "filtered_audio": repo_relative(filtered_path),
            "reference_audio": repo_relative(reference_audio),
            "chunk_a": repo_relative(chunk_a),
            "chunk_b": repo_relative(chunk_b),
            "chunk_lookup_mode": chunk_lookup_mode,
            "sample_rate_hz": sr,
            "requested_low_hz": args.low_hz,
            "requested_high_hz": args.high_hz,
            "effective_low_hz": effective_low,
            "effective_high_hz": effective_high,
            "before_ab_status": ab_label(before_a_dec, before_b_dec),
            "after_ab_status": ab_label(after_a_dec, after_b_dec),
            "before_ab_same_count": int(before_a_dec) + int(before_b_dec),
            "after_ab_same_count": int(after_a_dec) + int(after_b_dec),
            "ab_improved_after_filter": (
                int(after_a_dec) + int(after_b_dec)
                > int(before_a_dec) + int(before_b_dec)
            ),
            "a_transition": transition(before_a_dec, after_a_dec),
            "b_transition": transition(before_b_dec, after_b_dec),
            "full_reference_transition": transition(before_full_dec, after_full_dec),
            "any_different_to_same": (
                ((not before_a_dec) and after_a_dec)
                or ((not before_b_dec) and after_b_dec)
                or ((not before_full_dec) and after_full_dec)
            ),
        }

        result_row.update(flatten_score("before_A_vs_original", before_a))
        result_row.update(flatten_score("before_B_vs_original", before_b))
        result_row.update(flatten_score("before_full_vs_original", before_full))
        result_row.update(flatten_score("after_A_vs_filtered", after_a))
        result_row.update(flatten_score("after_B_vs_filtered", after_b))
        result_row.update(flatten_score("after_full_vs_filtered", after_full))

        results.append(result_row)

        print(
            "  A:    "
            f"{before_a_dec} -> {after_a_dec}  "
            f"({result_row['a_transition']})"
        )
        print(
            "  B:    "
            f"{before_b_dec} -> {after_b_dec}  "
            f"({result_row['b_transition']})"
        )
        print(
            "  Full: "
            f"{before_full_dec} -> {after_full_dec}  "
            f"({result_row['full_reference_transition']})"
        )

    results_df = pd.DataFrame(results)
    results_csv = out_dir / "channel_matching_results.csv"
    results_df.to_csv(results_csv, index=False, encoding="utf-8-sig")

    # Compact summary for quick interpretation.
    summary_rows = []
    if not results_df.empty:
        for comparison, transition_col in [
            ("chunk_A", "a_transition"),
            ("chunk_B", "b_transition"),
            ("full_reference", "full_reference_transition"),
        ]:
            counts = results_df[transition_col].value_counts()
            summary_rows.append(
                {
                    "comparison": comparison,
                    "total": len(results_df),
                    "different_to_same": int(counts.get("DIFFERENT_TO_SAME", 0)),
                    "different_to_different": int(counts.get("DIFFERENT_TO_DIFFERENT", 0)),
                    "same_to_same": int(counts.get("SAME_TO_SAME", 0)),
                    "same_to_different": int(counts.get("SAME_TO_DIFFERENT", 0)),
                }
            )

        summary_rows.append(
            {
                "comparison": "A_or_B_or_full_any_improvement",
                "total": len(results_df),
                "different_to_same": int(results_df["any_different_to_same"].sum()),
                "different_to_different": "",
                "same_to_same": "",
                "same_to_different": "",
            }
        )

        summary_rows.append(
            {
                "comparison": "A_B_combined_improved",
                "total": len(results_df),
                "different_to_same": int(results_df["ab_improved_after_filter"].sum()),
                "different_to_different": "",
                "same_to_same": "",
                "same_to_different": "",
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "channel_matching_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    if warnings:
        pd.DataFrame(warnings).to_csv(
            out_dir / "channel_matching_warnings.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print("\n" + "=" * 78)
    print("CHANNEL-MATCHING DIAGNOSTIC COMPLETE")
    print("=" * 78)
    print(f"Rows tested:       {len(results_df)}")
    print(f"Results CSV:       {results_csv}")
    print(f"Summary CSV:       {summary_csv}")
    print(f"Filtered WAVs:     {filtered_dir}")

    if not summary_df.empty:
        print("\nSummary:")
        print(summary_df.to_string(index=False))

    print("\nInterpretation:")
    print(
        "  DIFFERENT_TO_SAME after filtering -> channel/spectral mismatch may be "
        "contributing to the original rejection."
    )
    print(
        "  DIFFERENT_TO_DIFFERENT -> this simple band-pass did not explain the "
        "speaker mismatch; TTS speaker-conditioning failure remains more likely."
    )
    print(
        "\nImportant: this is a diagnostic band-pass test, not a complete VHF "
        "channel model and not proof of speaker identity by itself."
    )


if __name__ == "__main__":
    main()
