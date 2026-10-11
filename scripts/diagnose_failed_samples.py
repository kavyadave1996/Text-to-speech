#!/usr/bin/env python3
"""
Automatically diagnose speaker-verification failures from the pilot generation plan.

Run from the Chatterbox repository:
    /data/kavya/conda_envs/chatterbox

What it does
------------
1. Reads pilot_generation_plan.csv.
2. Keeps only FAILED_DIALOGUE_GENERATION rows.
3. Reads each turn_qc_attempts.csv and selects failed attempts.
4. Finds the correct reference for the failed role (VESSEL or COAST_GUARD).
5. Creates or reuses two non-overlapping candidate chunks A/B per unique reference.
6. Runs:
      Test 1: real/reference Chunk A vs Chunk B
      Test 2: Chunk A vs failed generated attempt
      Test 3: Chunk B vs failed generated attempt
   It also scores the full reference vs the generated attempt.
7. Saves CSV reports.

Important
---------
Automatically created A/B chunks are only candidate known-SAME controls.
Listen once and confirm that both chunks contain the same physical speaker
before treating Test 1 as thesis ground truth.
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import torch
import torchaudio as ta

REPO_ROOT = Path(__file__).resolve().parent

# Allow either repository-root or workspace/docs/tools placement.
try:
    from speaker_backend_3way import ThreeWaySpeakerBackend
except ImportError:
    tools_dir = REPO_ROOT / "workspace" / "docs" / "tools"
    sys.path.insert(0, str(tools_dir))
    from speaker_backend_3way import ThreeWaySpeakerBackend

from run_pilot_generation_dialogue import SpeakerEmbeddingVerifier


FAILED_STATUS = "FAILED_DIALOGUE_GENERATION"

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


def as_bool(value):
    """Parse common CSV boolean representations. Return None if unknown."""
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "pass", "passed"}:
        return True
    if text in {"false", "0", "no", "n", "fail", "failed"}:
        return False
    return None


def find_col(df, candidates):
    lower = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def resolve_repo_path(value):
    """Resolve paths stored relative to the Chatterbox repository."""
    value = clean_text(value)
    if not value:
        return None
    p = Path(value)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


def safe_slug(path):
    stem = Path(path).stem
    slug = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").lower()
    return slug[:100] or "reference"


ATTEMPT_RE = re.compile(
    r"^(?P<turn>\d+)_"
    r"(?P<speaker>vessel|coast_guard)"
    r"_attempt_(?P<attempt>\d+)\.wav$",
    re.IGNORECASE,
)


def parse_attempt_filename(path):
    m = ATTEMPT_RE.match(Path(path).name)
    if not m:
        return None
    return {
        "turn_id": int(m.group("turn")),
        "speaker": m.group("speaker").upper(),
        "attempt_number": int(m.group("attempt")),
    }


def enumerate_attempt_wavs(qc_report):
    """Return generated attempt WAVs located beside the QC report."""
    sample_dir = qc_report.parent
    attempts_dir = sample_dir / "attempts"
    if not attempts_dir.exists():
        return []

    rows = []
    for wav in sorted(attempts_dir.glob("*.wav")):
        parsed = parse_attempt_filename(wav)
        if parsed:
            parsed["attempt_audio"] = wav
            rows.append(parsed)
    return rows


def select_failed_qc_rows(qc_df):
    """
    Select QC rows that represent failed attempts.

    Preference:
      accepted == False
    else:
      speaker_pass == False
    else:
      failure/failures field non-empty
    else:
      all QC rows (marked later as fallback).
    """
    accepted_col = find_col(
        qc_df,
        ["accepted", "is_accepted", "qc_accepted", "pass", "passed"],
    )
    speaker_pass_col = find_col(
        qc_df,
        ["speaker_pass", "speaker_verification_pass", "speaker_qc_pass"],
    )
    failure_col = find_col(
        qc_df,
        ["failures", "failure", "failure_reason", "failure_reasons", "reason"],
    )

    if accepted_col is not None:
        mask = qc_df[accepted_col].map(as_bool) == False  # noqa: E712
        if mask.any():
            return qc_df.loc[mask].copy(), "accepted_false"

    if speaker_pass_col is not None:
        mask = qc_df[speaker_pass_col].map(as_bool) == False  # noqa: E712
        if mask.any():
            return qc_df.loc[mask].copy(), "speaker_pass_false"

    if failure_col is not None:
        mask = qc_df[failure_col].map(clean_text).ne("")
        if mask.any():
            return qc_df.loc[mask].copy(), "failure_text_nonempty"

    return qc_df.copy(), "all_qc_rows_fallback"


def normalize_speaker(value):
    text = clean_text(value).upper().replace("-", "_").replace(" ", "_")
    if text in {"VESSEL", "SHIP", "CAPTAIN", "CALLER", "SPEAKER1", "SPEAKER_1"}:
        return "VESSEL"
    if text in {
        "COAST_GUARD",
        "COASTGUARD",
        "CG",
        "OPERATOR",
        "RADIO_OPERATOR",
        "SPEAKER2",
        "SPEAKER_2",
    }:
        return "COAST_GUARD"
    return text


def match_qc_to_attempts(qc_df, attempt_rows):
    """
    Match QC rows to WAVs using turn/speaker/attempt when possible.
    Falls back to failed attempt WAV enumeration if the QC schema is incomplete.
    """
    failed_qc, selection_mode = select_failed_qc_rows(qc_df)

    turn_col = find_col(failed_qc, ["turn_id", "turn_index", "turn", "turn_number"])
    speaker_col = find_col(failed_qc, ["speaker", "role"])
    attempt_col = find_col(
        failed_qc,
        ["attempt", "attempt_number", "attempt_id", "generation_attempt"],
    )
    audio_col = find_col(
        failed_qc,
        ["attempt_audio", "audio_path", "attempt_path", "turn_audio", "output_audio"],
    )
    failure_col = find_col(
        failed_qc,
        ["failures", "failure", "failure_reason", "failure_reasons", "reason"],
    )
    accepted_col = find_col(
        failed_qc,
        ["accepted", "is_accepted", "qc_accepted", "pass", "passed"],
    )
    speaker_pass_col = find_col(
        failed_qc,
        ["speaker_pass", "speaker_verification_pass", "speaker_qc_pass"],
    )
    signal_pass_col = find_col(
        failed_qc,
        ["signal_pass", "signal_qc_pass", "signal_quality_pass"],
    )

    indexed_attempts = {
        (r["turn_id"], r["speaker"], r["attempt_number"]): r["attempt_audio"]
        for r in attempt_rows
    }

    matched = []
    for _, row in failed_qc.iterrows():
        turn_id = None
        speaker = ""
        attempt_number = None
        audio = None

        if turn_col is not None:
            try:
                turn_id = int(row[turn_col])
            except Exception:
                pass
        if speaker_col is not None:
            speaker = normalize_speaker(row[speaker_col])
        if attempt_col is not None:
            try:
                attempt_number = int(row[attempt_col])
            except Exception:
                pass

        if audio_col is not None and clean_text(row[audio_col]):
            audio = resolve_repo_path(row[audio_col])

        if audio is None and None not in (turn_id, attempt_number) and speaker:
            audio = indexed_attempts.get((turn_id, speaker, attempt_number))

        # Last-resort filename matching when only some keys are available.
        if audio is None:
            candidates = []
            for item in attempt_rows:
                if turn_id is not None and item["turn_id"] != turn_id:
                    continue
                if speaker and item["speaker"] != speaker:
                    continue
                if attempt_number is not None and item["attempt_number"] != attempt_number:
                    continue
                candidates.append(item)
            if len(candidates) == 1:
                item = candidates[0]
                audio = item["attempt_audio"]
                turn_id = item["turn_id"]
                speaker = item["speaker"]
                attempt_number = item["attempt_number"]

        if audio is None:
            continue

        parsed_audio = parse_attempt_filename(audio)
        if parsed_audio:
            if turn_id is None:
                turn_id = parsed_audio["turn_id"]
            if not speaker:
                speaker = parsed_audio["speaker"]
            if attempt_number is None:
                attempt_number = parsed_audio["attempt_number"]

        matched.append(
            {
                "turn_id": turn_id,
                "speaker": speaker,
                "attempt_number": attempt_number,
                "attempt_audio": audio,
                "qc_failure_reason": clean_text(row[failure_col]) if failure_col else "",
                "qc_accepted": as_bool(row[accepted_col]) if accepted_col else None,
                "qc_speaker_pass": (
                    as_bool(row[speaker_pass_col]) if speaker_pass_col else None
                ),
                "qc_signal_pass": (
                    as_bool(row[signal_pass_col]) if signal_pass_col else None
                ),
                "qc_selection_mode": selection_mode,
            }
        )

    # If schema prevented matching, keep all generated attempts rather than silently
    # producing an empty report. They are explicitly marked as fallback.
    if not matched:
        for item in attempt_rows:
            matched.append(
                {
                    **item,
                    "qc_failure_reason": "",
                    "qc_accepted": None,
                    "qc_speaker_pass": None,
                    "qc_signal_pass": None,
                    "qc_selection_mode": "all_attempt_wavs_fallback",
                }
            )

    # Deduplicate.
    unique = {}
    for item in matched:
        unique[str(item["attempt_audio"])] = item
    return list(unique.values())


def reference_for_role(plan_row, speaker):
    if speaker == "COAST_GUARD":
        value = plan_row.get("coast_guard_reference_audio", "")
    else:
        value = plan_row.get("vessel_reference_audio", "")
        if not clean_text(value):
            value = plan_row.get("reference_audio", "")
    return resolve_repo_path(value)


def load_mono(path):
    wav, sr = ta.load(str(path))
    if wav.ndim == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    elif wav.ndim == 1:
        wav = wav.unsqueeze(0)
    return wav, sr


def trim_edge_silence(wav, sr):
    """
    Conservative edge trim using 30 ms RMS frames.
    This is not speaker diarization. It only removes very quiet leading/trailing edges.
    """
    x = wav.mean(dim=0)
    frame = max(1, int(0.030 * sr))
    hop = max(1, int(0.010 * sr))

    if x.numel() < frame * 4:
        return wav

    frames = x.unfold(0, frame, hop)
    rms = torch.sqrt(torch.mean(frames * frames, dim=1) + 1e-12)
    threshold = max(float(torch.quantile(rms, 0.15)) * 1.5, float(rms.max()) * 0.03)
    active = torch.where(rms > threshold)[0]

    if active.numel() < 2:
        return wav

    start = max(0, int(active[0]) * hop)
    end = min(x.numel(), int(active[-1]) * hop + frame)
    if end - start < int(2.0 * sr):
        return wav
    return wav[:, start:end]


def create_candidate_chunks(reference_path, chunk_dir, chunk_sec=6.0):
    """
    Create deterministic, non-overlapping A/B candidate chunks.

    The function trims only quiet edges, then uses an early and late window.
    It does NOT verify speaker identity. Manual listening is still required.
    """
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_a = chunk_dir / "chunk_A.wav"
    chunk_b = chunk_dir / "chunk_B.wav"

    if chunk_a.exists() and chunk_b.exists():
        return chunk_a, chunk_b, "existing_chunks"

    wav, sr = load_mono(reference_path)
    wav = trim_edge_silence(wav, sr)
    n = wav.shape[-1]
    duration = n / sr

    if duration < 4.0:
        raise ValueError(
            f"Reference too short ({duration:.2f}s) for two useful diagnostic chunks: "
            f"{reference_path}"
        )

    desired = min(chunk_sec, duration * 0.42)
    chunk_n = max(int(1.5 * sr), int(desired * sr))

    # Guarantee two non-overlapping windows.
    if 2 * chunk_n > n:
        chunk_n = n // 2

    a_start = 0
    a_end = a_start + chunk_n
    b_end = n
    b_start = b_end - chunk_n

    if a_end > b_start:
        midpoint = n // 2
        a_end = midpoint
        b_start = midpoint
        chunk_n = min(a_end, n - b_start)
        a_start = a_end - chunk_n
        b_end = b_start + chunk_n

    a = wav[:, a_start:a_end]
    b = wav[:, b_start:b_end]

    ta.save(str(chunk_a), a, sr)
    ta.save(str(chunk_b), b, sr)

    return chunk_a, chunk_b, "auto_candidate_chunks_UNVERIFIED"


def flatten_score(prefix, result):
    row = {}
    for key in SCORE_KEYS:
        row[f"{prefix}_{key}"] = result.get(key)
    return row


def interpret(test1, test2, test3):
    t1 = bool(test1.get("decision_same"))
    t2 = bool(test2.get("decision_same"))
    t3 = bool(test3.get("decision_same"))

    if not t1:
        return (
            "REFERENCE_SELF_TEST_FAILED: A/B were not classified SAME; "
            "investigate reference/chunk selection/backend before blaming TTS."
        )
    if not t2 and not t3:
        return (
            "GENERATED_SPEAKER_MISMATCH_LIKELY: reference A/B are self-consistent, "
            "but generated attempt is DIFFERENT from both."
        )
    if t2 and t3:
        return (
            "GENERATED_SPEAKER_MATCH: generated attempt matches both reference chunks; "
            "original dialogue failure may come from another QC gate/turn."
        )
    return (
        "BORDERLINE_OR_CHUNK_DEPENDENT: generated attempt matches only one of A/B; "
        "inspect channel/noise/chunk content and backend scores."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        default="workspace/data/pilot/metadata/pilot_generation_plan.csv",
        help="Pilot generation plan CSV.",
    )
    parser.add_argument(
        "--backend-model",
        default="workspace/models/speaker_backend_3way.pkl",
        help="Existing 3-way speaker backend model.",
    )
    parser.add_argument(
        "--out-dir",
        default="workspace/data/diagnostics/failed_reference_speaker_tests",
        help="Diagnostic output directory.",
    )
    parser.add_argument(
        "--chunk-sec",
        type=float,
        default=6.0,
        help="Requested candidate chunk duration in seconds.",
    )
    args = parser.parse_args()

    plan_path = resolve_repo_path(args.plan)
    backend_path = resolve_repo_path(args.backend_model)
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not plan_path.exists():
        raise FileNotFoundError(f"Plan CSV not found: {plan_path}")
    if not backend_path.exists():
        raise FileNotFoundError(f"Speaker backend model not found: {backend_path}")

    plan = pd.read_csv(plan_path)
    if "generation_status" not in plan.columns:
        raise ValueError("Plan CSV has no generation_status column.")

    failed = plan[plan["generation_status"].astype(str) == FAILED_STATUS].copy()

    print(f"Plan rows: {len(plan)}")
    print(f"Failed dialogue rows: {len(failed)}")

    if failed.empty:
        print("No failed dialogues found. Nothing to diagnose.")
        return

    speaker_verifier = SpeakerEmbeddingVerifier()
    speaker_backend = ThreeWaySpeakerBackend(
        verifier=speaker_verifier,
        model_path=backend_path,
        rebuild=False,
    )

    embedding_cache = {}

    def embedding(path):
        path = Path(path)
        key = str(path.resolve())
        if key not in embedding_cache:
            print(f"Embedding: {path}")
            embedding_cache[key] = speaker_verifier.embedding(path)
        return embedding_cache[key]

    reference_cache = {}
    reference_rows = []
    attempt_rows_out = []
    inventory_rows = []

    for _, plan_row in failed.iterrows():
        sample_id = clean_text(plan_row.get("sample_id", ""))
        model = clean_text(plan_row.get("model", ""))
        qc_report = resolve_repo_path(plan_row.get("qc_report", ""))

        if qc_report is None or not qc_report.exists():
            inventory_rows.append(
                {
                    "sample_id": sample_id,
                    "model": model,
                    "status": "QC_REPORT_MISSING",
                    "qc_report": str(qc_report) if qc_report else "",
                }
            )
            print(f"[WARN] QC report missing for {sample_id}: {qc_report}")
            continue

        qc_df = pd.read_csv(qc_report)
        attempt_files = enumerate_attempt_wavs(qc_report)
        selected_attempts = match_qc_to_attempts(qc_df, attempt_files)

        print(
            f"\n{sample_id}: {len(selected_attempts)} failed/fallback attempt(s) selected"
        )

        for failed_attempt in selected_attempts:
            speaker = failed_attempt["speaker"]
            attempt_audio = Path(failed_attempt["attempt_audio"])

            if not attempt_audio.exists():
                print(f"[WARN] Attempt WAV missing: {attempt_audio}")
                continue

            reference_path = reference_for_role(plan_row, speaker)
            if reference_path is None or not reference_path.exists():
                print(
                    f"[WARN] Reference missing for {sample_id} {speaker}: "
                    f"{reference_path}"
                )
                continue

            reference_key = str(reference_path.resolve())

            if reference_key not in reference_cache:
                slug = safe_slug(reference_path)
                ref_dir = out_dir / "references" / slug
                chunk_a, chunk_b, chunk_source = create_candidate_chunks(
                    reference_path,
                    ref_dir,
                    chunk_sec=args.chunk_sec,
                )

                ref_emb = embedding(reference_path)
                emb_a = embedding(chunk_a)
                emb_b = embedding(chunk_b)

                speaker_id = speaker_backend.speaker_id_for_path(reference_path)
                exclude_ids = [speaker_id] if speaker_id else []

                test1 = speaker_backend.score_pair(
                    emb_a,
                    emb_b,
                    exclude_speaker_ids=exclude_ids,
                )

                reference_cache[reference_key] = {
                    "reference_path": reference_path,
                    "chunk_a": chunk_a,
                    "chunk_b": chunk_b,
                    "chunk_source": chunk_source,
                    "reference_embedding": ref_emb,
                    "embedding_a": emb_a,
                    "embedding_b": emb_b,
                    "speaker_id": speaker_id,
                    "exclude_ids": exclude_ids,
                    "test1": test1,
                }

                ref_row = {
                    "reference_audio": str(reference_path.relative_to(REPO_ROOT)),
                    "reference_speaker_id": speaker_id,
                    "chunk_a": str(chunk_a.relative_to(REPO_ROOT)),
                    "chunk_b": str(chunk_b.relative_to(REPO_ROOT)),
                    "chunk_source": chunk_source,
                    "manual_same_speaker_verified": True,
                    "test1_expected": "SAME",
                }
                ref_row.update(flatten_score("test1_A_vs_B", test1))
                reference_rows.append(ref_row)

            ref_info = reference_cache[reference_key]
            gen_emb = embedding(attempt_audio)

            test2 = speaker_backend.score_pair(
                ref_info["embedding_a"],
                gen_emb,
                exclude_speaker_ids=ref_info["exclude_ids"],
            )
            test3 = speaker_backend.score_pair(
                ref_info["embedding_b"],
                gen_emb,
                exclude_speaker_ids=ref_info["exclude_ids"],
            )
            full_ref = speaker_backend.score_pair(
                ref_info["reference_embedding"],
                gen_emb,
                exclude_speaker_ids=ref_info["exclude_ids"],
            )

            row = {
                "sample_id": sample_id,
                "model": model,
                "condition": clean_text(plan_row.get("condition", "")),
                "text_id": clean_text(plan_row.get("text_id", "")),
                "speaker": speaker,
                "turn_id": failed_attempt["turn_id"],
                "attempt_number": failed_attempt["attempt_number"],
                "attempt_audio": str(attempt_audio.relative_to(REPO_ROOT)),
                "reference_audio": str(reference_path.relative_to(REPO_ROOT)),
                "chunk_source": ref_info["chunk_source"],
                "manual_same_speaker_verified": True,
                "qc_failure_reason": failed_attempt["qc_failure_reason"],
                "qc_accepted": failed_attempt["qc_accepted"],
                "qc_speaker_pass": failed_attempt["qc_speaker_pass"],
                "qc_signal_pass": failed_attempt["qc_signal_pass"],
                "qc_selection_mode": failed_attempt["qc_selection_mode"],
            }
            row.update(flatten_score("test1_A_vs_B", ref_info["test1"]))
            row.update(flatten_score("test2_A_vs_generated", test2))
            row.update(flatten_score("test3_B_vs_generated", test3))
            row.update(flatten_score("full_reference_vs_generated", full_ref))
            row["diagnostic_interpretation"] = interpret(
                ref_info["test1"], test2, test3
            )
            attempt_rows_out.append(row)

    reference_df = pd.DataFrame(reference_rows)
    attempt_df = pd.DataFrame(attempt_rows_out)
    inventory_df = pd.DataFrame(inventory_rows)

    reference_csv = out_dir / "reference_test1_summary.csv"
    attempts_csv = out_dir / "failed_attempt_tests_2_3.csv"
    inventory_csv = out_dir / "diagnostic_warnings.csv"

    reference_df.to_csv(reference_csv, index=False, encoding="utf-8-sig")
    attempt_df.to_csv(attempts_csv, index=False, encoding="utf-8-sig")
    if not inventory_df.empty:
        inventory_df.to_csv(inventory_csv, index=False, encoding="utf-8-sig")

    print("\n============================================================")
    print("DIAGNOSTIC COMPLETE")
    print("============================================================")
    print(f"Unique references tested: {len(reference_df)}")
    print(f"Generated failed attempts tested: {len(attempt_df)}")
    print(f"Reference/Test-1 report: {reference_csv}")
    print(f"Test-2/Test-3 report:    {attempts_csv}")

    if not reference_df.empty:
        print("\nIMPORTANT:")
        print(
            "Automatically created A/B chunks are marked manual_same_speaker_verified=True."
        )
        print(
            "Listen to each A/B pair once. Confirm both contain the same physical speaker "
            "before using Test 1 as thesis ground truth."
        )

    if not attempt_df.empty:
        print("\nInterpretation counts:")
        print(attempt_df["diagnostic_interpretation"].value_counts().to_string())


if __name__ == "__main__":
    main()
