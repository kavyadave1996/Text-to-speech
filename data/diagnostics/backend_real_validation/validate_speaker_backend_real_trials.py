#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

from run_pilot_generation_dialogue import SpeakerEmbeddingVerifier
from speaker_backend_3way import ThreeWaySpeakerBackend

BACKEND_MODEL = Path("workspace/models/speaker_backend_3way.pkl")

OUTPUT_DIR = Path("workspace/data/diagnostics/backend_real_validation")
RESULTS_CSV = OUTPUT_DIR / "real_trials_results.csv"
SUMMARY_CSV = OUTPUT_DIR / "method_summary.csv"

SAME_PAIRS = {
    "yacht": {
        "original_ref": Path(
            "workspace/docs/Noisy_dataset/reference_voices/"
            "(51) Yacht's mayday call_ 'We are sinking'.wav"
        ),
        "a": Path(
            "workspace/data/diagnostics/failed_reference_speaker_tests/"
            "references/51_yacht_s_mayday_call_we_are_sinking/chunk_A.wav"
        ),
        "b": Path(
            "workspace/data/diagnostics/failed_reference_speaker_tests/"
            "references/51_yacht_s_mayday_call_we_are_sinking/chunk_B.wav"
        ),
    },
    "boaters": {
        "original_ref": Path(
            "workspace/docs/Noisy_dataset/reference_voices/"
            "(54) Boaters mayday call.wav"
        ),
        "a": Path(
            "workspace/data/diagnostics/failed_reference_speaker_tests/"
            "references/54_boaters_mayday_call/chunk_A.wav"
        ),
        "b": Path(
            "workspace/data/diagnostics/failed_reference_speaker_tests/"
            "references/54_boaters_mayday_call/chunk_B.wav"
        ),
    },
    "saginaw": {
        "original_ref": Path(
            "workspace/docs/Noisy_dataset/reference_voices/"
            "(54) Mayday call from capsized vessel on Saginaw Bay.wav"
        ),
        "a": Path(
            "workspace/data/diagnostics/failed_reference_speaker_tests/"
            "references/54_mayday_call_from_capsized_vessel_on_saginaw_bay/"
            "chunk_A.wav"
        ),
        "b": Path(
            "workspace/data/diagnostics/failed_reference_speaker_tests/"
            "references/54_mayday_call_from_capsized_vessel_on_saginaw_bay/"
            "chunk_B.wav"
        ),
    },
    "whatsapp": {
        "original_ref": Path(
            "workspace/docs/Noisy_dataset/reference_voices/"
            "WhatsApp_audio_1.wav"
        ),
        "a": Path(
            "workspace/data/diagnostics/failed_reference_speaker_tests/"
            "references/whatsapp_audio_1/chunk_A.wav"
        ),
        "b": Path(
            "workspace/data/diagnostics/failed_reference_speaker_tests/"
            "references/whatsapp_audio_1/chunk_B.wav"
        ),
    },
    "hormuz": {
        "original_ref": Path(
            "workspace/docs/Noisy_dataset/reference_voices/"
            "(51) Audio captures captain’s distress call after Iran fires "
            "on Indian tanker in Hormuz.wav"
        ),
        "a": Path(
            "workspace/data/diagnostics/failed_reference_speaker_tests/"
            "references/"
            "51_audio_captures_captain_s_distress_call_after_iran_fires_"
            "on_indian_tanker_in_hormuz/chunk_A.wav"
        ),
        "b": Path(
            "workspace/data/diagnostics/failed_reference_speaker_tests/"
            "references/"
            "51_audio_captures_captain_s_distress_call_after_iran_fires_"
            "on_indian_tanker_in_hormuz/chunk_B.wav"
        ),
    },
}

DIFFERENT_REFS = {
    "coast_guard_1": Path(
        "workspace/docs/Noisy_dataset/reference_voices/"
        "Northern Belle_ coast_guard.wav"
    ),
    "coast_guard_2": Path(
        "workspace/docs/Noisy_dataset/reference_voices/"
        "coast_guard_noisy.wav"
    ),
}


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file:\n{path}")


def add_score_columns(row, prefix, result):
    keys = [
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
    for key in keys:
        row[f"{prefix}_{key}"] = result.get(key)


def method_correct(expected_same: bool, actual_same: bool) -> bool:
    return bool(actual_same) == bool(expected_same)


def print_result(label, expected, result):
    print(f"\n{label}")
    print(f"  expected: {expected}")
    print(f"  raw_cosine:             {result['raw_cosine']:.6f}")
    print(f"  raw_plda_score:         {result['raw_plda_score']:.6f}")
    print(f"  raw_cohort_snorm:       {result['raw_cohort_snorm']:.6f}")
    print(f"  calibrated_cosine_same: {result['calibrated_cosine_same']}")
    print(f"  plda_same:              {result['plda_same']}")
    print(f"  cohort_same:            {result['cohort_same']}")
    print(f"  same_votes:             {result['same_votes']}")
    print(f"  FINAL decision_same:    {result['decision_same']}")


def main():
    print("=" * 70)
    print("REAL-HUMAN SPEAKER BACKEND VALIDATION")
    print("=" * 70)

    require_file(BACKEND_MODEL)

    for data in SAME_PAIRS.values():
        require_file(data["original_ref"])
        require_file(data["a"])
        require_file(data["b"])

    for path in DIFFERENT_REFS.values():
        require_file(path)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nAll required files exist.")

    print("\nLoading SpeakerEmbeddingVerifier...")
    verifier = SpeakerEmbeddingVerifier()

    print("Loading existing 3-way backend...")
    backend = ThreeWaySpeakerBackend(
        verifier=verifier,
        model_path=BACKEND_MODEL,
        rebuild=False,
    )

    embedding_cache = {}

    def embedding(path: Path):
        key = str(path)
        if key not in embedding_cache:
            print(f"Embedding: {path}")
            embedding_cache[key] = verifier.embedding(path)
        return embedding_cache[key]

    different_embeddings = {
        name: embedding(path)
        for name, path in DIFFERENT_REFS.items()
    }

    rows = []

    print("\n" + "=" * 70)
    print("KNOWN SAME-SPEAKER TRIALS")
    print("=" * 70)

    for speaker_name, data in SAME_PAIRS.items():
        emb_a = embedding(data["a"])
        emb_b = embedding(data["b"])

        target_speaker_id = backend.speaker_id_for_path(data["original_ref"])

        result = backend.score_pair(
            emb_a,
            emb_b,
            exclude_speaker_ids=[target_speaker_id],
        )

        print_result(
            f"SAME TEST: {speaker_name} A vs {speaker_name} B",
            "SAME",
            result,
        )

        row = {
            "trial_type": "SAME",
            "expected_same": True,
            "speaker_1": speaker_name,
            "speaker_2": speaker_name,
            "audio_1": str(data["a"]),
            "audio_2": str(data["b"]),
            "manual_ground_truth_verified": True,
        }

        add_score_columns(row, "score", result)

        row["cosine_correct"] = method_correct(
            True, result["calibrated_cosine_same"]
        )
        row["plda_correct"] = method_correct(
            True, result["plda_same"]
        )
        row["cohort_correct"] = method_correct(
            True, result["cohort_same"]
        )
        row["final_correct"] = method_correct(
            True, result["decision_same"]
        )

        rows.append(row)

    print("\n" + "=" * 70)
    print("KNOWN DIFFERENT-SPEAKER TRIALS")
    print("=" * 70)

    for speaker_name, data in SAME_PAIRS.items():
        emb_a = embedding(data["a"])
        target_speaker_id = backend.speaker_id_for_path(data["original_ref"])

        for different_name, different_path in DIFFERENT_REFS.items():
            emb_different = different_embeddings[different_name]
            different_speaker_id = backend.speaker_id_for_path(different_path)

            result = backend.score_pair(
                emb_a,
                emb_different,
                exclude_speaker_ids=[
                    target_speaker_id,
                    different_speaker_id,
                ],
            )

            print_result(
                f"DIFFERENT TEST: {speaker_name} A vs {different_name}",
                "DIFFERENT",
                result,
            )

            row = {
                "trial_type": "DIFFERENT",
                "expected_same": False,
                "speaker_1": speaker_name,
                "speaker_2": different_name,
                "audio_1": str(data["a"]),
                "audio_2": str(different_path),
                "manual_ground_truth_verified": True,
            }

            add_score_columns(row, "score", result)

            row["cosine_correct"] = method_correct(
                False, result["calibrated_cosine_same"]
            )
            row["plda_correct"] = method_correct(
                False, result["plda_same"]
            )
            row["cohort_correct"] = method_correct(
                False, result["cohort_same"]
            )
            row["final_correct"] = method_correct(
                False, result["decision_same"]
            )

            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")

    summary_rows = []

    methods = {
        "calibrated_cosine": "cosine_correct",
        "plda": "plda_correct",
        "cohort": "cohort_correct",
        "final_2_of_3": "final_correct",
    }

    for method_name, correctness_column in methods.items():
        same_df = df[df["trial_type"] == "SAME"]
        different_df = df[df["trial_type"] == "DIFFERENT"]

        same_correct = int(same_df[correctness_column].sum())
        different_correct = int(different_df[correctness_column].sum())

        summary_rows.append(
            {
                "method": method_name,
                "same_correct": same_correct,
                "same_total": len(same_df),
                "same_accuracy": (
                    same_correct / len(same_df) if len(same_df) else 0.0
                ),
                "different_correct": different_correct,
                "different_total": len(different_df),
                "different_accuracy": (
                    different_correct / len(different_df)
                    if len(different_df)
                    else 0.0
                ),
                "overall_correct": same_correct + different_correct,
                "overall_total": len(df),
                "overall_accuracy": (
                    (same_correct + different_correct) / len(df)
                    if len(df)
                    else 0.0
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    print(
        summary_df[
            [
                "method",
                "same_correct",
                "same_total",
                "different_correct",
                "different_total",
                "overall_correct",
                "overall_total",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(f"  Full trials: {RESULTS_CSV}")
    print(f"  Summary:     {SUMMARY_CSV}")

    print("\nExpected trial count:")
    print("  SAME trials:      5")
    print("  DIFFERENT trials: 10")
    print("  TOTAL:            15")


if __name__ == "__main__":
    main()
