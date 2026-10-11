run_pilot_generation_dialogue.py
#!/usr/bin/env python3
"""
Generate two-speaker maritime dialogue audio using Chatterbox or Fish-Speech.

Speaker policy
--------------
This script NEVER infers, repairs, or alternates speakers.
It requires explicit VESSEL:/COAST_GUARD: labels already stored in dialogue_text.

The labels are control metadata only. They are removed before sending text to
Fish-Speech / Chatterbox and are used solely to select the correct reference voice.
"""

import argparse
import re
from pathlib import Path

import msgpack
import pandas as pd
import requests
import torch
import torchaudio as ta
import torchaudio.functional as F
from chatterbox.tts import ChatterboxTTS


FISH_DEFAULT_URL = "http://127.0.0.1:8080/v1/tts"

ROLE_SEEDS = {
    "VESSEL": 42001,
    "COAST_GUARD": 43001,
}


def set_seed(seed):
    """
    Make repeated synthesis more reproducible.
    """

    import random

    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_missing(value):
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    return str(value).strip() == ""


def row_value(row, names, default=""):
    for name in names:
        if name in row.index and not is_missing(row[name]):
            return str(row[name]).strip()
    return default


def canonical_speaker(name):
    value = str(name).strip().lower().replace("-", "_").replace(" ", "_")

    if value in {"vessel", "ship", "speaker1", "speaker_1", "captain", "caller"}:
        return "VESSEL"

    if value in {
        "coast_guard",
        "coastguard",
        "cg",
        "speaker2",
        "speaker_2",
        "operator",
        "radio_operator",
    }:
        return "COAST_GUARD"

    return None


def parse_dialogue(text):
    """
    Split explicit VESSEL and COAST_GUARD turns.

    This works when labels are:
    - on separate lines;
    - separated by <br>;
    - embedded in one long Excel cell;
    - written with spaces such as COAST_GUARD :.
    """

    text = (
        str(text)
        .replace("<br>", "\n")
        .replace("\r\n", "\n")
        .strip()
    )

    if not text:
        raise ValueError("Dialogue text is empty.")

    if "REVIEW_REQUIRED:" in text:
        raise ValueError(
            "Dialogue contains REVIEW_REQUIRED labels. "
            "Resolve them before generation."
        )

    speaker_pattern = re.compile(
        r"(?i)(?<![A-Za-z0-9_])"
        r"(?P<label>VESSEL|COAST[\s_-]*GUARD)"
        r"\s*:"
    )

    matches = list(speaker_pattern.finditer(text))

    if not matches:
        raise ValueError(
            "No VESSEL or COAST_GUARD labels were found."
        )

    text_before_first_label = text[:matches[0].start()].strip()

    if text_before_first_label:
        raise ValueError(
            "Text exists before the first speaker label:\n"
            f"{text_before_first_label[:150]}"
        )

    turns = []

    for index, match in enumerate(matches):
        raw_label = match.group("label")

        normalized_label = re.sub(
            r"[\s-]+",
            "_",
            raw_label.upper().strip(),
        )

        if normalized_label == "COAST_GUARD":
            speaker = "COAST_GUARD"
        else:
            speaker = "VESSEL"

        turn_start = match.end()

        if index + 1 < len(matches):
            turn_end = matches[index + 1].start()
        else:
            turn_end = len(text)

        turn_text = text[turn_start:turn_end]
        turn_text = re.sub(r"\s+", " ", turn_text).strip()

        if not turn_text:
            raise ValueError(
                f"Empty text after {speaker} label."
            )

        turns.append(
            {
                "speaker": speaker,
                "text": turn_text,
            }
        )

    speakers = {turn["speaker"] for turn in turns}

    if "VESSEL" not in speakers:
        raise ValueError(
            "No VESSEL turn was parsed."
        )

    if "COAST_GUARD" not in speakers:
        raise ValueError(
            "No COAST_GUARD turn was parsed."
        )

    if len(turns) < 2:
        raise ValueError(
            "The dialogue was parsed as fewer than two turns."
        )

    return turns


def get_reference_audio(row, speaker):
    """
    Select reference audio using the already-known speaker label.

    COAST_GUARD intentionally has no fallback to generic reference_audio,
    because generic reference_audio is the vessel/source reference.
    """
    if speaker == "COAST_GUARD":
        return row_value(
            row,
            [
                "coast_guard_reference_audio",
                "coastguard_reference_audio",
                "cg_reference_audio",
                "speaker2_reference_audio",
                "reference_audio_coast_guard",
            ],
        )

    return row_value(
        row,
        [
            "vessel_reference_audio",
            "ship_reference_audio",
            "speaker1_reference_audio",
            "reference_audio_vessel",
            "reference_audio",
        ],
    )


def validate_required_references(row, turns):
    speakers = {turn["speaker"] for turn in turns}

    vessel_ref = get_reference_audio(row, "VESSEL")
    coast_guard_ref = get_reference_audio(row, "COAST_GUARD")

    if "VESSEL" in speakers and not vessel_ref:
        raise ValueError(
            "Missing vessel reference audio. Add vessel_reference_audio or reference_audio."
        )

    if "COAST_GUARD" in speakers and not coast_guard_ref:
        raise ValueError(
            "Missing Coast Guard reference audio. Add coast_guard_reference_audio."
        )

    if vessel_ref and coast_guard_ref:
        vessel_path = Path(vessel_ref)
        cg_path = Path(coast_guard_ref)

        try:
            same_ref = vessel_path.resolve() == cg_path.resolve()
        except Exception:
            same_ref = vessel_path == cg_path

        if same_ref:
            raise ValueError(
                "VESSEL and COAST_GUARD use the same reference audio. "
                "Provide two separate voices."
            )

    return vessel_ref, coast_guard_ref


def get_reference_text(row, speaker):
    if speaker == "COAST_GUARD":
        # Do NOT fall back to generic reference_text; that generic transcript
        # may belong to the vessel reference and would mismatch the CG audio.
        return row_value(
            row,
            [
                "coast_guard_reference_text",
                "coastguard_reference_text",
                "cg_reference_text",
                "speaker2_reference_text",
                "reference_text_coast_guard",
            ],
            default="",
        )

    return row_value(
        row,
        [
            "vessel_reference_text",
            "ship_reference_text",
            "speaker1_reference_text",
            "reference_text_vessel",
            "reference_text",
        ],
        default="",
    )


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def ensure_2d_audio(wav):
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.dim() == 3 and wav.size(0) == 1:
        wav = wav.squeeze(0)
    return wav


def generate_chatterbox_turn(
    model,
    text,
    reference_audio,
    output_path,
    exaggeration,
    cfg_weight,
    seed,
):
    reference_audio = Path(reference_audio)
    output_path = Path(output_path)

    if not reference_audio.exists():
        raise FileNotFoundError(f"Reference audio not found: {reference_audio}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    wav = model.generate(
        text,
        audio_prompt_path=str(reference_audio),
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
    )

    wav = ensure_2d_audio(wav)
    ta.save(str(output_path), wav.cpu(), model.sr)

    return output_path


def generate_fishspeech_turn(
    text,
    speaker,
    reference_audio,
    reference_text,
    output_path,
    fish_url,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    references = []

    if reference_audio:
        reference_audio = Path(reference_audio)

        if not reference_audio.exists():
            raise FileNotFoundError(f"Reference audio not found: {reference_audio}")

        references.append(
            {
                "audio": reference_audio.read_bytes(),
                "text": reference_text or "",
            }
        )

    data = {
    "text": text,
    "chunk_length": 300,
    "format": "wav",
    "latency": "normal",
    "references": references,
    "reference_id": None,

    # Same seed for all turns of the same role.
    "seed": ROLE_SEEDS[speaker],

    "use_memory_cache": "on",
    "normalize": True,
    "streaming": False,

    # Prevent long first turns from ending too early.
    "max_new_tokens": 2048,

    # Lower randomness for better speaker consistency.
    "top_p": 0.80,
    "repetition_penalty": 1.10,
    "temperature": 0.55,
}

    packed = msgpack.packb(data, use_bin_type=True)

    response = requests.post(
        fish_url,
        data=packed,
        headers={
            "Content-Type": "application/msgpack",
            "Accept": "audio/wav",
        },
        timeout=None,
    )

    if response.status_code == 200:
        output_path.write_bytes(response.content)
        return output_path

    print(response.text)
    raise RuntimeError(
        f"Fish-Speech generation failed with status {response.status_code}"
    )


def load_audio_mono(path, target_sr=None):
    wav, sr = ta.load(str(path))

    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if target_sr is not None and sr != target_sr:
        wav = F.resample(wav, sr, target_sr)
        sr = target_sr

    return wav, sr


def concatenate_turns(turn_paths, output_path, silence_sec=0.45):
    if not turn_paths:
        raise ValueError("No turn audio files to concatenate.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio_parts = []
    target_sr = None

    for path in turn_paths:
        wav, sr = load_audio_mono(path, target_sr=target_sr)

        if target_sr is None:
            target_sr = sr

        audio_parts.append(wav)

        silence_len = int(silence_sec * target_sr)
        if silence_len > 0:
            audio_parts.append(
                torch.zeros((1, silence_len), dtype=wav.dtype)
            )

    # Remove the final inserted silence so the file does not end with an
    # artificial inter-turn gap.
    if audio_parts and turn_paths and silence_sec > 0:
        audio_parts = audio_parts[:-1]

    final_wav = torch.cat(audio_parts, dim=1)
    ta.save(str(output_path), final_wav, target_sr)

    return output_path, target_sr, final_wav.size(1) / target_sr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        default="workspace/data/pilot/metadata/pilot_generation_plan.csv",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fish_url", default=FISH_DEFAULT_URL)
    parser.add_argument("--silence_sec", type=float, default=0.45)
    parser.add_argument("--overwrite", action="store_true")

    # Kept for backward CLI compatibility.
    parser.add_argument("--exaggeration", type=float, default=0.7)
    parser.add_argument("--cfg_weight", type=float, default=0.3)

    parser.add_argument(
    "--vessel_exaggeration",
    type=float,
    default=0.70,
    )

    parser.add_argument(
        "--vessel_cfg_weight",
        type=float,
        default=0.35,
    )

    parser.add_argument(
        "--coast_guard_exaggeration",
        type=float,
        default=0.35,
    )

    parser.add_argument(
        "--coast_guard_cfg_weight",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print parsed turns/references and save metadata without synthesizing audio.",
    )

    args = parser.parse_args()

    plan_path = Path(args.plan)

    if not plan_path.exists():
        raise FileNotFoundError(f"Plan file not found: {plan_path}")

    full_df = pd.read_csv(plan_path)
    work_df = full_df.head(args.limit) if args.limit is not None else full_df

    print("Plan:", plan_path)
    print("Total rows in plan:", len(full_df))
    print("Rows selected:", len(work_df))

    chatterbox_model = None

    for idx, row in work_df.iterrows():
        model_name = str(row["model"]).strip()
        if model_name == "Fish-Speech":
            dialogue_text = str(row["target_text"])
        else:
            dialogue_text = str(row["dialogue_text"])

        print(f"\nModel: {model_name}")
        print(f"Text being generated:\n{dialogue_text}\n")

        output_audio_value = row_value(row, ["output_audio"])
        if not output_audio_value:
            print(f"Row {idx}: missing output_audio")
            full_df.at[idx, "generation_status"] = "FAILED_MISSING_OUTPUT_AUDIO"
            continue

        output_audio = Path(output_audio_value)

        if not dialogue_text:
            print(f"Row {idx}: missing dialogue/target text")
            full_df.at[idx, "generation_status"] = "FAILED_MISSING_TEXT"
            continue

        if output_audio.exists() and not args.overwrite and not args.dry_run:
            print(f"Skipping existing: {output_audio}")
            full_df.at[idx, "generation_status"] = "EXISTS"
            continue

        print("\n========== RAW LABELED DIALOGUE ==========")
        print(dialogue_text)
        print("==========================================\n")

        try:
            turns = parse_dialogue(dialogue_text)

            speaker_sequence = [
                turn["speaker"]
                for turn in turns
            ]

            print(
                f"Parsed {len(turns)} turns: "
                f"{speaker_sequence}"
            )

            speakers = set(speaker_sequence)

            if speakers != {"VESSEL", "COAST_GUARD"}:
                raise ValueError(
                    "Both VESSEL and COAST_GUARD are required. "
                    f"Parsed speakers: {sorted(speakers)}"
                )

            validate_required_references(row, turns)

        except Exception as exc:
            print(
                f"Row {idx}: dialogue/reference validation failed: "
                f"{exc}"
            )
            full_df.at[
                idx,
                "generation_status",
            ] = "FAILED_VALIDATION"

            full_df.to_csv(
                plan_path,
                index=False,
                encoding="utf-8-sig",
            )
            continue

        sample_id = row_value(row, ["sample_id"], default=output_audio.stem)
        turn_dir = output_audio.parent / "turns" / output_audio.stem
        turn_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating {sample_id}: {model_name}, {len(turns)} turns")

        try:
            turn_paths = []
            turn_rows = []

            if model_name == "Chatterbox" and chatterbox_model is None and not args.dry_run:
                device = get_device()
                print(f"Loading Chatterbox model on {device}")
                chatterbox_model = ChatterboxTTS.from_pretrained(device=device)

            for turn_index, turn in enumerate(turns, start=1):
                speaker = turn["speaker"]
                turn_text = turn["text"]

                reference_audio = get_reference_audio(row, speaker)
                reference_text = get_reference_text(row, speaker)

                if model_name == "Fish-Speech" and not reference_text:
                    raise ValueError(
                        f"Missing Fish-Speech reference transcript "
                        f"for {speaker}: {reference_audio}"
                    )

                if not reference_audio:
                    raise ValueError(
                        f"No reference audio found for speaker {speaker}"
                    )

                turn_path = turn_dir / f"{turn_index:03d}_{speaker.lower()}.wav"

                print(
                    f"  Turn {turn_index:03d} [{speaker}]: "
                    f"{turn_text[:100]}..."
                )
                print(f"      reference_audio: {reference_audio}")

                turn_rows.append(
                    {
                        "sample_id": sample_id,
                        "turn_id": turn_index,
                        "speaker": speaker,
                        "turn_text": turn_text,
                        "reference_audio": reference_audio,
                        "reference_text": reference_text,
                        "turn_audio": str(turn_path),
                    }
                )

                if args.dry_run:
                    continue

                if model_name == "Chatterbox":
                    if speaker == "VESSEL":
                        turn_exaggeration = args.vessel_exaggeration
                        turn_cfg_weight = args.vessel_cfg_weight
                    else:
                        turn_exaggeration = args.coast_guard_exaggeration
                        turn_cfg_weight = args.coast_guard_cfg_weight

                    generate_chatterbox_turn(
                        chatterbox_model,
                        turn_text,
                        reference_audio,
                        turn_path,
                        turn_exaggeration,
                        turn_cfg_weight,
                        ROLE_SEEDS[speaker],
                    )

                elif model_name == "Fish-Speech":
                    generate_fishspeech_turn(
                        turn_text,
                        speaker,
                        reference_audio,
                        reference_text,
                        turn_path,
                        args.fish_url,
                    )

                else:
                    raise ValueError(f"Unknown model: {model_name}")

                turn_paths.append(turn_path)

            turns_csv = turn_dir / "turns_metadata.csv"
            pd.DataFrame(turn_rows).to_csv(
                turns_csv,
                index=False,
                encoding="utf-8-sig",
            )

            if args.dry_run:
                full_df.at[idx, "generation_status"] = "DRY_RUN_OK"
                full_df.at[idx, "dialogue_num_turns"] = len(turns)
                full_df.at[idx, "dialogue_turns_metadata"] = str(turns_csv)
                full_df.to_csv(plan_path, index=False, encoding="utf-8-sig")

                print(f"Dry run only. Saved turn metadata: {turns_csv}")
                continue

            _, sr, duration = concatenate_turns(
                turn_paths,
                output_audio,
                silence_sec=args.silence_sec,
            )

            full_df.at[idx, "generation_status"] = "DONE"
            full_df.at[idx, "dialogue_num_turns"] = len(turns)
            full_df.at[idx, "dialogue_turns_metadata"] = str(turns_csv)
            full_df.at[idx, "duration_sec"] = round(duration, 3)

            print(f"Saved final dialogue: {output_audio}")
            print(f"Saved turn metadata: {turns_csv}")
            print(f"Duration: {duration:.2f} sec, sample rate: {sr}")

        except Exception as exc:
            print(f"FAILED row {idx}: {exc}")
            full_df.at[idx, "generation_status"] = "FAILED_DIALOGUE_GENERATION"

        full_df.to_csv(plan_path, index=False, encoding="utf-8-sig")

    print("\nUpdated generation plan:")
    print(plan_path)


if __name__ == "__main__":
    main()
