create_pilot_generation_plan.py
#!/usr/bin/env python3
"""
Create the Fish-Speech / Chatterbox pilot generation plan.

Input requirement
-----------------
This script expects an Excel file that already contains explicit speaker labels,
normally produced by tools/add_speaker_labels.py.

It does NOT infer or repair speakers.
"""

import argparse
import re
from pathlib import Path
import pandas as pd


# Use only real Coast Guard/operator voice references here.
# Do NOT put vessel distress voices here.
COAST_GUARD_REFS = [
    "workspace/docs/Noisy_dataset/reference_voices/Northern Belle_ coast_guard.wav",
    "workspace/docs/Noisy_dataset/reference_voices/coast_guard_noisy.wav",
]

REFERENCE_TEXTS = {
      # -------------------------
    # COAST GUARD
    # -------------------------
     "workspace/docs/Noisy_dataset/reference_voices/Northern Belle_ coast_guard.wav":
        "Roger understand the vessel's name is northen bell"
        "break break vessel northern bell vessel northern bell"
        "the city United states coast guard kodak"
        "Alaska communication station on 425",

     "workspace/docs/Noisy_dataset/reference_voices/coast_guard_noisy.wav" :
        "mayday yacht delias this is belfast coast guard"
        "received your mayday regarding the motorboat voyager"
        "standby on this channel over",

    # =========================================================
    # NOISY VESSEL REFERENCES
    # =========================================================

    "workspace/docs/Noisy_dataset/reference_voices/Northern_Belle.wav":
        "MAYDAY! MAYDAY! MAYDAY! "
        "This is the NORTHERN BELL! The NORTHERN BELL! "
        "We are at five nine degrees, one zero minutes North! "
        "One four six degrees, four seven minutes West! "
        "This is the NORTHERN BELL! "
        "MAYDAY! MAYDAY! MAYDAY!",

    "workspace/docs/Noisy_dataset/reference_voices/(51) Audio captures captain’s distress call after Iran fires on Indian tanker in Hormuz.wav":
        "You gave me clearance to go"
        "You are firing now"
        "Let me turn back"
        "Sepah Navy, motor tanker SANMAR HERALD, SANMAR HERALD!"
        "Sepah Navy, you gave me clearance!"
        "You gave me clearance to go!"
        "you gave me clearance to go!",

    "workspace/docs/Noisy_dataset/reference_voices/(54) Boaters mayday call.wav":
        "We got hit by another vessel!"
        "We are out in front of Nine Mile Tower!"
        "Real close to Nine Mile Tower!"
        "We are taking on water heavily!"
        "We are sinking!"
        "We are sinking really quick!",

    "workspace/docs/Noisy_dataset/reference_voices/(51) Mayday Call from Fishing Vessel Off New Hampshire.wav":
        "United States Coast Guard! United States Coast Guard!"
        "This is the fishing vessel THREE GIRLS!"
        "My position is Four Three Zero Six, Six Eight Zero Zero decimal One!"
        "We are on fire!"
        "We are going to be abandoning ship!",

    "workspace/docs/Noisy_dataset/reference_voices/(51)Mayday call jet blue airline landing medical emergency but atc stop him #airplane #atcmemes.wav":
        "MAYDAY! MAYDAY! MAYDAY!"
        "JETBLUE Six One One declaring a medical emergency!"
        "We have an eight-year-old pediatric passenger in septic shock!"
        "Cold, pale, unresponsive!"
        "Blood pressure collapsing!"
        "Request immediate vectors and priority!"
        "Straight-in landing to Washington D C!",

    "workspace/docs/Noisy_dataset/reference_voices/(54) Mayday call from capsized vessel on Saginaw Bay.wav":
        "Vessel taking on water!"
        "Approximately one mile east of the Spark Plug!"
        "Under power, heading into the Saginaw River!"
        "This is SIGNAL POWER!"
        "We are going down!"
        "I repeat, we are going down!",

    "workspace/docs/Noisy_dataset/reference_voices/(51) Yacht's mayday call_ 'We are sinking'.wav":
        "MAYDAY! MAYDAY! MAYDAY!"
        "This is motor yacht SERENA THREE!"
        "We are sinking!"   
        "We are sinking!"
        "We are sinking!",

    "workspace/docs/Noisy_dataset/reference_voices/French coastguard's mayday call after boat capsized – audio.wav":
        "MAYDAY RELAY! MAYDAY RELAY! MAYDAY RELAY!"
        "This is GRIS-NEZ EMERGENCY!"
        "GRIS-NEZ EMERGENCY! GRIS-NEZ EMERGENCY!"
        "Fifteen man overboard, approximately!" 
        "Fifteen man overboard!"
        "All ships in this area are requested to have a sharp lookout!"
        "Proceed to this area!"
        "Take contact and report any information to GRIS-NEZ EMERGENCY!"
        "coordinating this operation!",

    "workspace/docs/Noisy_dataset/reference_voices/WhatsApp_audio_1.wav":
        "Under"
       "MAYDAY! MAYDAY! MAYDAY!"
        "M S C ARIES! M S C ARIES! M S C ARIES!"
        "Call sign: Charlie Quebec Echo Bravo Four!"
        "Repeat, Charlie Quebec Echo Bravo Four!"
        "Vessel under missile attack!"
        "Vessel under missile attack!"
        "Location: Bandar Abbas anchorage!"   
        "MAYDAY! MAYDAY!"
        "S S B intercept!",

    "workspace/docs/Noisy_dataset/reference_voices/(135) City Sailing SRC VHF Radio Course May Day Distress Call tutorial.wav":
        "mayday mayday mayday this is duet duet"
        "duet call sign Mike alpha bravo charlie"
        "eight mmsi two three two one two three four five six",
    
    "workspace/docs/Noisy_dataset/reference_voices/(135) Mayday Relay for sinking vessel on HF 8291 kHz in Australian bush.wav":
        "47 utc july one eight incident two zero one eight oblique four eight six one "
        "chart of eight two seven cape bowling green to palm isle"
        "one one meter steel crusier gonna go"
        "with four persons on board taking on water and sinking in position one eight"
        "three nine decimal zero zero shelf"
        "one four seven zero two decimal three zero east ",

    "workspace/docs/Noisy_dataset/reference_voices/(135) Mein Schiff 7_ Erfolgreiche Übung auf der Ostsee.wav":
        "excerise exercise exercise this is Mein Schiff 7"
        "We had a collision with Motorschiff Hoffnung"
        "We request immediate medical assistance as we expect immense casualty incident",

    # =========================================================
    # CLEAN VESSEL REFERENCES
    # =========================================================

    "workspace/docs/clean_dataset/reference_voices/(41) Practical Boating MBM How to do a mayday DSC Motor Boats Monthly MBM.wav": 
        "MAYDAY! MAYDAY! MAYDAY!"
        "This is motorboat BLUEFIN! Motorboat BLUEFIN! Motorboat BLUEFIN!"
        "M M S I number: Two Three Five Zero Zero!",

    "workspace/docs/clean_dataset/reference_voices/(54) Example Mayday Message.wav":
        "MAYDAY! MAYDAY! MAYDAY!"
        "This is motor vessel KIEV! Motor vessel KIEV! Motor vessel KIEV!"
        "Call sign: Hector Victor Golf Three Two One!"
        "M M S I number: Two Three Two Zero Nine Nine Eight Zero Three!"
        "MAYDAY!"
        "Motor vessel KIEV!",

    "workspace/docs/clean_dataset/reference_voices/(135) ScotSail VHF Radio Licence - Mayday Relay Voice Call.wav":
        "mayday relay mayday relay mayday relay"
        "this is yacht dilius yacht dilius yacht dilius"
        "call sign:zulu tango echo eight"
        "golf mmsi two three five eight niner niner niner eight six",

    "workspace/docs/clean_dataset/reference_voices/(135) VHF Mayday call.wav":
        "mayday mayday mayday"
        "this is yacht walhalla yacht walhalla yacht walhalla"
        "mmsi two three five eight nine nine nine eight one "
        "call sign: alpha lima november golf",
     
    "workspace/docs/clean_dataset/reference_voices/WhatsApp Audio 2026-08-01 at 08.13.34.wav":
        "MAYDAY! MAYDAY! MAYDAY!"
        "This is NEW ADDEN! NEW ADDEN! NEW ADDEN!"
        "Our call sign is Victor Romeo Uniform Zero Two!"
        "M M S I number: Four Seven Seven Eight Three Three Four Zero Zero!"
        "Our position is Three Two degrees",

}

# Clean vessel/source reference voices
CLEAN_REFS = [
    "workspace/docs/clean_dataset/reference_voices/(41) Practical Boating MBM How to do a mayday DSC Motor Boats Monthly MBM.wav",
    "workspace/docs/clean_dataset/reference_voices/(54) Example Mayday Message.wav",
    "workspace/docs/clean_dataset/reference_voices/(135) ScotSail VHF Radio Licence - Mayday Relay Voice Call.wav",
    "workspace/docs/clean_dataset/reference_voices/(135) VHF Mayday call.wav",
    "workspace/docs/clean_dataset/reference_voices/WhatsApp Audio 2026-08-01 at 08.13.34.wav",
]

# Noisy vessel/source reference voices
NOISY_REFS = [
    "workspace/docs/Noisy_dataset/reference_voices/(51) Audio captures captain’s distress call after Iran fires on Indian tanker in Hormuz.wav",
    "workspace/docs/Noisy_dataset/reference_voices/(51) Mayday Call from Fishing Vessel Off New Hampshire.wav",
    "workspace/docs/Noisy_dataset/reference_voices/(51) Yacht's mayday call_ 'We are sinking'.wav",
    "workspace/docs/Noisy_dataset/reference_voices/(51)Mayday call jet blue airline landing medical emergency but atc stop him #airplane #atcmemes.wav",
    "workspace/docs/Noisy_dataset/reference_voices/(54) Boaters mayday call.wav",
    "workspace/docs/Noisy_dataset/reference_voices/(54) Mayday call from capsized vessel on Saginaw Bay.wav",
    "workspace/docs/Noisy_dataset/reference_voices/French coastguard's mayday call after boat capsized – audio.wav",
    "workspace/docs/Noisy_dataset/reference_voices/Northern_Belle.wav",
    "workspace/docs/Noisy_dataset/reference_voices/WhatsApp_audio_1.wav",
    "workspace/docs/Noisy_dataset/reference_voices/(135) City Sailing SRC VHF Radio Course May Day Distress Call tutorial.wav",
    "workspace/docs/Noisy_dataset/reference_voices/(135) Mayday Relay for sinking vessel on HF 8291 kHz in Australian bush.wav",
    "workspace/docs/Noisy_dataset/reference_voices/(135) Mein Schiff 7_ Erfolgreiche Übung auf der Ostsee.wav",
]

CONDITIONS = [
    {
        "condition": "chatterbox_clean_ref",
        "model": "Chatterbox",
        "reference_condition": "clean_ref",
        "source_reference_dataset": "clean_dataset",
        "refs": CLEAN_REFS,
        "out_dir": "workspace/data/pilot/chatterbox_clean_ref",
    },
    {
        "condition": "fishspeech_clean_ref",
        "model": "Fish-Speech",
        "reference_condition": "clean_ref",
        "source_reference_dataset": "clean_dataset",
        "refs": CLEAN_REFS,
        "out_dir": "workspace/data/pilot/fishspeech_clean_ref",
    },
    {
        "condition": "chatterbox_noisy_ref",
        "model": "Chatterbox",
        "reference_condition": "noisy_ref",
        "source_reference_dataset": "Noisy_dataset",
        "refs": NOISY_REFS,
        "out_dir": "workspace/data/pilot/chatterbox_noisy_ref",
    },
    {
        "condition": "fishspeech_noisy_ref",
        "model": "Fish-Speech",
        "reference_condition": "noisy_ref",
        "source_reference_dataset": "Noisy_dataset",
        "refs": NOISY_REFS,
        "out_dir": "workspace/data/pilot/fishspeech_noisy_ref",
    },
]


def is_missing(value):
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return str(value).strip() == ""


def get_optional_column(row, column_name):
    if column_name in row.index and not is_missing(row[column_name]):
        return str(row[column_name]).strip()
    return ""

def get_reference_text(audio_path):
    text = REFERENCE_TEXTS.get(audio_path, "").strip()

    if not text:
        raise ValueError(
            f"No reference transcript configured for:\n{audio_path}\n"
            "Add this audio and its exact transcript to REFERENCE_TEXTS."
        )

    return text

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


def validate_and_normalize_labeled_dialogue(text, text_id):
    """
    Validate explicit speaker labels and preserve consecutive same-speaker turns.

    No speaker inference and no alternation logic are performed.
    """
    text = str(text).strip()

    if not text:
        raise ValueError(f"Text ID {text_id}: labeled dialogue is empty.")

    if "REVIEW_REQUIRED:" in text:
        raise ValueError(
            f"Text ID {text_id}: speaker_labeled_text still contains "
            "REVIEW_REQUIRED. Resolve the flagged turn before plan generation."
        )

    label_pattern = re.compile(
        r"^\s*(?:\[(?P<bracket>[^\]]+)\]|"
        r"(?P<label>[A-Za-z0-9 _-]+):)\s*(?P<text>.*)$"
    )

    turns = []
    current_speaker = None
    current_text = []

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        match = label_pattern.match(line)

        if match:
            raw_speaker = match.group("bracket") or match.group("label")
            speaker = canonical_speaker(raw_speaker)

            if speaker not in {"VESSEL", "COAST_GUARD"}:
                raise ValueError(
                    f"Text ID {text_id}, line {line_number}: "
                    f"unknown speaker label '{raw_speaker}'."
                )

            if current_speaker is not None:
                turn_text = " ".join(current_text).strip()
                if not turn_text:
                    raise ValueError(
                        f"Text ID {text_id}: empty text for speaker {current_speaker}."
                    )
                turns.append((current_speaker, turn_text))

            current_speaker = speaker
            first_text = match.group("text").strip()
            current_text = [first_text] if first_text else []
            continue

        # Allow continuation lines only after an explicit speaker label.
        if current_speaker is None:
            raise ValueError(
                f"Text ID {text_id}, line {line_number}: text occurs before any "
                f"VESSEL:/COAST_GUARD: label:\n{line}"
            )

        current_text.append(line)

    if current_speaker is not None:
        turn_text = " ".join(current_text).strip()
        if not turn_text:
            raise ValueError(
                f"Text ID {text_id}: empty text for speaker {current_speaker}."
            )
        turns.append((current_speaker, turn_text))

    if not turns:
        raise ValueError(f"Text ID {text_id}: no labeled dialogue turns found.")

    return "\n".join(f"{speaker}: {turn_text}" for speaker, turn_text in turns)

def add_fish_emotion_tags(dialogue_text):
    """
    Add stable role-specific emotion tags for Fish-Speech.

    All Vessel turns use the same style.
    All Coast Guard turns use the same style.
    """

    output_lines = []

    text = (
        str(dialogue_text)
        .replace("<br>", "\n")
        .replace("\r\n", "\n")
        .strip()
    )

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        if line.upper().startswith("VESSEL:"):
            turn_text = line.split(":", 1)[1].strip()

            output_lines.append(
                "VESSEL: "
                "[urgent, controlled distress] "
                f"{turn_text}"
            )

        elif re.match(
            r"^COAST[\s_-]*GUARD\s*:",
            line,
            flags=re.IGNORECASE,
        ):
            turn_text = line.split(":", 1)[1].strip()

            output_lines.append(
                "COAST_GUARD: "
                "[calm, clear, authoritative] "
                f"{turn_text}"
            )

        else:
            raise ValueError(
                "Unrecognized speaker-labelled line:\n"
                f"{line}"
            )

    return "\n".join(output_lines)

ALL_REQUIRED_REFS = CLEAN_REFS + NOISY_REFS + COAST_GUARD_REFS

for ref in ALL_REQUIRED_REFS:
    if ref not in REFERENCE_TEXTS:
        raise ValueError(
            f"Missing REFERENCE_TEXTS entry for:\n{ref}"
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_xlsx", required=True)
    parser.add_argument(
        "--output_csv",
        default="workspace/data/pilot/metadata/pilot_generation_plan.csv",
    )
    parser.add_argument(
        "--text_column",
        default="speaker_labeled_text",
        help="Use speaker_labeled_text produced by add_speaker_labels.py.",
    )

    parser.add_argument("--num_texts", type=int, default=100)

    # Optional row-range selection
    parser.add_argument("--start_row", type=int, default=None)
    parser.add_argument("--end_row", type=int, default=None)

    args = parser.parse_args()

    df = pd.read_excel(args.input_xlsx)

    if args.text_column not in df.columns:
        raise ValueError(
            f"Missing text column: {args.text_column}. "
            "Run tools/add_speaker_labels.py first."
        )

    if "id" not in df.columns:
        raise ValueError("Missing required column: id")

    if args.start_row is not None or args.end_row is not None:
        if args.start_row is None or args.end_row is None:
            raise ValueError("Both --start_row and --end_row must be provided.")

        if args.start_row < 1:
            raise ValueError("--start_row must be >= 1.")

        if args.end_row < args.start_row:
            raise ValueError(
                "--end_row must be greater than or equal to --start_row."
            )

        df = df.iloc[args.start_row - 1 : args.end_row]
    else:
        df = df.head(args.num_texts)

    rows = []

    for _, row in df.iterrows():
        text_id = int(row["id"])

        # If the labeler status column exists, enforce it.
        if (
            "speaker_label_status" in row.index
            and not is_missing(row["speaker_label_status"])
            and str(row["speaker_label_status"]).strip().upper() != "OK"
        ):
            raise ValueError(
                f"Text ID {text_id}: speaker_label_status is "
                f"'{row['speaker_label_status']}'. Resolve this row before "
                "creating the TTS generation plan."
            )

        raw_text = str(row[args.text_column]).strip()
        dialogue_text = validate_and_normalize_labeled_dialogue(raw_text, text_id)

        scenario_type = get_optional_column(row, "scenario_type")
        keywords = get_optional_column(row, "keywords")
        numbers = get_optional_column(row, "numbers")

        for cfg in CONDITIONS:
            refs = cfg["refs"]

            # Vessel/source reference rotates across the selected dataset.
            ref_audio = refs[(text_id - 1) % len(refs)]

            vessel_reference_audio = ref_audio
            coast_guard_reference_audio = COAST_GUARD_REFS
            vessel_reference_text = get_reference_text(
                vessel_reference_audio
            )

            coast_guard_reference_text = get_reference_text(
                coast_guard_reference_audio
            )

            # Fish-Speech receives emotion tags.
            # Chatterbox receives the original dialogue.
            if cfg["model"] == "Fish-Speech":
                target_text = add_fish_emotion_tags(dialogue_text)
            else:
                target_text = dialogue_text

            sample_id = f"{cfg['condition']}_{text_id:03d}"
            output_audio = f"{cfg['out_dir']}/{sample_id}.wav"

            rows.append(
    {
        "sample_id": sample_id,
        "text_id": text_id,
        "scenario_type": scenario_type,
        "condition": cfg["condition"],
        "model": cfg["model"],
        "reference_condition": cfg["reference_condition"],
        "source_reference_dataset": cfg["source_reference_dataset"],

        # Compatibility column
        "reference_audio": ref_audio,

        # Two speaker references
        "vessel_reference_audio": vessel_reference_audio,
        "coast_guard_reference_audio": coast_guard_reference_audio,

        # Original dialogue
        "dialogue_text": dialogue_text,

        # Fish-Speech gets emotion tags;
        # Chatterbox gets normal text
        "target_text": target_text,

        # Reference transcripts
        "vessel_reference_text": vessel_reference_text,
        "coast_guard_reference_text": coast_guard_reference_text,

        "keywords": keywords,
        "numbers": numbers,
        "output_audio": output_audio,
        "generation_status": "PENDING",
    }
)

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    for cfg in CONDITIONS:
        Path(cfg["out_dir"]).mkdir(parents=True, exist_ok=True)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"Saved generation plan: {out}")
    print(f"Rows: {len(result_df)}")

    print("\nRows per condition:")
    print(result_df["condition"].value_counts())

    print("\nReference datasets used:")
    print(result_df["source_reference_dataset"].value_counts())

    print("\nSpeaker handling:")
    print("- speaker labels were read from speaker_labeled_text")
    print("- no speaker alternation was assumed")
    print("- no speaker guessing/repair was performed")


if __name__ == "__main__":
    main()
