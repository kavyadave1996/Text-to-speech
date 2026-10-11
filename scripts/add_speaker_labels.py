#!/usr/bin/env python3
"""
Add explicit speaker labels to maritime distress dialogues.

Purpose
-------
The source Excel contains dialogue text but no speaker metadata. This script is
the ONLY stage in the pipeline that infers speaker identity.

It:
1. reads `clarity_formatted_text` (configurable),
2. splits the dialogue at VHF turn endings such as "Over." / "Out.",
3. labels only turns with sufficiently strong semantic evidence,
4. NEVER assumes strict VESSEL -> COAST_GUARD alternation,
5. marks ambiguous/mixed turns as REVIEW_REQUIRED instead of silently guessing,
6. writes a new Excel containing `speaker_labeled_text`, plus a detailed
   `speaker_label_report` sheet for inspection.

Downstream scripts must only consume rows whose labels are resolved.
"""

import argparse
import re
from pathlib import Path

import pandas as pd


VESSEL = "VESSEL"
COAST_GUARD = "COAST_GUARD"
REVIEW = "REVIEW_REQUIRED"


def normalize(text):
    return re.sub(r"\s+", " ", str(text).strip())


def split_dialogue_turns(text):
    """
    Split a plain VHF dialogue after 'Over' or 'Out', keeping the marker
    in the preceding turn.

    No speaker alternation is assumed here.
    """
    normalized = normalize(text)
    if not normalized:
        return []

    pieces = []
    start = 0

    # Match Over / Out and punctuation. The look-ahead makes this also work
    # when the final marker is at the end of the text.
    pattern = re.compile(r"(?i)\b(?:over|out)\b[.!?…]*")

    for match in pattern.finditer(normalized):
        end = match.end()
        piece = normalized[start:end].strip()
        if piece:
            pieces.append(piece)
        start = end

    tail = normalized[start:].strip()
    if tail:
        pieces.append(tail)

    return pieces


def add_reason(reasons, speaker, reason, points):
    reasons.append(f"{speaker}+{points}:{reason}")


def score_speaker(turn_text, turn_index):
    """
    Return semantic evidence scores for both speakers.

    These are role/context cues, not an alternation rule.
    """
    low = normalize(turn_text).lower()

    cg = 0
    vessel = 0
    reasons = []

    # -----------------------
    # Very strong identifiers
    # -----------------------
    if "this is coast guard" in low or "this is the coast guard" in low:
        cg += 12
        add_reason(reasons, "CG", "explicit Coast Guard identification", 12)

    if low.startswith("mayday"):
        vessel += 12
        add_reason(reasons, "VESSEL", "Mayday distress originator", 12)

    vessel_identity_patterns = [
        r"\bthis is (?:motor vessel|tanker|vessel|mv|m/v|sailing vessel|yacht|towing vessel)\b",
        r"\bmmsi\b",
        r"\bcall sign\b",
    ]
    for pattern in vessel_identity_patterns:
        if re.search(pattern, low):
            vessel += 6
            add_reason(reasons, "VESSEL", f"vessel identity cue: {pattern}", 6)

    # Directly addressing Coast Guard is strong evidence that the current
    # transmitter is the vessel.
    if re.search(
        r"^(?:roger|copy|negative|affirmative|yes|no|understood)?[,. ]*coast guard\b",
        low,
    ):
        vessel += 9
        add_reason(reasons, "VESSEL", "turn directly addresses Coast Guard", 9)

    if re.search(
        r"\b(?:roger|copy|negative|affirmative|understood),?\s+coast guard\b",
        low,
    ):
        vessel += 9
        add_reason(reasons, "VESSEL", "acknowledgement addressed to Coast Guard", 9)

    if low.startswith("thank you, coast guard") or low.startswith("thank you coast guard"):
        vessel += 9
        add_reason(reasons, "VESSEL", "thanks addressed to Coast Guard", 9)

    # -----------------------
    # Coast Guard role cues
    # -----------------------
    cg_phrases = [
        "we have received your distress call",
        "what is your position",
        "what is the current situation",
        "what is the status",
        "what is the nature",
        "what is the condition",
        "are there any injuries",
        "are you able to provide",
        "can you provide",
        "provide your position",
        "provide more details",
        "confirm your position",
        "confirm the number",
        "report your position",
        "report your status",
        "stand by",
        "remain on this frequency",
        "stay on this channel",
        "switch to channel",
        "continue evasive",
        "maintain radio silence",
        "maintain communication",
        "hold tight",
        "keep all crew",
        "keep your crew",
        "keep the radio silent",
        "stay safe",
        "assistance is on the way",
        "assistance is being dispatched",
        "assistance is being sent",
        "we are dispatching",
        "we have dispatched",
        "has been dispatched",
        "is being dispatched",
        "naval vessel is en route",
        "patrol vessel is en route",
        "helicopter is en route",
        "lifeboat is en route",
        "rescue boat is en route",
        "patrol boat is en route",
        "patrol aircraft is being dispatched",
        "aerial assistance",
        "additional support is on the way",
        "we will provide support",
        "we will attempt communication",
        "await further instructions",
    ]

    for phrase in cg_phrases:
        if phrase in low:
            cg += 4
            add_reason(reasons, "CG", phrase, 4)

    # Questions in these distress exchanges are usually requests from the
    # operator/Coast Guard. This is supporting evidence only.
    if "?" in turn_text:
        cg += 3
        add_reason(reasons, "CG", "information-request question", 3)

    # Imperative/operator language at the start of a transmission.
    if re.search(
        r"^(?:roger[,.]?\s*)?(?:understood[,.]?\s*)?"
        r"(?:maintain|continue|remain|stay|stand by|hold|keep|switch|report|confirm|provide)\b",
        low,
    ):
        cg += 5
        add_reason(reasons, "CG", "operator instruction", 5)

    # Rescue resource + ETA is strong Coast Guard evidence.
    if "eta" in low and any(
        phrase in low
        for phrase in [
            "dispatch",
            "en route",
            "assistance",
            "patrol",
            "naval",
            "helicopter",
            "aircraft",
            "support",
            "aerial",
        ]
    ):
        cg += 5
        add_reason(reasons, "CG", "rescue asset/ETA information", 5)

    # -----------------------
    # Vessel role cues
    # -----------------------
    vessel_phrases = [
        "under attack",
        "request immediate assistance",
        "requesting immediate assistance",
        "we require assistance",
        "we need assistance",
        "no injuries",
        "injuries on board",
        "we have injuries",
        "we have casualties",
        "we have no casualties",
        "pirates",
        "intruders",
        "attackers",
        "assailants",
        "citadel",
        "we are sinking",
        "taking on water",
        "flooding",
        "fire on board",
        "engine failure",
        "not under command",
        "running low on fuel",
        "low on fuel",
        "fire suppression",
        "we are evasive",
        "evasive maneuvers",
        "evasive manoeuvres",
        "crew is secured",
        "crew are secured",
        "we are defending",
        "we are attempting to secure",
        "we have secured",
        "we are able to fight back",
        "taken necessary measures",
        "unable to make any distress calls",
        "we are unable",
        "we cannot",
        "i am alone",
        "awaiting assistance",
        "we will maintain radio silence",
        "we will maintain radio contact",
    ]

    for phrase in vessel_phrases:
        if phrase in low:
            vessel += 4
            add_reason(reasons, "VESSEL", phrase, 4)

    # Position reports containing coordinates are usually supplied by vessel.
    if "position" in low and any(
        x in low
        for x in ["degrees", "minutes north", "minutes south", " north", " south", " east", " west"]
    ):
        vessel += 4
        add_reason(reasons, "VESSEL", "position/coordinate report", 4)

    # First turn is only strengthened as vessel if it already resembles a
    # distress transmission; this is not a generic "first turn = vessel" rule.
    if turn_index == 1 and any(
        cue in low
        for cue in ["mayday", "mmsi", "under attack", "request immediate assistance"]
    ):
        vessel += 3
        add_reason(reasons, "VESSEL", "first turn contains distress-originator cues", 3)

    # Acknowledgement followed by a vessel commitment/status is a common
    # vessel response, e.g. "Copy. We will maintain radio silence."
    if low.startswith(("roger", "copy", "understood")) and re.search(r"\bwe will\b", low):
        if not any(
            x in low
            for x in [
                "we will provide support",
                "we will attempt communication",
                "we will dispatch",
            ]
        ):
            vessel += 4
            add_reason(reasons, "VESSEL", "acknowledgement followed by vessel commitment", 4)

    # Weak status/report evidence. Do not use this by itself to force a label.
    if re.search(r"\bwe (?:are|have|cannot|need|require|will)\b", low):
        if not any(
            x in low
            for x in [
                "we are dispatching",
                "we have dispatched",
                "we have received your distress call",
                "we will provide support",
                "we will attempt communication",
            ]
        ):
            vessel += 1
            add_reason(reasons, "VESSEL", "first-person onboard/status language", 1)

    return cg, vessel, reasons


def classify_turn(turn_text, turn_index):
    """
    Conservative decision:
      - strong, clearly separated score -> label
      - mixed or weak evidence -> REVIEW_REQUIRED

    This deliberately prefers review over a wrong voice assignment.
    """
    cg, vessel, reasons = score_speaker(turn_text, turn_index)

    top = max(cg, vessel)
    margin = abs(cg - vessel)

    # Strong mixed evidence is particularly dangerous: the source sentence may
    # itself contain conflicting role language.
    if cg >= 5 and vessel >= 5 and margin < 4:
        return REVIEW, cg, vessel, reasons, "mixed Coast Guard and vessel evidence"

    if cg >= 4 and cg >= vessel + 3:
        return COAST_GUARD, cg, vessel, reasons, "strong Coast Guard evidence"

    if vessel >= 4 and vessel >= cg + 3:
        return VESSEL, cg, vessel, reasons, "strong vessel evidence"

    return REVIEW, cg, vessel, reasons, "insufficient or ambiguous evidence"


def label_dialogue(text, text_id):
    turns = split_dialogue_turns(text)

    if not turns:
        return "", "REVIEW_REQUIRED", [
            {
                "id": text_id,
                "turn_id": 0,
                "assigned_speaker": REVIEW,
                "coast_guard_score": 0,
                "vessel_score": 0,
                "status": "REVIEW_REQUIRED",
                "decision": "no dialogue turns detected",
                "reasons": "",
                "turn_text": "",
            }
        ]

    labeled_lines = []
    report_rows = []
    needs_review = False

    for turn_index, turn_text in enumerate(turns, start=1):
        speaker, cg_score, vessel_score, reasons, decision = classify_turn(
            turn_text, turn_index
        )

        if speaker == REVIEW:
            needs_review = True

        labeled_lines.append(f"{speaker}: {turn_text}")

        report_rows.append(
            {
                "id": text_id,
                "turn_id": turn_index,
                "assigned_speaker": speaker,
                "coast_guard_score": cg_score,
                "vessel_score": vessel_score,
                "status": "REVIEW_REQUIRED" if speaker == REVIEW else "OK",
                "decision": decision,
                "reasons": " | ".join(reasons),
                "turn_text": turn_text,
            }
        )

    row_status = "REVIEW_REQUIRED" if needs_review else "OK"
    return "\n".join(labeled_lines), row_status, report_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_xlsx", required=True)
    parser.add_argument(
        "--output_xlsx",
        default="workspace/data/pilot/metadata/maritime_distress_speaker_labeled.xlsx",
    )
    parser.add_argument("--text_column", default="clarity_formatted_text")
    parser.add_argument("--num_texts", type=int, default=None)
    args = parser.parse_args()

    input_path = Path(args.input_xlsx)
    if not input_path.exists():
        raise FileNotFoundError(f"Input Excel not found: {input_path}")

    df = pd.read_excel(input_path)

    if "id" not in df.columns:
        raise ValueError("Missing required column: id")

    if args.text_column not in df.columns:
        raise ValueError(f"Missing text column: {args.text_column}")

    if args.num_texts is not None:
        df = df.head(args.num_texts).copy()
    else:
        df = df.copy()

    labeled_values = []
    statuses = []
    all_report_rows = []

    for _, row in df.iterrows():
        text_id = int(row["id"])
        source_text = "" if pd.isna(row[args.text_column]) else str(row[args.text_column])

        labeled_text, status, report_rows = label_dialogue(source_text, text_id)

        labeled_values.append(labeled_text)
        statuses.append(status)
        all_report_rows.extend(report_rows)

    df["speaker_labeled_text"] = labeled_values
    df["speaker_label_status"] = statuses

    report_df = pd.DataFrame(all_report_rows)

    output_path = Path(args.output_xlsx)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="dialogues")
        report_df.to_excel(writer, index=False, sheet_name="speaker_label_report")

    ok_rows = int((df["speaker_label_status"] == "OK").sum())
    review_rows = int((df["speaker_label_status"] != "OK").sum())
    review_turns = int((report_df["status"] == "REVIEW_REQUIRED").sum())

    print(f"Saved speaker-labeled workbook: {output_path}")
    print(f"Dialogue rows: {len(df)}")
    print(f"Rows fully resolved: {ok_rows}")
    print(f"Rows requiring review: {review_rows}")
    print(f"Individual turns requiring review: {review_turns}")

    if review_rows:
        print(
            "\nIMPORTANT: Do not generate TTS for REVIEW_REQUIRED rows yet. "
            "Open the 'speaker_label_report' sheet, inspect the flagged turns, "
            "and replace REVIEW_REQUIRED: with VESSEL: or COAST_GUARD: in "
            "speaker_labeled_text. Then set speaker_label_status to OK."
        )


if __name__ == "__main__":
    main()
