import argparse
import re
from pathlib import Path

import pandas as pd
from jiwer import wer, cer


NUMBER_WORDS = {
    "zero": "0",
    "oh": "0",
    "o": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "niner": "9",
}


CRITICAL_KEYWORDS = {
    "mayday": ["mayday"],
    "vessel_identity": [
        "frb zero niner",
        "f r b zero niner",
        "frb 0 9",
        "frb zero nine",
        "frb09",
        "frb 09",
    ],
    "mmsi": ["mmsi", "m m s i"],
    "under_attack": ["under attack"],
    "armed_men": ["armed men", "shooting at us"],
    "position": ["position"],
    "north_77": ["77 degrees north", "seven seven degrees north"],
    "east_99": ["99 degrees east", "nine nine degrees east"],
    "defending": ["defending the vessel", "defending vessel"],
    "assistance": ["immediate military assistance", "military assistance", "immediate assistance"],
    "over": ["over"],
}


NUMBER_FIELDS = {
    "mmsi_219016758": ["219016758"],
    "latitude_77": ["77"],
    "longitude_99": ["99"],
}


def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove emotion/prosody tags such as [urgent], [panicked]
    text = re.sub(r"\[[^\]]*\]", " ", text)

    # Normalize common ASR variants
    text = text.replace("niner", "nine")
    text = text.replace("zero niner", "zero nine")
    text = text.replace("0 niner", "0 9")

    # Separate letters/numbers where needed
    text = re.sub(r"frb[-\s]*09[a-z]*", "frb zero nine", text)
    text = re.sub(r"frb[-\s]*0[-\s]*9[a-z]*", "frb zero nine", text)

    # Remove punctuation but keep letters/numbers/spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Normalize spacing
    text = re.sub(r"\s+", " ", text).strip()

    return text


def spoken_numbers_to_digits(text):
    """
    Converts number words into digit characters.
    Example:
    'two one nine zero one six seven five eight'
    -> '219016758'
    """
    text = normalize_text(text)
    tokens = text.split()

    digit_tokens = []
    for token in tokens:
        if token in NUMBER_WORDS:
            digit_tokens.append(NUMBER_WORDS[token])
        elif token.isdigit():
            digit_tokens.append(token)

    return "".join(digit_tokens)


def keyword_accuracy_score(text):
    text_norm = normalize_text(text)

    detected = 0
    total = len(CRITICAL_KEYWORDS)
    missing = []

    for key, variants in CRITICAL_KEYWORDS.items():
        found = any(normalize_text(v) in text_norm for v in variants)
        if found:
            detected += 1
        else:
            missing.append(key)

    accuracy = detected / total if total else 0.0
    return round(accuracy * 100, 2), ";".join(missing)


def number_accuracy_score(text):
    digits = spoken_numbers_to_digits(text)

    detected = 0
    total = len(NUMBER_FIELDS)
    missing = []

    for key, variants in NUMBER_FIELDS.items():
        found = any(v in digits for v in variants)
        if found:
            detected += 1
        else:
            missing.append(key)

    accuracy = detected / total if total else 0.0
    return round(accuracy * 100, 2), ";".join(missing)


def get_metric_text(row):
    """
    Priority:
    1. manual_transcript
    2. generated_text_asr
    3. generated_text
    """
    for col in ["manual_transcript", "generated_text_asr", "generated_text"]:
        if col in row and not pd.isna(row[col]) and str(row[col]).strip():
            return str(row[col]), col

    return "", "missing"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input metadata .xlsx or .csv")
    parser.add_argument("--output", required=True, help="Output .xlsx or .csv")
    parser.add_argument("--sheet", default=0, help="Excel sheet name or index. Default: first sheet")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.suffix.lower() == ".xlsx":
        try:
            sheet = int(args.sheet)
        except ValueError:
            sheet = args.sheet
        df = pd.read_excel(input_path, sheet_name=sheet)
    else:
        df = pd.read_csv(input_path)

    required_cols = ["target_text"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    wer_values = []
    cer_values = []
    keyword_values = []
    number_values = []
    metric_text_used_values = []
    metric_notes_values = []

    for _, row in df.iterrows():
        target = normalize_text(row.get("target_text", ""))
        hypothesis_raw, used_col = get_metric_text(row)
        hypothesis = normalize_text(hypothesis_raw)

        metric_text_used_values.append(used_col)

        if not target or not hypothesis:
            wer_values.append("")
            cer_values.append("")
            keyword_values.append("")
            number_values.append("")
            metric_notes_values.append("Missing target_text or transcript")
            continue

        row_wer = round(wer(target, hypothesis) * 100, 2)
        row_cer = round(cer(target, hypothesis) * 100, 2)

        key_acc, missing_keywords = keyword_accuracy_score(hypothesis)
        num_acc, missing_numbers = number_accuracy_score(hypothesis)

        notes = []
        if missing_keywords:
            notes.append(f"missing_keywords={missing_keywords}")
        if missing_numbers:
            notes.append(f"missing_numbers={missing_numbers}")

        wer_values.append(row_wer)
        cer_values.append(row_cer)
        keyword_values.append(key_acc)
        number_values.append(num_acc)
        metric_notes_values.append(" | ".join(notes) if notes else "OK")

    df["wer"] = wer_values
    df["cer"] = cer_values
    df["keyword_accuracy"] = keyword_values
    df["number_accuracy"] = number_values
    df["metric_text_used"] = metric_text_used_values
    df["metric_notes"] = metric_notes_values

    if output_path.suffix.lower() == ".xlsx":
        df.to_excel(output_path, index=False)
    else:
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()