#!/usr/bin/env python3
"""Normalize maritime distress data into one multi-scenario Excel workbook.

Supports:
- the current mixed workbook format (first formatted rows + one or more embedded JSON blocks), and
- additional standalone JSON task files with structure {task_name, results:[...]}.

Output columns are designed for the downstream speaker-label/TTS pipeline.
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def normalize_scenario_name(task_name: str) -> str:
    value = str(task_name or "unknown").strip().lower()
    value = re.sub(r"^reporting_", "", value)
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value or "unknown"


def split_after_over_out(text: str):
    normalized = re.sub(r"\s+", " ", str(text).strip())
    if not normalized:
        return []

    pieces = []
    start = 0

    for match in re.finditer(r"(?i)\b(?:over|out)\b[.!?…]*", normalized):
        end = match.end()
        piece = normalized[start:end].strip()
        if piece:
            pieces.append(piece)
        start = end

    tail = normalized[start:].strip()
    if tail:
        pieces.append(tail)

    return pieces


def records_from_task_json(payload, source_name):
    if not isinstance(payload, dict):
        return []

    scenario = normalize_scenario_name(payload.get("task_name"))
    results = payload.get("results", [])

    if not isinstance(results, list):
        return []

    rows = []

    for pos, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            continue

        try:
            source_id = int(item.get("id", pos))
        except (TypeError, ValueError):
            source_id = pos

        raw_output = item.get("output") or []

        if isinstance(raw_output, str):
            turns = split_after_over_out(raw_output)
            if not turns and raw_output.strip():
                turns = [raw_output.strip()]
        elif isinstance(raw_output, list):
            turns = [
                str(x).strip()
                for x in raw_output
                if x is not None and str(x).strip()
            ]
        else:
            turns = []

        rows.append({
            "scenario_id": f"{scenario}_{source_id:03d}",
            "scenario_type": scenario,
            "source_id": source_id,
            "source_origin": source_name,
            "scenario_label_source": f"explicit:{payload.get('task_name', '')}",
            "text": " ".join(turns),
            "clarity_formatted_text": "\n\n".join(turns),
            "natural_prosody_text": "",
            "format_status": "NATURAL_PROSODY_NOT_IN_SOURCE",
            "dialogue_turns_json": json.dumps(turns, ensure_ascii=False),
        })

    return rows


def extract_embedded_json_from_dataframe(df):
    """Extract one or more JSON objects embedded line-by-line in column 1.

    Returns:
        list[dict]: every valid JSON object found after the first JSON start.

    This intentionally uses JSONDecoder.raw_decode() instead of json.loads(),
    because json.loads() rejects concatenated JSON documents with:
        JSONDecodeError: Extra data
    """

    first_col = df.iloc[:, 0].tolist()

    start = None
    for i, value in enumerate(first_col):
        if pd.isna(value):
            continue

        stripped = str(value).lstrip()
        if stripped.startswith("{") or stripped.startswith("["):
            start = i
            break

    if start is None:
        return []

    lines = []
    for value in first_col[start:]:
        if pd.isna(value):
            lines.append("")
        else:
            lines.append(str(value))

    raw = "\n".join(lines)
    decoder = json.JSONDecoder()

    payloads = []
    pos = 0
    n = len(raw)

    while pos < n:
        # Skip whitespace and non-JSON garbage between blocks.
        while pos < n and raw[pos].isspace():
            pos += 1

        if pos >= n:
            break

        # Find next JSON object/array opening if current character is not JSON.
        if raw[pos] not in "{[":
            next_obj = raw.find("{", pos + 1)
            next_arr = raw.find("[", pos + 1)

            candidates = [x for x in (next_obj, next_arr) if x != -1]
            if not candidates:
                break

            pos = min(candidates)

        try:
            payload, end = decoder.raw_decode(raw, pos)
        except json.JSONDecodeError:
            # The current '{' or '[' may be ordinary text.
            next_obj = raw.find("{", pos + 1)
            next_arr = raw.find("[", pos + 1)

            candidates = [x for x in (next_obj, next_arr) if x != -1]
            if not candidates:
                break

            pos = min(candidates)
            continue

        if isinstance(payload, dict):
            payloads.append(payload)

        elif isinstance(payload, list):
            # Also tolerate a JSON list containing task dictionaries.
            for item in payload:
                if isinstance(item, dict):
                    payloads.append(item)

        pos = end

    return payloads


def load_extra_json_file(json_path: Path):
    with json_path.open("r", encoding="utf-8-sig") as f:
        raw = f.read()

    decoder = json.JSONDecoder()
    payloads = []
    pos = 0
    n = len(raw)

    while pos < n:
        while pos < n and raw[pos].isspace():
            pos += 1

        if pos >= n:
            break

        payload, end = decoder.raw_decode(raw, pos)

        if isinstance(payload, dict):
            payloads.append(payload)
        elif isinstance(payload, list):
            payloads.extend(x for x in payload if isinstance(x, dict))

        pos = end

    return payloads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_xlsx", required=True)
    parser.add_argument("--base_scenario", default="armed_attack")
    parser.add_argument("--extra_json", action="append", default=[])
    parser.add_argument(
        "--output_xlsx",
        default="workspace/data/pilot/metadata/maritime_distress_normalized.xlsx",
    )
    args = parser.parse_args()

    input_path = Path(args.input_xlsx)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    raw_df = pd.read_excel(input_path, header=None)
    table_df = pd.read_excel(input_path)

    rows = []

    # ------------------------------------------------------------------
    # 1. Existing formatted table
    # ------------------------------------------------------------------
    if "id" in table_df.columns:
        json_start = None

        for pos, value in enumerate(table_df["id"].tolist()):
            if pd.isna(value):
                continue

            stripped = str(value).lstrip()
            if stripped.startswith("{") or stripped.startswith("["):
                json_start = pos
                break

        formatted_region = (
            table_df.iloc[:json_start].copy()
            if json_start is not None
            else table_df.copy()
        )

        numeric = pd.to_numeric(
            formatted_region["id"],
            errors="coerce",
        ).notna()

        formatted = formatted_region[numeric].copy()

        for _, row in formatted.iterrows():
            source_id = int(row["id"])

            text = (
                ""
                if pd.isna(row.get("text"))
                else str(row.get("text"))
            )

            clarity = (
                ""
                if pd.isna(row.get("clarity_formatted_text"))
                else str(row.get("clarity_formatted_text"))
            )

            natural = (
                ""
                if pd.isna(row.get("natural_prosody_text"))
                else str(row.get("natural_prosody_text"))
            )

            turns = split_after_over_out(text)

            rows.append({
                "scenario_id": f"{args.base_scenario}_{source_id:03d}",
                "scenario_type": args.base_scenario,
                "source_id": source_id,
                "source_origin": "existing_formatted_table",
                "scenario_label_source": "provided_by_cli/base_scenario",
                "text": text,
                "clarity_formatted_text": clarity,
                "natural_prosody_text": natural,
                "format_status": "READY",
                "dialogue_turns_json": json.dumps(
                    turns,
                    ensure_ascii=False,
                ),
            })

    # ------------------------------------------------------------------
    # 2. One or more JSON blocks embedded inside the Excel sheet
    # ------------------------------------------------------------------
    embedded_payloads = extract_embedded_json_from_dataframe(raw_df)

    print(
        f"Found {len(embedded_payloads)} embedded JSON block(s) "
        f"in {input_path.name}"
    )

    for idx, payload in enumerate(embedded_payloads, start=1):
        task_name = payload.get("task_name", "unknown")
        result_count = len(payload.get("results", []))

        print(
            f"  Embedded JSON {idx}: "
            f"task_name={task_name}, results={result_count}"
        )

        rows.extend(
            records_from_task_json(
                payload,
                f"embedded_json_{idx}",
            )
        )

    # ------------------------------------------------------------------
    # 3. Optional standalone JSON files
    # ------------------------------------------------------------------
    for json_path_str in args.extra_json:
        json_path = Path(json_path_str)

        if not json_path.exists():
            raise FileNotFoundError(json_path)

        payloads = load_extra_json_file(json_path)

        for idx, payload in enumerate(payloads, start=1):
            rows.extend(
                records_from_task_json(
                    payload,
                    f"{json_path.name}#{idx}",
                )
            )

    if not rows:
        raise ValueError("No usable dialogues found.")

    # ------------------------------------------------------------------
    # 4. Final normalized dataset
    # ------------------------------------------------------------------
    out_df = pd.DataFrame(rows)

    # Duplicate scenario IDs can occur if the same task exists twice.
    duplicates = out_df[
        out_df["scenario_id"].duplicated(keep=False)
    ]

    if not duplicates.empty:
        duplicate_ids = duplicates["scenario_id"].unique().tolist()

        raise ValueError(
            "Duplicate scenario_id values detected. "
            "This usually means the same scenario/task was loaded more than once. "
            f"Examples: {duplicate_ids[:10]}"
        )

    out_df.insert(
        0,
        "global_id",
        range(1, len(out_df) + 1),
    )

    summary = (
        out_df.groupby("scenario_type", as_index=False)
        .size()
        .rename(columns={"size": "dialogue_count"})
    )

    output_path = Path(args.output_xlsx)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(
        output_path,
        engine="openpyxl",
    ) as writer:
        out_df.to_excel(
            writer,
            index=False,
            sheet_name="dialogues",
        )

        summary.to_excel(
            writer,
            index=False,
            sheet_name="summary",
        )

    print()
    print(f"Saved normalized dataset: {output_path}")
    print()
    print(summary.to_string(index=False))
    print()
    print(f"Total dialogues: {len(out_df)}")


if __name__ == "__main__":
    main()
