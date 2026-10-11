import argparse
import subprocess
from pathlib import Path

import pandas as pd


def run_command(cmd):
    print("\nRunning:")
    print(" ".join(str(x) for x in cmd))
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        default="workspace/data/pilot/metadata/pilot_generation_plan.csv",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fish_url", default="http://127.0.0.1:8080/v1/tts")
    args = parser.parse_args()

    plan_path = Path(args.plan)

    if not plan_path.exists():
        raise FileNotFoundError(f"Plan file not found: {plan_path}")

    full_df = pd.read_csv(plan_path)

    print("Plan:", plan_path)
    print("Total rows in plan:", len(full_df))

    if args.limit is not None:
        work_df = full_df.head(args.limit)
        print("Rows selected for this run:", len(work_df))
    else:
        work_df = full_df
        print("Rows selected for this run: all")

    for idx, row in work_df.iterrows():
        model = row["model"]
        text = row["target_text"]
        ref_audio = str(row["reference_audio"]).strip()
        output_audio = str(row["output_audio"]).strip()

        Path(output_audio).parent.mkdir(parents=True, exist_ok=True)

        if not Path(ref_audio).exists():
            print(f"Reference audio not found: {ref_audio}")
            full_df.at[idx, "generation_status"] = "FAILED_REFERENCE_NOT_FOUND"
            continue

        if Path(output_audio).exists():
            print(f"Skipping existing: {output_audio}")
            full_df.at[idx, "generation_status"] = "EXISTS"
            continue

        if model == "Chatterbox":
            cmd = [
                "python",
                "generate_chatterbox_one.py",
                "--text",
                text,
                "--reference_audio",
                ref_audio,
                "--output_audio",
                output_audio,
            ]

        elif model == "Fish-Speech":
            cmd = [
                "python",
                "generate_fishspeech_one.py",
                "--text",
                text,
                "--reference_audio",
                ref_audio,
                "--reference_text",
                "",
                "--output_audio",
                output_audio,
                "--url",
                args.fish_url,
            ]

        else:
            raise ValueError(f"Unknown model: {model}")

        ok = run_command(cmd)
        full_df.at[idx, "generation_status"] = "DONE" if ok else "FAILED"

        full_df.to_csv(plan_path, index=False, encoding="utf-8-sig")

    print("\nUpdated generation plan:")
    print(plan_path)
    print("Rows still preserved:", len(full_df))


if __name__ == "__main__":
    main()