import pandas as pd
from pathlib import Path

INPUT = Path("workspace/docs/metadata/noisy_reference_inventory.csv")
OUTPUT = Path("workspace/docs/metadata/noisy_threshold_test_results.csv")

df = pd.read_csv(INPUT)

snr_thresholds = [15, 20, 25, 30, 35, 40]
noise_thresholds = [-30, -35, -40, -45, -50, -55, -60]

rows = []

for snr_t in snr_thresholds:
    for noise_t in noise_thresholds:
        predicted_noisy = (
            (df["reference_snr_db"] <= snr_t)
            | (df["reference_noise_dbfs"] >= noise_t)
        )

        total_noisy = len(df)
        noisy_detected = int(predicted_noisy.sum())
        noisy_missed = total_noisy - noisy_detected
        recall = noisy_detected / total_noisy

        rows.append({
            "snr_threshold": snr_t,
            "noise_dbfs_threshold": noise_t,
            "rule": f"SNR <= {snr_t} OR noise_dBFS >= {noise_t}",
            "total_noisy_files": total_noisy,
            "noisy_detected": noisy_detected,
            "noisy_missed": noisy_missed,
            "noisy_recall": round(recall, 4),
        })

result = pd.DataFrame(rows)

result = result.sort_values(
    by=["noisy_recall", "snr_threshold", "noise_dbfs_threshold"],
    ascending=[False, True, False]
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT, index=False)

print(result.to_string(index=False))
print(f"\nSaved: {OUTPUT}")