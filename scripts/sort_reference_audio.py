import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
from tqdm import tqdm


def process_and_sort_audio(input_folder: str, output_base_folder: str):
  # 1. Create target directories for Node E and Node F
  base_path = Path(output_base_folder)
  node_e_dir = base_path / "Node_E_Cleaned_Reference"
  node_f_dir = base_path / "Node_F_Noisy_Reference"

  node_e_dir.mkdir(parents=True, exist_ok=True)
  node_f_dir.mkdir(parents=True, exist_ok=True)

  # 2. Find all audio files
  audio_files = [
      p for p in Path(input_folder).rglob("*") if p.suffix.lower() == ".wav"
  ]
  print(f"Found {len(audio_files)} files to classify...\n")

  results = []

  for file_path in tqdm(audio_files, desc="Classifying and Sorting"):
    try:
      # Load audio waveform
      sr, data = wav.read(str(file_path))
      if data.ndim > 1:
        data = data.mean(axis=1)  # Convert stereo to mono
      data = data.astype(np.float32)
      if np.max(np.abs(data)) > 0:
        data = data / np.max(np.abs(data))

      # --- A. Measure High Frequency Spectral Ratio ---
      fft_vals = np.abs(np.fft.rfft(data))
      fft_freqs = np.fft.rfftfreq(len(data), 1.0 / sr)

      energy_low = np.sum(fft_vals[fft_freqs <= 4000] ** 2)
      energy_high = np.sum(
          fft_vals[(fft_freqs > 4000) & (fft_freqs <= 8000)] ** 2
      )
      high_freq_ratio = float(energy_high / (energy_low + 1e-10))

      # --- B. Measure SNR (Speech vs Pause Floor) ---
      frame_len = int(0.030 * sr)  # 30ms window
      hop_len = int(0.010 * sr)  # 10ms hop
      num_frames = (len(data) - frame_len) // hop_len + 1
      rms = np.array([
          np.sqrt(np.mean(data[i * hop_len : i * hop_len + frame_len] ** 2))
          for i in range(num_frames)
      ])

      noise_thresh = np.percentile(rms, 15)
      speech_frames = rms[rms > (noise_thresh * 2.5)]
      noise_frames = rms[rms <= noise_thresh]

      if len(speech_frames) > 0 and len(noise_frames) > 0:
        speech_power = np.mean(speech_frames**2)
        noise_power = np.mean(noise_frames**2) + 1e-10
        snr_db = float(10.0 * np.log10(speech_power / noise_power))
      else:
        snr_db = 0.0

      # --- C. Decision Rule ---
      is_wideband = high_freq_ratio >= 0.005
      is_clean_pause = snr_db >= 22.0

      if is_wideband and is_clean_pause:
        verdict = "Node_E_Clean"
        destination_folder = node_e_dir
        reason = "Wideband voice, studio/mic clarity, quiet background"
      else:
        verdict = "Node_F_Noisy"
        destination_folder = node_f_dir
        if not is_wideband:
          reason = "VHF bandpass cutoff (<3.4 kHz) already present"
        else:
          reason = "Low SNR / Background channel static"

      # Copy file to destination folder
      shutil.copy(str(file_path), destination_folder / file_path.name)

      results.append({
          "filename": file_path.name,
          "verdict": verdict,
          "high_freq_ratio": round(high_freq_ratio, 6),
          "snr_db": round(snr_db, 2),
          "reason": reason,
      })

    except Exception as e:
      print(f"\nError processing {file_path.name}: {e}")

  # 3. Export Summary CSV
  df_summary = pd.DataFrame(results)
  summary_csv_path = base_path / "classification_summary.csv"
  df_summary.to_csv(summary_csv_path, index=False)

  print("\n Classification complete!")
  print(f" -> Node E (Clean) files saved to: {node_e_dir}")
  print(f" -> Node F (Noisy) files saved to: {node_f_dir}")
  print(f" -> Summary CSV saved to: {summary_csv_path}")


if __name__ == "__main__":
  # Change these paths to your project locations
  INPUT_AUDIO_DIR = (
      r"C:\Users\kda\Documents\ma-tts-tuning\workspace\docs\clean_dataset\reference_voices"
  )
  OUTPUT_SORTED_DIR = (
      r"C:\Users\kda\Documents\ma-tts-tuning\workspace\docs\sorted_pipeline_data"
  )

  process_and_sort_audio(INPUT_AUDIO_DIR, OUTPUT_SORTED_DIR)