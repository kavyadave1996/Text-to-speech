# Reference Audio Partitioning Pipeline for Maritime VHF TTS Adaptation

## 1. Overview

In maritime text-to-speech (TTS) synthesis and low-rank adaptation (LoRA) fine-tuning pipelines, reference voice prompt selection directly governs synthetic speech naturalness, channel fidelity, and downstream model convergence.

This methodology partitions an arbitrary real-world maritime reference-audio pool into two controlled acoustic subsets:

### Node E — Clean Reference Audio

Full-bandwidth (20 Hz–20 kHz) studio/direct-microphone speech without transmission-channel constraints.

These audio-text pairs are intended as wideband inputs for TTS models such as Chatterbox TTS and Fish-Speech TTS before downstream digital signal processing (DSP) simulation of VHF channel acoustics. The related project-level pipeline diagram is [main_pipeline.png](main_pipeline.png).

### Node F — Noisy / Native VHF Reference Audio

Real-world maritime emergency audio that may already contain:

- native narrowband filtering (approximately 300 Hz–3.4 kHz),
- carrier/static hash,
- additive engine noise, or
- low signal-to-noise ratio (SNR).

These recordings bypass synthetic DSP filtering to avoid double-filtering artifacts.

---

## 2. Classification Pipeline

```text
                    START
                      │
                      ▼
              ┌───────────────┐
              │   Input Audio │
              │      x[n]     │
              └───────┬───────┘
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
   ┌───────────────┐       ┌───────────────┐
   │ Spectral      │       │ Dynamic SNR   │
   │ Analysis      │       │ Analysis      │
   │               │       │               │
   │ Calculate     │       │ Calculate     │
   │ R_high        │       │ SNR_dB        │
   └───────┬───────┘       └───────┬───────┘
           │                       │
           └───────────┬───────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Decision Gate  │
              │                 │
              │ R_high ≥ 0.005? │
              │       AND       │
              │ SNR ≥ 22 dB?    │
              └────────┬────────┘
                       │
                 ┌─────┴─────┐
                 │           │
                YES          NO
                 │           │
                 ▼           ▼
        ┌──────────────┐  ┌──────────────┐
        │    NODE E    │  │    NODE F    │
        │    CLEAN     │  │  NOISY / VHF │
        └──────┬───────┘  └──────┬───────┘
               │                 │
               ▼                 ▼
        ┌──────────────┐  ┌──────────────┐
        │ Apply VHF    │  │ Bypass VHF   │
        │ DSP          │  │ DSP Filter   │
        │ Simulation   │  │              │
        └──────────────┘  └──────────────┘
```

### Decision rule

An audio sample is assigned to **Node E** only when **both** conditions are satisfied:

```text
R_high ≥ 0.005
AND
SNR_dB ≥ 22.0 dB
```

Otherwise, the sample is assigned to **Node F**.

---

## 3. Mathematical Formulation

### 3.1 Spectral Bandwidth Assessment

A Discrete Fourier Transform (DFT) is evaluated over the input sequence `x[n]` of length `N`:

$$
X[k] = \sum_{n=0}^{N-1} x[n] e^{-j\frac{2\pi}{N}kn}
$$

For sampling frequency `f_s`, the frequency represented by spectral bin `k` is:

$$
f_k = \frac{k f_s}{N}
$$

The low-frequency voice-band energy is:

$$
E_{low} = \sum_{f_k \le 4000} |X[k]|^2
$$

The high-frequency energy is:

$$
E_{high} = \sum_{4000 < f_k \le 8000} |X[k]|^2
$$

The high-frequency ratio is:

$$
R_{high} = \frac{E_{high}}{E_{low}+\epsilon}
$$

where:

$$
\epsilon = 10^{-10}
$$

According to the methodology, wideband studio speech is expected to satisfy:

```text
R_high ≥ 0.005
```

while recordings affected by a VHF-style bandwidth limitation are expected to fall below this threshold.

---

### 3.2 Short-Time Energy and Dynamic Noise-Floor SNR

Short-time Root Mean Square (RMS) energy is calculated using:

- Window duration: **30 ms**
- Hop size: **10 ms**

The RMS of frame `m` is:

$$
RMS_m =
\sqrt{
\frac{1}{L}
\sum_{n=0}^{L-1}
x^2[mH+n]
}
$$

where:

$$
L = \lfloor T_w f_s \rfloor
$$

and:

$$
H = \lfloor T_h f_s \rfloor
$$

The dynamic background-noise floor is estimated from the 15th percentile of the RMS distribution:

$$
\theta_{noise}
=
Percentile_{15}
\left(
\{RMS_m\}_{m=1}^{M}
\right)
$$

The speech-frame mask is:

$$
\mathcal{M}_{speech}
=
\{m \mid RMS_m > 2.5\theta_{noise}\}
$$

The noise-frame mask is:

$$
\mathcal{M}_{noise}
=
\{m \mid RMS_m \le \theta_{noise}\}
$$

The resulting dynamic SNR is:

$$
SNR_{dB}
=
10\log_{10}
\left(
\frac{
\frac{1}{|\mathcal{M}_{speech}|}
\sum_{m\in\mathcal{M}_{speech}} RMS_m^2
}{
\frac{1}{|\mathcal{M}_{noise}|}
\sum_{m\in\mathcal{M}_{noise}} RMS_m^2
+\epsilon
}
\right)
$$

The cleanliness threshold used by the classifier is:

```text
SNR_dB ≥ 22.0 dB
```

The use of percentile-based noise estimation follows the minimum-statistics / noise-estimation approach described by Martin (2001) [1] and Hirsch & Ehrlicher (1995) [2].

---

## 4. Final Decision Function

The complete classification rule is:

$$
\text{Node Assignment}
=
\begin{cases}
\textbf{Node E (Clean Reference)}, &
R_{high}\ge0.005
\land
SNR_{dB}\ge22.0\text{ dB}
\\[6pt]
\textbf{Node F (Noisy / VHF Reference)}, &
\text{otherwise}
\end{cases}
$$

### Practical interpretation

| Condition | Result | DSP treatment |
|---|---|---|
| `R_high ≥ 0.005` **and** `SNR ≥ 22 dB` | **Node E — Clean** | Apply simulated VHF DSP downstream |
| `R_high < 0.005` | **Node F — VHF / narrowband** | Bypass VHF filter |
| `R_high ≥ 0.005` but `SNR < 22 dB` | **Node F — Noisy** | Bypass synthetic VHF filtering |

---

### Related project files

The existing SNR/noise-dBFS calculation can be run with [noise_cal.py](tools/noise_cal.py). Threshold experiments and inventory checks are tracked in [test_noisy_thresholds.py](tools/test_noisy_thresholds.py), [noisy_reference_inventory.csv](metadata/noisy_reference_inventory.csv), and [noisy_threshold_test_results.csv](metadata/noisy_threshold_test_results.csv).

---

## 5. Implementation

Save the following proposed implementation as [sort_reference_audio.py](tools/sort_reference_audio.py).

```python
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
from tqdm import tqdm


def process_and_sort_audio(input_folder: str, output_base_folder: str):
    """Classify audio into Node E (clean) and Node F (noisy/VHF)."""

    base_path = Path(output_base_folder)
    node_e_dir = base_path / "Node_E_Cleaned_Reference"
    node_f_dir = base_path / "Node_F_Noisy_Reference"

    node_e_dir.mkdir(parents=True, exist_ok=True)
    node_f_dir.mkdir(parents=True, exist_ok=True)

    audio_files = [
        p
        for p in Path(input_folder).rglob("*")
        if p.suffix.lower() in [".wav", ".flac"]
    ]

    print(f"Found {len(audio_files)} audio file(s) for classification.\n")

    results = []

    for file_path in tqdm(audio_files, desc="Classifying and Sorting"):
        try:
            # 1. Ingestion & preprocessing
            sr, data = wav.read(str(file_path))

            if data.ndim > 1:
                data = data.mean(axis=1)

            data = data.astype(np.float32)

            max_val = np.max(np.abs(data))
            if max_val > 0:
                data = data / max_val

            # 2. Spectral bandwidth assessment
            fft_vals = np.abs(np.fft.rfft(data))
            fft_freqs = np.fft.rfftfreq(len(data), 1.0 / sr)

            energy_low = np.sum(
                fft_vals[fft_freqs <= 4000] ** 2
            )

            energy_high = np.sum(
                fft_vals[
                    (fft_freqs > 4000) &
                    (fft_freqs <= 8000)
                ] ** 2
            )

            high_freq_ratio = float(
                energy_high / (energy_low + 1e-10)
            )

            # 3. Dynamic pause-floor SNR estimation
            frame_len = int(0.030 * sr)
            hop_len = int(0.010 * sr)

            num_frames = (
                (len(data) - frame_len) // hop_len + 1
            )

            rms = np.array([
                np.sqrt(
                    np.mean(
                        data[
                            i * hop_len :
                            i * hop_len + frame_len
                        ] ** 2
                    )
                )
                for i in range(num_frames)
            ])

            noise_thresh = np.percentile(rms, 15)

            speech_frames = rms[
                rms > (noise_thresh * 2.5)
            ]

            noise_frames = rms[
                rms <= noise_thresh
            ]

            if len(speech_frames) > 0 and len(noise_frames) > 0:
                speech_power = np.mean(
                    speech_frames ** 2
                )

                noise_power = (
                    np.mean(noise_frames ** 2)
                    + 1e-10
                )

                snr_db = float(
                    10.0 * np.log10(
                        speech_power / noise_power
                    )
                )
            else:
                snr_db = 0.0

            # 4. Decision gate
            is_wideband = high_freq_ratio >= 0.005
            is_clean_pause = snr_db >= 22.0

            if is_wideband and is_clean_pause:
                verdict = "Node_E_Clean"
                destination_folder = node_e_dir
                reason = (
                    "Wideband voice, studio/mic clarity, "
                    "quiet background"
                )
            else:
                verdict = "Node_F_Noisy"
                destination_folder = node_f_dir

                if not is_wideband:
                    reason = (
                        "VHF bandpass cutoff "
                        "(<3.4 kHz) already present"
                    )
                else:
                    reason = (
                        "Low SNR / Background channel static"
                    )

            # 5. File migration and logging
            shutil.copy(
                str(file_path),
                destination_folder / file_path.name
            )

            results.append({
                "filename": file_path.name,
                "verdict": verdict,
                "high_freq_ratio": round(
                    high_freq_ratio, 6
                ),
                "snr_db": round(snr_db, 2),
                "reason": reason,
            })

        except Exception as e:
            print(
                f"\n[Error processing "
                f"{file_path.name}]: {e}"
            )

    # 6. Summary export
    df_summary = pd.DataFrame(results)

    summary_csv_path = (
        base_path / "classification_summary.csv"
    )

    df_summary.to_csv(
        summary_csv_path,
        index=False
    )

    print(
        f"\nProcessing complete. "
        f"Summary saved to: {summary_csv_path}"
    )


if __name__ == "__main__":
    INPUT_AUDIO_DIR = (
        r"C:\Users\kda\Documents\ma-tts-tuning"
        r"\workspace\docs\reference_voices"
    )

    OUTPUT_SORTED_DIR = (
        r"C:\Users\kda\Documents\ma-tts-tuning"
        r"\workspace\docs\sorted_pipeline_data"
    )

    process_and_sort_audio(
        INPUT_AUDIO_DIR,
        OUTPUT_SORTED_DIR
    )
```

---

## 6. Output Folder Structure

After executing [sort_reference_audio.py](tools/sort_reference_audio.py), the expected structure is:

```text
sorted_pipeline_data/
│
├── Node_E_Cleaned_Reference/
│   ├── clean_audio_01.wav
│   ├── clean_audio_02.wav
│   └── ...
│
├── Node_F_Noisy_Reference/
│   ├── vhf_audio_01.wav
│   ├── noisy_audio_02.wav
│   └── ...
│
└── classification_summary.csv
```

The generated summary file [classification_summary.csv](sorted_pipeline_data/classification_summary.csv) records:

```text
filename
verdict
high_freq_ratio
snr_db
reason
```

---

## Local Project File References

The method and supporting material can be connected to the following project files. These links assume this Markdown file is stored inside [workspace/docs/](./). The same Markdown-link style is also used wherever project scripts or files are mentioned in the text.

| File | Purpose |
|---|---|
| [main_pipeline.png](main_pipeline.png) | Main project pipeline diagram. |
| [README.md](README.md) | Documentation for `workspace/docs/`. |
| [noise_cal.py](tools/noise_cal.py) | Existing project script for SNR/noise-dBFS calculation. |
| [test_noisy_thresholds.py](tools/test_noisy_thresholds.py) | Threshold-testing script for noisy-reference classification. |
| [sort_reference_audio.py](tools/sort_reference_audio.py) | Proposed script for partitioning reference audio into clean and noisy/VHF folders. |
| [classification_summary.csv](sorted_pipeline_data/classification_summary.csv) | Expected output summary from the reference-audio partitioning script. |
| [noisy_reference_inventory.csv](metadata/noisy_reference_inventory.csv) | Inventory of noisy-source reference files and measured acoustic values. |
| [noisy_threshold_test_results.csv](metadata/noisy_threshold_test_results.csv) | Candidate threshold-test results. |
| [VAD-review.pdf](../../lit/VAD-review.pdf) | Literature support for VAD/noise-estimation background. |
| [T-REC-P.80-199303-S__PDF-E.pdf](../../lit/T-REC-P.80-199303-S__PDF-E.pdf) | ITU-T speech-quality/listening-evaluation reference. |
| [1904.02882v1.pdf](../../lit/1904.02882v1.pdf) | TTS corpus/dataset-design background reference. |
| [Thesis_proposal.docx](../../lit/Thesis_proposal.docx) | Thesis proposal document. |
| [report.docx](../../lit/report.docx) | Thesis/report draft document. |
| [lit/README.md](../../lit/README.md) | Literature-folder documentation. |
| [Noise_estimation_techniques_for_robust_speech_recognition.pdf](snr_outputs/Noise_estimation_techniques_for_robust_speech_recognition.pdf) | Local PDF for Hirsch & Ehrlicher (1995), used for noise-estimation support. |
| [Noise_power_spectral_density_estimation_based_on_optimal_smoothing_and_minimum_statistics.pdf](snr_outputs/Noise_power_spectral_density_estimation_based_on_optimal_smoothing_and_minimum_statistics.pdf) | Local PDF for Martin (2001), used for minimum-statistics noise-floor estimation. |
| [On_Reference_Signal_Estimation_from_Noisy_Speech_Using_Deep_Learning_for_Intelligibility_Estimation.pdf](snr_outputs/On_Reference_Signal_Estimation_from_Noisy_Speech_Using_Deep_Learning_for_Intelligibility_Estimation.pdf) | Local PDF for Takahashi & Kondo (2018), used for noisy reference-signal estimation context. |
| [Peeters_2003_cuidadoaudiofeatures.pdf](snr_outputs/Peeters_2003_cuidadoaudiofeatures.pdf) | Local PDF for Peeters (2004), used for audio feature analysis support. |
| [R-REC-M.493-15-201901-S!!PDF-E.pdf](snr_outputs/R-REC-M.493-15-201901-S!!PDF-E.pdf) | Local PDF for ITU-R M.493-15, used for maritime communication context. |
| [T-REC-G.712-199611-S!!PDF-E.pdf](snr_outputs/T-REC-G.712-199611-S!!PDF-E.pdf) | Local PDF for ITU-T G.712, used for transmission-channel/bandwidth context. |

---

## 7. Documentation Links

The surrounding documentation is available in [README.md](README.md), while the literature and thesis documents are available as [VAD-review.pdf](../../lit/VAD-review.pdf), [T-REC-P.80-199303-S__PDF-E.pdf](../../lit/T-REC-P.80-199303-S__PDF-E.pdf), [1904.02882v1.pdf](../../lit/1904.02882v1.pdf), [Thesis_proposal.docx](../../lit/Thesis_proposal.docx), [report.docx](../../lit/report.docx), and [lit/README.md](../../lit/README.md).

---

## 8. References

The citation numbers below are intentionally explicit so that a reader can identify exactly which source `[1]`, `[2]`, etc. refers to.

### [1] Martin (2001)

Martin, Rainer. **"Noise power spectral density estimation based on optimal smoothing and minimum statistics."**

*IEEE Transactions on Speech and Audio Processing*, 9(5), 504–512, 2001.

DOI: `10.1109/89.928915`

Local PDF: [Noise_power_spectral_density_estimation_based_on_optimal_smoothing_and_minimum_statistics.pdf](snr_outputs/Noise_power_spectral_density_estimation_based_on_optimal_smoothing_and_minimum_statistics.pdf)

**Used for:** percentile/minimum-statistics-based background-noise estimation.

### [2] Hirsch & Ehrlicher (1995)

Hirsch, Hans-Günter, and Christoph Ehrlicher. **"Noise estimation techniques for robust speech recognition."**

*1995 International Conference on Acoustics, Speech, and Signal Processing (ICASSP)*, Vol. 1, pp. 153–156, 1995.

DOI: `10.1109/ICASSP.1995.479532`

Local PDF: [Noise_estimation_techniques_for_robust_speech_recognition.pdf](snr_outputs/Noise_estimation_techniques_for_robust_speech_recognition.pdf)

**Used for:** statistical noise estimation and speech/noise discrimination.

### [3] Peeters (2004)

Peeters, Geoffroy. **"A large set of audio features for sound description (similarity and classification) in the CUIDADO project."**

IRCAM, 2004.

Local PDF: [Peeters_2003_cuidadoaudiofeatures.pdf](snr_outputs/Peeters_2003_cuidadoaudiofeatures.pdf)

**Used as:** supporting reference for audio-feature-based analysis.

### [4] Takahashi & Kondo (2018)

Takahashi, Hiroto, and Kazuhiro Kondo. **"On Reference Signal Estimation from Noisy Speech Using Deep Learning for Intelligibility Estimation."**

*2018 IEEE 7th Global Conference on Consumer Electronics (GCCE)*, pp. 347–348, 2018.

DOI: `10.1109/GCCE.2018.8574768`

Local PDF: [On_Reference_Signal_Estimation_from_Noisy_Speech_Using_Deep_Learning_for_Intelligibility_Estimation.pdf](snr_outputs/On_Reference_Signal_Estimation_from_Noisy_Speech_Using_Deep_Learning_for_Intelligibility_Estimation.pdf)

**Used as:** supporting reference for reference-signal estimation from noisy speech.

### [5] ITU-R M.493-15 (2019)

International Telecommunication Union. **ITU-R Recommendation M.493-15: Digital selective-calling system for use in the maritime mobile service.**

Geneva, Switzerland, 2019.

Local PDF: [R-REC-M.493-15-201901-S!!PDF-E.pdf](snr_outputs/R-REC-M.493-15-201901-S!!PDF-E.pdf)

**Used as:** maritime communications context.

### [6] ITU-T G.712 (2001)

International Telecommunication Union. **ITU-T Recommendation G.712: Transmission performance characteristics of pulse code modulation channels.**

Geneva, Switzerland, 2001.

Local PDF: [T-REC-G.712-199611-S!!PDF-E.pdf](snr_outputs/T-REC-G.712-199611-S!!PDF-E.pdf)

**Used as:** transmission-channel context for the discussion of bandwidth limitations.

### [7] Rabiner & Schafer (2010)

Rabiner, Lawrence R., and Ronald W. Schafer. **Theory and Applications of Digital Speech Processing.**

Prentice Hall / Pearson, 2010.

ISBN: `978-0132629621`

**Used as:** general digital speech-processing reference.

---

## 9. Citation Map

For quick reference:

| Citation | Source | Purpose in this document |
|---|---|---|
| **[1]** | Martin (2001) | Minimum-statistics / noise-floor estimation |
| **[2]** | Hirsch & Ehrlicher (1995) | Noise estimation for speech processing |
| **[3]** | Peeters (2004) | Audio feature analysis |
| **[4]** | Takahashi & Kondo (2018) | Noisy reference-signal estimation |
| **[5]** | ITU-R M.493-15 | Maritime communication context |
| **[6]** | ITU-T G.712 | Transmission-channel characteristics |
| **[7]** | Rabiner & Schafer (2010) | Digital speech processing background |

> **Important:** The original document contained citation placeholders such as `[cite: 1]` and `[cite: 4]`. Those placeholders were ambiguous in rendered Markdown. This version replaces them with explicit numbered references and a citation map so a reader can understand exactly what each citation refers to.

---

## 10. Important Implementation Note

The supplied implementation in [sort_reference_audio.py](tools/sort_reference_audio.py) uses `scipy.io.wavfile.read()`, which directly supports WAV files. Although the file-discovery code also searches for `.flac`, the shown implementation does not provide FLAC decoding through `wav.read()`.

If FLAC input is required, the ingestion step should be replaced with an audio reader that explicitly supports both WAV and FLAC (for example, `soundfile`).

This note is included to make the Markdown documentation accurately reflect the behavior of the supplied script.
