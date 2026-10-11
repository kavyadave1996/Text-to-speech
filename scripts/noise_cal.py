import argparse
import math
import warnings
import librosa
import numpy as np
import webrtcvad

# Suppress setuptools pkg_resources UserWarning
warnings.filterwarnings("ignore", category=UserWarning)


def float_to_pcm16(x: np.ndarray) -> bytes:
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767).astype(np.int16).tobytes()


def frame_audio(y: np.ndarray, sr: int, frame_ms: int = 30):
    frame_len = int(sr * frame_ms / 1000)
    num_frames = len(y) // frame_len

    frames = []
    for i in range(num_frames):
        start = i * frame_len
        end = start + frame_len
        frames.append(y[start:end])

    return frames


def power_dbfs(power: float, eps: float = 1e-12) -> float:
    return 10.0 * math.log10(max(power, eps))


def calculate_100_sample_noise(
    y: np.ndarray, num_samples: int = 100, start_idx: int = 0
):
    """Calculates RMS and dBFS for a specific slice of samples (e.g., 100 samples)."""
    sample_slice = y[start_idx : start_idx + num_samples]

    if len(sample_slice) == 0:
        return None, None

    rms = float(np.sqrt(np.mean(sample_slice**2)))
    dbfs = 10.0 * math.log10(max(rms**2, 1e-12))

    return round(rms, 6), round(dbfs, 2)


def estimate_noise_and_snr(
    audio_path: str, vad_mode: int = 2, frame_ms: int = 30
):
    # WebRTC VAD supports 8000, 16000, 32000, 48000 Hz.
    sr = 16000
    y, _ = librosa.load(audio_path, sr=sr, mono=True)

    # Remove DC offset
    y = y - np.mean(y)

    # 1. Calculate noise over the FIRST 100 samples
    first_100_rms, first_100_dbfs = calculate_100_sample_noise(
        y, num_samples=100, start_idx=0
    )

    # 2. Calculate noise over the QUIETEST 100 samples in the entire track
    sample_powers = y**2
    window_sum = np.convolve(
        sample_powers, np.ones(100) / 100, mode="valid"
    )
    quietest_idx = int(np.argmin(window_sum))

    quietest_100_rms, quietest_100_dbfs = calculate_100_sample_noise(
        y, num_samples=100, start_idx=quietest_idx
    )

    # Avoid silent file crash
    if np.max(np.abs(y)) < 1e-6:
        return {
            "audio_path": audio_path,
            "status": "silent_or_invalid",
            "first_100_samples_rms": first_100_rms,
            "first_100_samples_dbfs": first_100_dbfs,
            "quietest_100_samples_dbfs": quietest_100_dbfs,
            "noise_dbfs": None,
            "active_dbfs": None,
            "estimated_snr_db": None,
            "speech_frame_ratio": 0.0,
            "decision": "reject",
        }

    vad = webrtcvad.Vad(vad_mode)
    frames = frame_audio(y, sr, frame_ms=frame_ms)

    speech_powers = []
    noise_powers = []

    for frame in frames:
        pcm = float_to_pcm16(frame)
        is_speech = vad.is_speech(pcm, sr)
        p = float(np.mean(frame**2))

        if is_speech:
            speech_powers.append(p)
        else:
            noise_powers.append(p)

    speech_frame_ratio = len(speech_powers) / max(len(frames), 1)

    all_powers = np.array([float(np.mean(f**2)) for f in frames])
    if len(noise_powers) < 3:
        noise_power = float(np.percentile(all_powers, 10))
    else:
        noise_power = float(np.mean(noise_powers))

    if len(speech_powers) < 3:
        active_power = float(np.percentile(all_powers, 90))
    else:
        active_power = float(np.mean(speech_powers))

    speech_power = max(active_power - noise_power, 1e-12)

    noise_dbfs = power_dbfs(noise_power)
    active_dbfs = power_dbfs(active_power)
    estimated_snr_db = 10.0 * math.log10(speech_power / max(noise_power, 1e-12))

    if estimated_snr_db >= 20 and noise_dbfs <= -35:
        decision = "good_reference_for_fishspeech_and_chatterbox"
    elif estimated_snr_db >= 15:
        decision = "usable_but_inspect_output"
    elif estimated_snr_db >= 10:
        decision = "borderline_chatterbox_trial_only"
    else:
        decision = "reject_as_tts_reference_use_only_for_vhf_analysis"

    return {
        "audio_path": audio_path,
        "status": "ok",
        "first_100_samples_rms": first_100_rms,
        "first_100_samples_dbfs": first_100_dbfs,
        "quietest_100_samples_dbfs": quietest_100_dbfs,
        "noise_dbfs": round(noise_dbfs, 2),
        "active_dbfs": round(active_dbfs, 2),
        "estimated_snr_db": round(estimated_snr_db, 2),
        "speech_frame_ratio": round(speech_frame_ratio, 3),
        "decision": decision,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path")
    parser.add_argument("--vad_mode", type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument("--frame_ms", type=int, default=30, choices=[10, 20, 30])
    args = parser.parse_args()

    result = estimate_noise_and_snr(
        args.audio_path, vad_mode=args.vad_mode, frame_ms=args.frame_ms
    )

    for k, v in result.items():
        print(f"{k}: {v}")