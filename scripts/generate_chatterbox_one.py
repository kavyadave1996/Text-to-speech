import argparse
from pathlib import Path

import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--reference_audio", required=True)
    parser.add_argument("--output_audio", required=True)
    parser.add_argument("--exaggeration", type=float, default=0.7)
    parser.add_argument("--cfg_weight", type=float, default=0.3)
    args = parser.parse_args()

    reference_audio = Path(args.reference_audio)
    output_audio = Path(args.output_audio)

    if not reference_audio.exists():
        raise FileNotFoundError(f"Reference audio not found: {reference_audio}")

    output_audio.parent.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")
    print(f"Reference: {reference_audio}")
    print(f"Output: {output_audio}")

    model = ChatterboxTTS.from_pretrained(device=device)

    wav = model.generate(
        args.text,
        audio_prompt_path=str(reference_audio),
        exaggeration=args.exaggeration,
        cfg_weight=args.cfg_weight,
    )

    ta.save(str(output_audio), wav, model.sr)

    print(f"Saved: {output_audio}")


if __name__ == "__main__":
    main()