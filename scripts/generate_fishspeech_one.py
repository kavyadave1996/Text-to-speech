import argparse
import msgpack
import requests
from pathlib import Path

URL = "http://127.0.0.1:8080/v1/tts"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--output_audio", required=True)
    parser.add_argument("--reference_audio", default=None)
    parser.add_argument("--reference_text", default="")
    parser.add_argument("--url", default=URL)
    args = parser.parse_args()

    output_audio = Path(args.output_audio)
    output_audio.parent.mkdir(parents=True, exist_ok=True)

    references = []

    if args.reference_audio:
        reference_audio = Path(args.reference_audio)

        if not reference_audio.exists():
            raise FileNotFoundError(f"Reference audio not found: {reference_audio}")

        references.append(
            {
                "audio": reference_audio.read_bytes(),
                "text": args.reference_text,
            }
        )

    data = {
        "text": args.text,
        "chunk_length": 200,
        "format": "wav",
        "latency": "normal",
        "references": references,
        "reference_id": None,
        "seed": None,
        "use_memory_cache": "off",
        "normalize": True,
        "streaming": False,
        "max_new_tokens": 512,
        "top_p": 0.9,
        "repetition_penalty": 1.05,
        "temperature": 0.85,
    }

    packed = msgpack.packb(data, use_bin_type=True)

    print(f"Sending request to Fish-Speech API: {args.url}")
    print(f"Output: {output_audio}")

    response = requests.post(
        args.url,
        data=packed,
        headers={
            "Content-Type": "application/msgpack",
            "Accept": "audio/wav",
        },
        timeout=None,
    )

    print("Status:", response.status_code)

    if response.status_code == 200:
        output_audio.write_bytes(response.content)
        print(f"Saved: {output_audio}")
    else:
        print(response.text)
        raise RuntimeError("Fish-Speech generation failed")


if __name__ == "__main__":
    main()