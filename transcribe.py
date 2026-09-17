import argparse
import sys
from pathlib import Path
import torch
from transformers import pipeline

from common import PROJECT_ROOT, load_audio_16k


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio using fine-tuned Whisper model for Vietnamese-accented English"
    )
    parser.add_argument("audio_path", type=str, help="Path to input audio file")
    parser.add_argument(
        "--model",
        type=str,
        default="models/whisper-medium-en-vi-accent",
        help="Path to model directory or Hugging Face Hub ID",
    )
    parser.add_argument(
        "--chunk_length_s",
        type=int,
        default=30,
        help="Chunk length in seconds for long audio transcription",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for pipeline chunk processing",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference (cuda or cpu)",
    )
    args = parser.parse_args()

    audio_file = Path(args.audio_path)
    if not audio_file.is_absolute():
        audio_file = (PROJECT_ROOT / audio_file).resolve()

    if not audio_file.exists():
        print(f"Error: audio file not found: {audio_file}", file=sys.stderr)
        sys.exit(1)

    # Load audio via soundfile and soxr (no FFmpeg needed)
    audio = load_audio_16k(audio_file)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=args.model,
        dtype=torch.float16 if args.device == "cuda" else torch.float32,
        device=args.device,
        chunk_length_s=args.chunk_length_s,
        batch_size=args.batch_size,
    )

    result = pipe(audio)
    transcript = result.get("text", "").strip()
    print(transcript)


if __name__ == "__main__":
    main()
