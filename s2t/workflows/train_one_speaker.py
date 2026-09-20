#!/usr/bin/env python3
"""End-to-end Whisper LoRA pipeline for one L2-ARCTIC speaker."""

import argparse
from pathlib import Path
import sys
import time

from s2t.common import PROJECT_ROOT, normalize_speaker


from s2t.workflows.runner import run_cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare, fine-tune, merge, and evaluate Whisper for one Vietnamese speaker"
    )
    parser.add_argument(
        "--speaker",
        type=str.upper,
        required=True,
        help="Vietnamese L2-ARCTIC speaker: HQTV, PNV, THV, or TLV",
    )
    parser.add_argument("--batch_size", type=int, default=28)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--force-data", action="store_true", help="Rebuild resampled audio")
    parser.add_argument("--restart", action="store_true", help="Train from scratch instead of resuming")
    parser.add_argument("--skip-existing-model", action="store_true", help="Reuse an existing merged model")
    args = parser.parse_args()

    try:
        speaker = normalize_speaker(args.speaker, accent="vietnamese")
    except ValueError as exc:
        parser.error(str(exc))

    slug = speaker.lower()
    python = sys.executable
    run_dir = Path("runs") / f"vietnamese-{slug}-lora"
    adapter_dir = run_dir / "best_adapter"
    model_dir = Path("models") / f"whisper-medium-en-vi-{slug}-personalized"
    model_weights = PROJECT_ROOT / model_dir / "model.safetensors"
    tag = f"finetuned_medium_en_vi_{slug}_personalized"

    prepare_command = [python, "-m", "s2t.datasets.prepare_data", "--accent", "vietnamese", "--speaker", speaker]
    if args.force_data:
        prepare_command.append("--force")
    run_cmd(prepare_command, f"1/4 Prepare data for Vietnamese speaker {speaker}")

    if args.skip_existing_model and model_weights.exists():
        print(f"Reusing existing model: {model_dir}")
    else:
        train_command = [
            python,
            "-m", "s2t.training.train_lora",
            "--accent",
            "vietnamese",
            "--speaker",
            speaker,
            "--output",
            str(run_dir),
            "--batch_size",
            str(args.batch_size),
            "--grad_accum",
            str(args.grad_accum),
            "--epochs",
            str(args.epochs),
            "--lr",
            str(args.lr),
        ]
        if not args.restart:
            train_command.append("--resume")
        run_cmd(train_command, f"2/4 Fine-tune LoRA for {speaker}")

        run_cmd(
            [
                python,
                "-m", "s2t.training.merge_lora",
                "--adapter",
                str(adapter_dir),
                "--out",
                str(model_dir),
                "--speaker",
                speaker,
            ],
            f"3/4 Merge the {speaker} adapter into a standalone model",
        )

    run_cmd(
        [
            python,
            "-m", "s2t.evaluation.evaluate_wer",
            "--model",
            str(model_dir),
            "--accent",
            "vietnamese",
            "--speaker",
            speaker,
            "--splits",
            "test,suitcase",
            "--batch_size",
            "16",
            "--tag",
            tag,
        ],
        f"4/4 Evaluate the personalized model for {speaker}",
    )

    print(f"\nPersonalized model: {model_dir}")
    print(f"Evaluation JSON: reports/{tag}.json")


if __name__ == "__main__":
    main()
