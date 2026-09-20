#!/usr/bin/env python3
"""End-to-end personal Whisper fine-tuning and benchmark schedule."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

from s2t.common import PROJECT_ROOT


from s2t.workflows.runner import run_cmd as run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run personal Whisper fine-tuning schedule")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=28)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--skip-openrouter", action="store_true")
    args = parser.parse_args()

    python = sys.executable
    run_dir = Path("runs/personal-lora")
    adapter_dir = run_dir / "best_adapter"
    model_dir = Path("models/whisper-medium-en-personal")

    run(
        [
            python,
            "-m", "s2t.datasets.prepare_personal_data",
            "--accept-user-verification",
        ],
        "1/6 Validate and freeze the verified personal dataset",
    )
    run(
        [
            python,
            "-m", "s2t.evaluation.evaluate_wer",
            "--model", "openai/whisper-medium.en",
            "--accent", "personal",
            "--splits", "test",
            "--batch_size", "16",
            "--tag", "zeroshot_medium_en_personal",
        ],
        "2/6 Benchmark the zero-shot Whisper medium.en baseline",
    )

    train_command = [
        python,
        "-m", "s2t.training.train_lora",
        "--accent", "personal",
        "--output", str(run_dir),
        "--batch_size", str(args.batch_size),
        "--grad_accum", str(args.grad_accum),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
    ]
    if not args.restart:
        train_command.append("--resume")
    run(train_command, "3/6 Fine-tune the personal LoRA adapter")

    run(
        [
            python,
            "-m", "s2t.training.merge_lora",
            "--adapter", str(adapter_dir),
            "--out", str(model_dir),
        ],
        "4/6 Merge and smoke-test the standalone personal model",
    )
    run(
        [
            python,
            "-m", "s2t.evaluation.evaluate_wer",
            "--model", str(model_dir),
            "--accent", "personal",
            "--splits", "test",
            "--batch_size", "16",
            "--tag", "finetuned_medium_en_personal",
        ],
        "5/6 Evaluate the personal model on the held-out test split",
    )

    if args.skip_openrouter:
        print("6/6 OpenRouter benchmark skipped by request.", flush=True)
    else:
        run(
            [
                python,
                "-m", "s2t.evaluation.benchmark_openrouter_stt",
                "--manifest", "data/personal/manifest.csv",
                "--split", "test",
                "--models", "microsoft/mai-transcribe-2",
                "--local-report", "reports/finetuned_medium_en_personal.json",
            ],
            "6/6 Benchmark Microsoft MAI-Transcribe 2 through OpenRouter",
        )

    print("\nPersonal fine-tuning schedule completed successfully.", flush=True)


if __name__ == "__main__":
    main()
