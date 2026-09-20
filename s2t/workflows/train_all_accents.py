#!/usr/bin/env python3
"""
Orchestration pipeline to fine-tune, merge, and evaluate Whisper medium (LoRA)
for all L2-ARCTIC accent regions on the RTX 5070 (12GB VRAM).
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import sys
import time

from s2t.common import ACCENT_SPEAKERS, PROJECT_ROOT
from s2t.datasets.cmu_arctic import CMU_ACCENT_SPEAKERS


TARGET_ACCENTS = ["arabic", "chinese", "hindi", "korean", "spanish"]


from s2t.workflows.runner import run_cmd


def process_accent(accent: str, py_bin: str, batch_size: int = 16, grad_accum: int = 1, epochs: int = 10):
    accent = accent.lower()
    print(f"\n{'#'*65}")
    print(f"# PROCESSING ACCENT: {accent.upper()}")
    print(f"{'#'*65}\n")

    model_dir = PROJECT_ROOT / "models" / f"whisper-medium-en-{accent}-accent"
    model_weights = model_dir / "model.safetensors"
    speaker = ACCENT_SPEAKERS[accent][0]

    merge_complete = model_dir / "merge_complete.json"
    if (model_weights.exists() and model_weights.stat().st_size > 500 * 1024 * 1024
            and (accent not in CMU_ACCENT_SPEAKERS or merge_complete.exists())):
        print(f"Model weights already exist at {model_weights} ({model_weights.stat().st_size / 1e9:.2f} GB). Skipping training.")
    else:
        # 1. Data Prep
        manifest_file = PROJECT_ROOT / "data" / accent / "manifest.csv"
        if not manifest_file.exists():
            run_cmd(
                [py_bin, "-m", "s2t.datasets.prepare_data", "--accent", accent],
                f"Data preparation for {accent}",
            )
        else:
            print(f"Data manifest for {accent} already exists at {manifest_file.relative_to(PROJECT_ROOT)}")

        # 2. Fine-tune LoRA
        run_output = f"runs/{accent}-lora"
        run_cmd(
            [
                py_bin,
                "-m", "s2t.training.train_lora",
                "--accent", accent,
                "--output", run_output,
                "--batch_size", str(batch_size),
                "--grad_accum", str(grad_accum),
                "--epochs", str(epochs),
                "--resume",
            ],
            f"LoRA Training for {accent}",
        )

        # 3. Merge weights into standalone model
        adapter_path = f"{run_output}/best_adapter"
        run_cmd(
            [
                py_bin,
                "-m", "s2t.training.merge_lora",
                "--adapter", adapter_path,
                "--out", str(model_dir),
                *( ["--speaker", speaker] if speaker else [] ),
            ],
            f"Merge LoRA into standalone model for {accent}",
        )

        merge_complete.write_text(json.dumps({"accent": accent, "speakers": ACCENT_SPEAKERS[accent]}))

    # 4. Evaluate standalone model
    report_tag = f"finetuned_medium_en_{accent}_lora"
    manifest_file = PROJECT_ROOT / "data" / accent / "manifest.csv"
    available_splits = ["test"]
    if manifest_file.exists():
        import pandas as pd
        split_values = set(pd.read_csv(manifest_file, usecols=["split"])["split"])
        if "suitcase" in split_values:
            available_splits.append("suitcase")

    run_cmd(
        [
            py_bin,
            "-m", "s2t.evaluation.evaluate_wer",
            "--model", str(model_dir),
            "--accent", accent,
            "--splits", ",".join(available_splits),
            "--batch_size", "16",
            "--tag", report_tag,
        ],
        f"Evaluation for {accent}",
    )

    assert model_weights.exists(), f"Expected merged weights at {model_weights}"
    print(f"\nACCENT {accent.upper()} SUCCESSFULLY COMPLETED AND VERIFIED!")


def compile_master_report():
    reports_dir = PROJECT_ROOT / "reports"
    models_dir = PROJECT_ROOT / "models"

    accents_all = [acc for acc in ACCENT_SPEAKERS if acc != "librispeech"]
    summary_rows = []

    for acc in accents_all:
        model_name = f"whisper-medium-en-{acc}-accent"
        model_p = models_dir / model_name
        if not model_p.exists() and acc == "vietnamese":
            model_name = "whisper-medium-en-vi-accent"
            model_p = models_dir / model_name
        weights_p = model_p / "model.safetensors"
        weights_size_gb = (weights_p.stat().st_size / 1e9) if weights_p.exists() else 0.0

        # Try to find corresponding report JSON
        tag = f"finetuned_medium_en_{acc}_lora" if acc != "vietnamese" else "finetuned_medium_en_lora"
        json_file = reports_dir / f"{tag}.json"

        test_wer = "N/A"
        test_wer_sm = "N/A"
        test_cer = "N/A"
        sc_wer = "N/A"

        if json_file.exists():
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                splits = data.get("splits", {})
                if "test" in splits:
                    test_wer = f"{splits['test'].get('wer', 'N/A')}%"
                    wer_sm_val = splits['test'].get('wer_splitmerge', 'N/A')
                    test_wer_sm = f"{wer_sm_val}%" if wer_sm_val != 'N/A' else 'N/A'
                    test_cer = f"{splits['test'].get('cer', 'N/A')}%"
                if "suitcase" in splits:
                    sc_wer = f"{splits['suitcase'].get('wer', 'N/A')}%"
            except Exception:
                pass

        summary_rows.append({
            "accent": acc.capitalize(),
            "model": model_name,
            "weights_size_gb": f"{weights_size_gb:.2f} GB" if weights_size_gb > 0 else "Missing",
            "test_wer": test_wer,
            "test_wer_sm": test_wer_sm,
            "test_cer": test_cer,
            "suitcase_wer": sc_wer,
        })

    md_path = reports_dir / "all_accents_master_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# ARCTIC Multi-Accent Whisper Fine-Tuning Summary Report\n\n")
        f.write(f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Hardware:** NVIDIA GeForce RTX 5070 (12 GB VRAM)  \n")
        f.write(f"**Base Model:** `openai/whisper-medium.en` (LoRA, r=32, alpha=64)  \n\n")
        f.write("## Fine-Tuned Model Weights & Benchmark Results Across All Regions\n\n")
        f.write("| Region / Accent | Standalone Merged Model Directory | Weights Size | Test Set WER | Test WER (split/merge-tol.) | Test Set CER | Suitcase (Spontaneous) WER |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for row in summary_rows:
            f.write(f"| **{row['accent']}** | `{row['model']}` | {row['weights_size_gb']} | **{row['test_wer']}** | {row['test_wer_sm']} | {row['test_cer']} | {row['suitcase_wer']} |\n")
        f.write("\n\nAll models are standalone merged FP16 models compatible with `WhisperForConditionalGeneration.from_pretrained()`.\n")

    print(f"\nCompiled master report at {md_path.relative_to(PROJECT_ROOT)}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune all accent models on RTX 5070")
    parser.add_argument(
        "--accents",
        type=str,
        default=",".join(TARGET_ACCENTS),
        help="Comma-separated list of accents to train",
    )
    parser.add_argument("--batch_size", type=int, default=28, help="Train batch size (~8GB VRAM target)")
    parser.add_argument("--grad_accum", type=int, default=1, help="Grad accum steps")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    args = parser.parse_args()

    py_bin = sys.executable
    accents = [a.strip() for a in args.accents.split(",") if a.strip()]

    start_total = time.time()
    for acc in accents:
        process_accent(
            accent=acc,
            py_bin=py_bin,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            epochs=args.epochs,
        )

    compile_master_report()
    total_time = time.time() - start_total
    print(f"\n{'='*65}")
    print(f" ALL ACCENTS COMPLETED SUCCESSFULLY IN {total_time/3600:.2f} HOURS!")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
