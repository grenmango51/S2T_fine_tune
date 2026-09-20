#!/usr/bin/env python3
"""
Step 3.2: Run out-of-domain evaluation on LibriSpeech test-clean
for zero-shot baseline and all 6 fine-tuned CMU accent models.
Then run paired bootstrap comparisons against zero-shot baseline.
"""
import os
import sys
import time
from pathlib import Path

from s2t.paths import PROJECT_ROOT
PYTHON = sys.executable or str(PROJECT_ROOT / ".venv_linux" / "bin" / "python3")
REPORTS_DIR = PROJECT_ROOT / "reports"

CMU_ACCENTS = ["cmu_us", "cmu_indian", "cmu_canadian", "cmu_german", "cmu_israeli", "cmu_scottish"]
BASELINE_TAG = "ood_librispeech_zeroshot_medium_en"


from s2t.workflows.runner import run_cmd


def main():
    total_t0 = time.time()

    # 1. Zero-shot baseline
    baseline_json = REPORTS_DIR / f"{BASELINE_TAG}.json"
    if baseline_json.exists():
        print(f"Baseline report already exists at {baseline_json}. Re-evaluating to ensure fresh splitmerge metrics.")

    run_cmd(
        [
            PYTHON, "-u", "-m", "s2t.evaluation.evaluate_wer",
            "--model", "openai/whisper-medium.en",
            "--accent", "librispeech",
            "--splits", "test",
            "--batch_size", "16",
            "--tag", BASELINE_TAG,
        ],
        "Zero-shot medium.en on LibriSpeech test-clean"
    )

    # 2. 6 Fine-tuned CMU models
    for acc in CMU_ACCENTS:
        model_dir = PROJECT_ROOT / "models" / f"whisper-medium-en-{acc}-accent"
        tag = f"ood_librispeech_{acc}"
        run_cmd(
            [
                PYTHON, "-u", "-m", "s2t.evaluation.evaluate_wer",
                "--model", str(model_dir),
                "--model_accent", acc,
                "--accent", "librispeech",
                "--splits", "test",
                "--batch_size", "16",
                "--tag", tag,
            ],
            f"Fine-tuned {acc} on LibriSpeech test-clean"
        )

    # 3. Paired comparisons
    print("\n" + "#"*70)
    print("# RUNNING PAIRED BOOTSTRAP COMPARISONS")
    print("#"*70 + "\n", flush=True)

    for acc in CMU_ACCENTS:
        tag = f"ood_librispeech_{acc}"
        run_cmd(
            [
                PYTHON, "-u", "-m", "s2t.evaluation.compare_paired",
                "--accent", "librispeech",
                "--baseline", BASELINE_TAG,
                "--candidate", tag,
                "--split", "test",
            ],
            f"Paired comparison: {tag} vs {BASELINE_TAG}"
        )

    total_time = time.time() - total_t0
    print(f"\n{'='*70}")
    print(f"ALL LIBRISPEECH EVALUATIONS COMPLETED IN {total_time/60:.1f} MINUTES ({total_time/3600:.2f} HOURS)!")
    print(f"{'='*70}\n", flush=True)


if __name__ == "__main__":
    main()
