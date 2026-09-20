#!/usr/bin/env python3
"""
Step 1.4: Backfill existing report JSONs with split/merge-tolerant WER metrics
without re-decoding. Reads ref_raw/hyp_raw from each sample, computes the new
fields, writes back in place, then regenerates all markdown reports.
"""
import json
import sys
from pathlib import Path

# Project root is the directory containing this script
from s2t.paths import PROJECT_ROOT

from s2t.common import (
    ACCENT_SPEAKERS,
    _splitmerge_edits,
    normalize_text,
)
from s2t.datasets.cmu_arctic import CMU_ACCENT_SPEAKERS


def backfill_report(json_path: Path) -> bool:
    """Add split/merge metrics to a report JSON. Returns True if modified."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or "splits" not in data:
        return False

    modified = False
    for split_name, split_data in data["splits"].items():
        samples = split_data.get("samples", [])
        if not samples:
            continue

        # Backfill per-sample fields
        for sample in samples:
            if "errors_splitmerge" in sample and "n_ref_words" in sample and "errors_std" in sample:
                continue  # Already has the fields

            ref_norm = sample.get("ref_norm", "")
            hyp_norm = sample.get("hyp_norm", "")
            # Use the normalized versions, applying same empty check as evaluate_wer
            ref_eval = ref_norm if ref_norm.strip() else "empty"
            hyp_eval = hyp_norm if hyp_norm.strip() else "empty"

            std_e, tol_e, n_ref = _splitmerge_edits(ref_eval, hyp_eval)
            sample["n_ref_words"] = n_ref
            sample["errors_std"] = std_e
            sample["errors_splitmerge"] = tol_e
            sample["sample_wer_splitmerge"] = tol_e / max(n_ref, 1)
            modified = True

        # Recompute split-level wer_splitmerge from per-sample data
        total_err_sm = sum(s.get("errors_splitmerge", 0) for s in samples)
        total_ref = sum(s.get("n_ref_words", 0) for s in samples)
        split_wer_sm = round((total_err_sm / max(total_ref, 1)) * 100, 2)
        if split_data.get("wer_splitmerge") != split_wer_sm:
            split_data["wer_splitmerge"] = split_wer_sm
            modified = True

        # Recompute per-speaker wer_splitmerge
        speaker_results = split_data.get("speaker_results", {})
        for spk, spk_data in speaker_results.items():
            spk_samples = [s for s in samples if s.get("speaker") == spk]
            if not spk_samples:
                continue
            spk_err_sm = sum(s.get("errors_splitmerge", 0) for s in spk_samples)
            spk_ref = sum(s.get("n_ref_words", 0) for s in spk_samples)
            spk_wer_sm = round((spk_err_sm / max(spk_ref, 1)) * 100, 2)
            if spk_data.get("wer_splitmerge") != spk_wer_sm:
                spk_data["wer_splitmerge"] = spk_wer_sm
                modified = True

    if modified:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"  Updated: {json_path.name}")
    else:
        print(f"  Already up to date: {json_path.name}")

    return modified


def main():
    reports_dir = PROJECT_ROOT / "reports"

    # 1. Backfill all report JSONs
    print("=" * 60)
    print("Step 1: Backfilling report JSONs with split/merge metrics")
    print("=" * 60)

    json_files = sorted(reports_dir.glob("*.json"))
    updated_count = 0
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("splits"), dict):
                if backfill_report(jf):
                    updated_count += 1
        except Exception as e:
            print(f"  Warning: could not process {jf.name}: {e}")

    print(f"\nBackfilled {updated_count} of {len(json_files)} JSON files.\n")

    # 2. Regenerate markdown reports for each CMU accent
    print("=" * 60)
    print("Step 2: Regenerating accent eval reports")
    print("=" * 60)

    # Import here to avoid circular import issues at module level
    from s2t.evaluation.evaluate_wer import generate_eval_report

    for accent in CMU_ACCENT_SPEAKERS:
        print(f"\n  Regenerating report for {accent}...")
        all_reports = []
        for jf in sorted(reports_dir.glob("*.json")):
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict) and isinstance(loaded.get("splits"), dict):
                    all_reports.append(loaded)
            except Exception:
                pass
        try:
            generate_eval_report(all_reports, accent=accent)
        except Exception as e:
            print(f"  Warning: could not generate report for {accent}: {e}")

    # 3. Regenerate master report
    print("\n" + "=" * 60)
    print("Step 3: Regenerating master report")
    print("=" * 60)

    from s2t.workflows.train_all_accents import compile_master_report
    try:
        compile_master_report()
    except Exception as e:
        print(f"  Warning: could not compile master report: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
