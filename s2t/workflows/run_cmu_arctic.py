#!/usr/bin/env python3
"""Durable, resumable CMU-only queue. One GPU training process at a time."""
import fcntl
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from s2t.datasets.cmu_arctic import CMU_ACCENT_SPEAKERS, ROOT

STATUS = ROOT / "runs" / "cmu_arctic_queue" / "status.json"


def save_status(state):
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    temp = STATUS.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2) + "\n")
    temp.replace(STATUS)


def notify(message):
    # A local desktop notification; no external account or messaging service.
    try:
        result = subprocess.run(["notify-send", "CMU ARCTIC fine-tuning", message],
                                timeout=10, capture_output=True, text=True)
        return {"returncode": result.returncode, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def validate_manifest(accent):
    import pandas as pd
    df = pd.read_csv(ROOT / "data" / accent / "manifest.csv")
    expected = set(CMU_ACCENT_SPEAKERS[accent])
    if set(df.speaker) != expected:
        raise ValueError(f"Incomplete speaker coverage in {accent}")
    if df.utt_id.duplicated().any() or df.groupby("sentence_id").split.nunique().max() != 1:
        raise ValueError(f"Duplicate utterances or prompt leakage in {accent}")
    for split in ("train", "val", "test"):
        if set(df.loc[df.split == split, "speaker"]) != expected:
            raise ValueError(f"Missing speakers in {accent}/{split}")
    return df.groupby(["speaker", "split"]).size().unstack().to_dict("index")


def main():
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with (STATUS.parent / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {"pid": os.getpid(), "status": "running", "stage": "download",
                 "completed": [], "failed": {}, "groups": CMU_ACCENT_SPEAKERS}
        save_status(state)
        try:
            subprocess.run([sys.executable, "-m", "s2t.datasets.download_cmu_arctic"], cwd=ROOT, check=True)
            from s2t.datasets.prepare_data import prepare_accent
            from s2t.workflows.train_all_accents import process_accent, compile_master_report
            for accent in CMU_ACCENT_SPEAKERS:
                try:
                    state.update(stage="prepare", accent=accent)
                    save_status(state)
                    manifest = ROOT / "data" / accent / "manifest.csv"
                    if not manifest.exists():
                        prepare_accent(accent)
                    state.setdefault("manifest_counts", {})[accent] = validate_manifest(accent)
                    state["stage"] = "train_merge_evaluate"
                    save_status(state)
                    process_accent(accent, sys.executable, batch_size=28, grad_accum=1, epochs=10)
                    report_path = ROOT / "reports" / f"finetuned_medium_en_{accent}_lora.json"
                    report = json.loads(report_path.read_text())
                    if "wer" not in report.get("splits", {}).get("test", {}):
                        raise ValueError(f"Missing test evaluation: {report_path}")
                    state["completed"].append(accent)
                    save_status(state)
                except Exception:
                    state["failed"][accent] = traceback.format_exc()
                    traceback.print_exc()
                    save_status(state)
            compile_master_report()
            state.update(status="failed" if state["failed"] else "complete", stage="finished")
        except Exception:
            state.update(status="failed", error=traceback.format_exc())
            traceback.print_exc()
        finally:
            save_status(state)
            message = f"{state['status']}: {len(state['completed'])}/6 accent models completed. Details: {STATUS}"
            print(message, flush=True)
            state["notification"] = notify(message)
            save_status(state)
        return 0 if state["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
