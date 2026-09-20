#!/usr/bin/env python3
"""Freeze the completed personal recorder inventory into training splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf

from s2t.common import PROJECT_ROOT


REQUIRED_COLUMNS = {
    "utt_id",
    "speaker",
    "sentence_id",
    "path",
    "text",
    "duration_s",
    "take_id",
    "review_status",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_audio(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and freeze personal recorder data for fine-tuning"
    )
    parser.add_argument(
        "--source",
        default="recordings/personal_arctic/manifest.csv",
        help="Recorder capture manifest",
    )
    parser.add_argument(
        "--split-ids",
        default="data/split_ids.json",
        help="Frozen sentence split assignment",
    )
    parser.add_argument(
        "--output",
        default="data/personal/manifest.csv",
        help="Derived training manifest",
    )
    parser.add_argument(
        "--accept-user-verification",
        action="store_true",
        help="Record the user's explicit verification attestation in provenance",
    )
    args = parser.parse_args()

    if not args.accept_user_verification:
        parser.error(
            "Personal captures remain marked unreviewed by the recorder. "
            "Pass --accept-user-verification only after the user explicitly confirms review."
        )

    source = (PROJECT_ROOT / args.source).resolve()
    split_path = (PROJECT_ROOT / args.split_ids).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    provenance_path = output.with_name("dataset_provenance.json")

    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    missing_columns = REQUIRED_COLUMNS - set(fieldnames)
    if missing_columns:
        raise ValueError(f"Source manifest is missing columns: {sorted(missing_columns)}")
    if len(rows) != 1132:
        raise ValueError(f"Expected all 1132 prompts, found {len(rows)}")

    sentence_ids = [row["sentence_id"] for row in rows]
    duplicate_ids = [sid for sid, count in Counter(sentence_ids).items() if count != 1]
    if duplicate_ids:
        raise ValueError(f"Expected one kept take per sentence; duplicates: {duplicate_ids[:10]}")

    splits = json.loads(split_path.read_text(encoding="utf-8"))
    split_for_sentence: dict[str, str] = {}
    for split in ("train", "val", "test"):
        for sentence_id in splits[split]:
            if sentence_id in split_for_sentence:
                raise ValueError(f"Sentence appears in multiple splits: {sentence_id}")
            split_for_sentence[sentence_id] = split

    missing_split = sorted(set(sentence_ids) - set(split_for_sentence))
    extra_split = sorted(set(split_for_sentence) - set(sentence_ids))
    if missing_split or extra_split:
        raise ValueError(
            f"Split/catalog mismatch: {len(missing_split)} missing and {len(extra_split)} extra IDs"
        )

    total_duration = 0.0
    for index, row in enumerate(rows, start=1):
        audio_path = resolve_audio(row["path"])
        if not audio_path.is_file():
            raise FileNotFoundError(f"Missing audio: {audio_path}")
        info = sf.info(str(audio_path))
        if info.samplerate != 16000 or info.channels != 1:
            raise ValueError(
                f"Expected mono 16 kHz audio, got {info.channels}ch/{info.samplerate}Hz: {audio_path}"
            )
        if info.frames <= 0:
            raise ValueError(f"Empty audio file: {audio_path}")
        manifest_duration = float(row["duration_s"])
        actual_duration = info.frames / info.samplerate
        if abs(manifest_duration - actual_duration) > 0.02:
            raise ValueError(
                f"Duration mismatch for {row['take_id']}: manifest={manifest_duration}, actual={actual_duration}"
            )
        total_duration += actual_duration
        row["split"] = split_for_sentence[row["sentence_id"]]
        row["capture_review_status"] = row.get("review_status", "unreviewed")
        row["review_status"] = "user_verified"

    output.parent.mkdir(parents=True, exist_ok=True)
    output_fields = fieldnames + ["capture_review_status"]
    tmp_output = output.with_suffix(".csv.tmp")
    with tmp_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp_output.replace(output)

    split_counts = Counter(row["split"] for row in rows)
    split_hours = {
        split: round(
            sum(float(row["duration_s"]) for row in rows if row["split"] == split) / 3600,
            6,
        )
        for split in ("train", "val", "test")
    }
    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(source.relative_to(PROJECT_ROOT)),
        "source_manifest_sha256": sha256_file(source),
        "split_ids": str(split_path.relative_to(PROJECT_ROOT)),
        "split_ids_sha256": sha256_file(split_path),
        "output_manifest": str(output.relative_to(PROJECT_ROOT)),
        "output_manifest_sha256": sha256_file(output),
        "speaker": "PERSONAL",
        "user_verification_attested": True,
        "source_review_status": dict(Counter(r["capture_review_status"] for r in rows)),
        "derived_review_status": dict(Counter(r["review_status"] for r in rows)),
        "rows": len(rows),
        "split_counts": dict(split_counts),
        "split_hours": split_hours,
        "total_hours": round(total_duration / 3600, 6),
    }
    tmp_provenance = provenance_path.with_suffix(".json.tmp")
    tmp_provenance.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    tmp_provenance.replace(provenance_path)

    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
