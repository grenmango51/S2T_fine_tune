#!/usr/bin/env python3
"""
Step 3.1: Download LibriSpeech test-clean, convert to 16kHz mono WAV,
and write a manifest CSV compatible with evaluate_wer.py.

Does NOT use the HF `datasets` package (not installed).
"""
import csv
import os
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from s2t.paths import PROJECT_ROOT
DATA_DIR = PROJECT_ROOT / "data" / "librispeech"
RAW_DIR = DATA_DIR / "raw"
WAV_DIR = DATA_DIR / "wav16k"
MANIFEST_PATH = DATA_DIR / "manifest.csv"
SAMPLE_RATE = 16000

DOWNLOAD_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
ARCHIVE_PATH = DATA_DIR / "test-clean.tar.gz"


def download_archive():
    """Download test-clean.tar.gz if not already present."""
    if ARCHIVE_PATH.exists():
        print(f"Archive already exists at {ARCHIVE_PATH} ({ARCHIVE_PATH.stat().st_size / 1e6:.1f} MB)")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DOWNLOAD_URL} ...")
    print(f"  -> {ARCHIVE_PATH}")

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb = downloaded / 1e6
            total_mb = total_size / 1e6
            print(f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct:.0f}%)", end="", flush=True)

    urllib.request.urlretrieve(DOWNLOAD_URL, str(ARCHIVE_PATH), reporthook=reporthook)
    print(f"\n  Downloaded {ARCHIVE_PATH.stat().st_size / 1e6:.1f} MB")


def extract_archive():
    """Extract tar.gz to RAW_DIR."""
    marker = RAW_DIR / ".extracted"
    if marker.exists():
        print(f"Already extracted to {RAW_DIR}")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {ARCHIVE_PATH} to {RAW_DIR} ...")
    with tarfile.open(str(ARCHIVE_PATH), "r:gz") as tar:
        tar.extractall(path=str(RAW_DIR))
    marker.write_text("done")
    print("  Extraction complete.")


def convert_and_write_manifest():
    """Walk the extracted tree, convert FLAC -> 16kHz mono WAV, write manifest."""
    if MANIFEST_PATH.exists():
        print(f"Manifest already exists at {MANIFEST_PATH}")
        return

    test_clean_dir = RAW_DIR / "LibriSpeech" / "test-clean"
    if not test_clean_dir.exists():
        raise FileNotFoundError(f"Expected directory not found: {test_clean_dir}")

    WAV_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    # Walk speaker/chapter directories
    for spk_dir in sorted(test_clean_dir.iterdir()):
        if not spk_dir.is_dir():
            continue
        spk_id = spk_dir.name
        spk_wav_dir = WAV_DIR / spk_id
        spk_wav_dir.mkdir(parents=True, exist_ok=True)

        for chap_dir in sorted(spk_dir.iterdir()):
            if not chap_dir.is_dir():
                continue

            # Read transcriptions
            trans_file = chap_dir / f"{spk_id}-{chap_dir.name}.trans.txt"
            if not trans_file.exists():
                print(f"  Warning: transcription file not found: {trans_file}")
                continue

            transcripts = {}
            with open(trans_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(" ", 1)
                    if len(parts) == 2:
                        transcripts[parts[0]] = parts[1]

            # Convert each FLAC
            for flac_path in sorted(chap_dir.glob("*.flac")):
                utt_id = flac_path.stem  # e.g., 1089-134686-0000
                text = transcripts.get(utt_id, "")
                if not text:
                    print(f"  Warning: no transcript for {utt_id}")
                    continue

                wav_path = spk_wav_dir / f"{utt_id}.wav"
                if not wav_path.exists():
                    audio, sr = sf.read(str(flac_path), dtype="float32")
                    if audio.ndim > 1:
                        audio = np.mean(audio, axis=1)
                    if sr != SAMPLE_RATE:
                        audio = soxr.resample(audio, sr, SAMPLE_RATE)
                    audio = audio.astype(np.float32)
                    sf.write(str(wav_path), audio, SAMPLE_RATE)

                # Duration
                info = sf.info(str(wav_path))
                duration_s = round(info.duration, 3)

                # Relative path from project root
                rel_path = wav_path.relative_to(PROJECT_ROOT)

                rows.append({
                    "utt_id": utt_id,
                    "speaker": spk_id,
                    "sentence_id": utt_id,  # For LibriSpeech: sentence_id == utt_id
                    "path": str(rel_path),
                    "text": text,
                    "split": "test",
                    "duration_s": duration_s,
                })

    # Write manifest
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["utt_id", "speaker", "sentence_id", "path", "text", "split", "duration_s"])
        writer.writeheader()
        writer.writerows(rows)

    total_dur = sum(r["duration_s"] for r in rows)
    n_speakers = len(set(r["speaker"] for r in rows))
    print(f"\nManifest written: {len(rows)} utterances, {n_speakers} speakers, "
          f"{total_dur / 3600:.1f} hours -> {MANIFEST_PATH}")


def main():
    print("=" * 60)
    print("  Preparing LibriSpeech test-clean")
    print("=" * 60)

    download_archive()
    extract_archive()
    convert_and_write_manifest()

    print("\nDone! Run python -m s2t.evaluation.evaluate_wer with --accent librispeech to evaluate.")


if __name__ == "__main__":
    main()
