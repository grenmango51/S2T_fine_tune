import argparse
import json
import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
from praatio import textgrid
import soundfile as sf
import soxr

from concurrent.futures import ThreadPoolExecutor
from common import (
    ACCENT_SPEAKERS,
    BASE_MODEL,
    DATA_DIR,
    MANIFEST,
    PROJECT_ROOT,
    SAMPLE_RATE,
    SEED,
    SPEAKERS,
    corpus_wer,
    load_audio_16k,
    normalize_text,
)


def get_speaker_dir(spk: str) -> Path:
    if (PROJECT_ROOT / "corpus" / spk).exists():
        return PROJECT_ROOT / "corpus" / spk
    return PROJECT_ROOT / spk


def get_suitcase_dir() -> Path:
    if (PROJECT_ROOT / "corpus" / "suitcase_corpus").exists():
        return PROJECT_ROOT / "corpus" / "suitcase_corpus"
    return PROJECT_ROOT / "suitcase_corpus"


def prepare_l2_arctic_for_accent(accent: str = "vietnamese", force: bool = False):
    speakers = ACCENT_SPEAKERS[accent.lower()]
    accent_data_dir = DATA_DIR if accent.lower() == "vietnamese" else DATA_DIR / accent.lower()
    wav16k_dir = accent_data_dir / "wav16k"
    wav16k_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover all sentence IDs
    spk_ids = {}
    for spk in speakers:
        spk_txt_dir = get_speaker_dir(spk) / "transcript"
        txt_files = [f.stem for f in spk_txt_dir.glob("*.txt")]
        spk_ids[spk] = sorted(txt_files)
        print(f"Speaker {spk}: {len(txt_files)} transcript files found.")

    all_sentence_ids = sorted(list(set.union(*(set(ids) for ids in spk_ids.values()))))
    print(f"Total unique sentence IDs for {accent}: {len(all_sentence_ids)}")

    # 2. Compute splits
    ids = sorted(list(all_sentence_ids))
    rng = random.Random(SEED)
    rng.shuffle(ids)

    n_test = round(0.05 * len(ids))
    n_val = round(0.05 * len(ids))
    test_ids = ids[:n_test]
    val_ids = ids[n_test : n_test + n_val]
    train_ids = ids[n_test + n_val :]

    split_ids = {
        "seed": SEED,
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }
    split_ids_file = accent_data_dir / "split_ids.json"
    with open(split_ids_file, "w", encoding="utf-8") as f:
        json.dump(split_ids, f, indent=2)
    print(f"Saved split IDs to {split_ids_file.relative_to(PROJECT_ROOT)}")

    sentence_to_split = {}
    for sid in train_ids:
        sentence_to_split[sid] = "train"
    for sid in val_ids:
        sentence_to_split[sid] = "val"
    for sid in test_ids:
        sentence_to_split[sid] = "test"

    # 3. Process audio and create manifest rows in parallel
    tasks = []
    for spk in speakers:
        spk_wav16k_dir = wav16k_dir / spk
        spk_wav16k_dir.mkdir(parents=True, exist_ok=True)
        spk_dir = get_speaker_dir(spk)
        spk_wav_dir = spk_dir / "wav"
        spk_txt_dir = spk_dir / "transcript"

        for sid in spk_ids[spk]:
            raw_wav_path = spk_wav_dir / f"{sid}.wav"
            txt_path = spk_txt_dir / f"{sid}.txt"
            if not raw_wav_path.exists() or not txt_path.exists():
                continue
            out_wav_path = spk_wav16k_dir / f"{sid}.wav"
            tasks.append((spk, sid, raw_wav_path, txt_path, out_wav_path))

    def convert_worker(item):
        spk, sid, raw_wav, txt_p, out_wav = item
        with open(txt_p, "r", encoding="utf-8") as f:
            text = " ".join(f.read().split())
        if not text:
            return None
        if force or not out_wav.exists():
            audio, sr = sf.read(str(raw_wav), dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            if sr != SAMPLE_RATE:
                audio_16k = soxr.resample(audio, sr, SAMPLE_RATE)
            else:
                audio_16k = audio
            sf.write(str(out_wav), audio_16k, SAMPLE_RATE, subtype="PCM_16")

        info = sf.info(str(out_wav))
        dur = round(info.duration, 4)
        rel_path = str(out_wav.relative_to(PROJECT_ROOT))
        return {
            "utt_id": f"{spk}_{sid}",
            "speaker": spk,
            "sentence_id": sid,
            "path": rel_path,
            "text": text,
            "split": sentence_to_split[sid],
            "duration_s": dur,
        }

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(convert_worker, tasks))

    rows = [r for r in results if r is not None]
    print(f"Processed L2-ARCTIC utterances for {accent}: {len(rows)}")
    return rows


def prepare_suitcase_for_accent(accent: str = "vietnamese", force: bool = False):
    speakers = [s.lower() for s in ACCENT_SPEAKERS[accent.lower()]]
    accent_data_dir = DATA_DIR if accent.lower() == "vietnamese" else DATA_DIR / accent.lower()
    suitcase_dir = get_suitcase_dir()
    out_dir = accent_data_dir / "wav16k" / "suitcase"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for spk in speakers:
        raw_wav_path = suitcase_dir / "wav" / f"{spk}.wav"
        tg_path = suitcase_dir / "annotation" / f"{spk}.TextGrid"
        txt_path = suitcase_dir / "transcript" / f"{spk}.txt"

        if not (raw_wav_path.exists() and tg_path.exists() and txt_path.exists()):
            print(f"Skipping suitcase for {spk} (recording or annotation not present)")
            continue

        audio_16k = load_audio_16k(raw_wav_path)
        total_dur = len(audio_16k) / SAMPLE_RATE

        with open(txt_path, "r", encoding="utf-8") as f:
            transcript_txt = " ".join(f.read().split())

        tg = textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=False)
        words_tier = tg.getTier("words")
        kept_intervals = [
            e for e in words_tier.entries
            if e.label.strip() and e.label.strip() not in {"sp", "sil", "spn", ""}
        ]

        chunks = []
        curr = []
        for w in kept_intervals:
            if not curr:
                curr.append(w)
            else:
                if w.end - curr[0].start <= 28.0:
                    curr.append(w)
                else:
                    chunks.append(curr)
                    curr = [w]
        if curr:
            chunks.append(curr)

        cut_points = [0.0]
        for i in range(len(chunks) - 1):
            mid = (chunks[i][-1].end + chunks[i + 1][0].start) / 2.0
            cut_points.append(mid)
        cut_points.append(total_dur)

        chunk_texts = []
        for k, chunk_words in enumerate(chunks):
            chunk_text = " ".join(w.label for w in chunk_words)
            chunk_texts.append(chunk_text)

            t_start = cut_points[k]
            t_end = cut_points[k + 1]

            s_start = int(round(t_start * SAMPLE_RATE))
            s_end = int(round(t_end * SAMPLE_RATE))
            chunk_audio = audio_16k[s_start:s_end]

            chunk_wav_file = out_dir / f"{spk}_{k}.wav"
            if force or not chunk_wav_file.exists():
                sf.write(str(chunk_wav_file), chunk_audio, SAMPLE_RATE, subtype="PCM_16")

            dur = round(len(chunk_audio) / SAMPLE_RATE, 4)
            rel_path = str(chunk_wav_file.relative_to(PROJECT_ROOT))
            rows.append({
                "utt_id": f"suitcase_{spk}_{k}",
                "speaker": spk.upper(),
                "sentence_id": f"suitcase_{spk}_{k}",
                "path": rel_path,
                "text": chunk_text,
                "split": "suitcase",
                "duration_s": dur,
            })

        joined_chunks = " ".join(chunk_texts)
        wer = corpus_wer([transcript_txt], [joined_chunks])
        print(f"Suitcase speaker {spk}: {len(chunks)} chunks, audio dur {total_dur:.2f}s, textgrid vs transcript WER: {wer*100:.2f}%")

    return rows


def prepare_accent(accent: str, force: bool = False) -> pd.DataFrame:
    print(f"\n==========================================")
    print(f"   PREPARING DATA FOR ACCENT: {accent.upper()}   ")
    print(f"==========================================")
    accent_data_dir = DATA_DIR if accent.lower() == "vietnamese" else DATA_DIR / accent.lower()
    accent_manifest = accent_data_dir / "manifest.csv"

    l2_rows = prepare_l2_arctic_for_accent(accent, force=force)
    suitcase_rows = prepare_suitcase_for_accent(accent, force=force)

    all_rows = l2_rows + suitcase_rows
    df = pd.DataFrame(all_rows)
    accent_manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(accent_manifest, index=False, encoding="utf-8")
    print(f"\nSaved manifest with {len(df)} rows to {accent_manifest.relative_to(PROJECT_ROOT)}")

    # Summary table
    summary = df.groupby(["split", "speaker"]).agg(
        utterances=("utt_id", "count"),
        total_duration_s=("duration_s", "sum"),
    ).reset_index()
    summary["duration_min"] = (summary["total_duration_s"] / 60.0).round(2)
    print(summary.to_string(index=False))

    split_summary = df.groupby("split").agg(
        utterances=("utt_id", "count"),
        total_hours=("duration_s", lambda x: round(x.sum() / 3600.0, 3)),
    ).reset_index()
    print("\n--- Split Summary ---")
    print(split_summary.to_string(index=False))

    # Verify every path exists
    for _, row in df.iterrows():
        p = PROJECT_ROOT / row["path"]
        assert p.exists(), f"Path does not exist: {p}"
    print(f"Verification passed: all {len(df)} files exist.")

    return df


def main():
    parser = argparse.ArgumentParser(description="Prepare L2-ARCTIC and Suitcase dataset for accents")
    parser.add_argument(
        "--accent",
        type=str,
        default="vietnamese",
        choices=list(ACCENT_SPEAKERS.keys()) + ["all"],
        help="Accent to prepare",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing 16k wavs")
    args = parser.parse_args()

    if args.accent == "all":
        for acc in ACCENT_SPEAKERS.keys():
            prepare_accent(acc, force=args.force)
    else:
        prepare_accent(args.accent, force=args.force)


if __name__ == "__main__":
    main()
