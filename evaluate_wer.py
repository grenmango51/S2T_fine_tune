import argparse
import datetime
import json
import os
from pathlib import Path
import time
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration
import jiwer

from common import (
    BASE_MODEL,
    DATA_DIR,
    MANIFEST,
    PROJECT_ROOT,
    SAMPLE_RATE,
    SPEAKERS,
    corpus_cer,
    corpus_wer,
    load_audio_16k,
    load_manifest,
    normalize_text,
)

REPORTS_DIR = PROJECT_ROOT / "reports"


def get_package_versions() -> Dict[str, str]:
    import importlib.metadata
    pkgs = ["torch", "transformers", "peft", "accelerate", "jiwer", "soundfile", "soxr", "praatio"]
    versions = {}
    for pkg in pkgs:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except Exception:
            versions[pkg] = "unknown"
    return versions


def evaluate_model_on_split(
    model: WhisperForConditionalGeneration,
    processor: AutoProcessor,
    split_name: str,
    batch_size: int = 16,
    device: str = "cuda",
    accent: Optional[str] = None,
) -> Dict:
    df = load_manifest(split_name, accent=accent)
    print(f"\nEvaluating split '{split_name}' for accent '{accent}' ({len(df)} utterances, batch_size={batch_size})...")

    is_multilingual = getattr(model.config, "is_multilingual", False)

    all_refs = []
    all_hyps = []
    sample_records = []

    start_time = time.time()

    for i in range(0, len(df), batch_size):
        batch_rows = df.iloc[i : i + batch_size]
        audio_batch = [load_audio_16k(row["path"]) for _, row in batch_rows.iterrows()]

        inputs = processor(audio_batch, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        input_features = inputs.input_features.to(device, dtype=torch.float16)

        gen_kwargs = {}
        if is_multilingual:
            gen_kwargs["language"] = "en"
            gen_kwargs["task"] = "transcribe"

        with torch.inference_mode():
            pred_ids = model.generate(input_features, **gen_kwargs)

        preds = processor.batch_decode(pred_ids, skip_special_tokens=True)

        for (_, row), pred_raw in zip(batch_rows.iterrows(), preds):
            ref_raw = str(row["text"])
            norm_ref = normalize_text(ref_raw)
            norm_hyp = normalize_text(pred_raw)
            if not norm_hyp.strip():
                norm_hyp_eval = "empty"
            else:
                norm_hyp_eval = norm_hyp
            norm_ref_eval = norm_ref if norm_ref.strip() else "empty"

            sample_wer = float(jiwer.wer([norm_ref_eval], [norm_hyp_eval]))
            sample_cer = float(jiwer.cer([norm_ref_eval], [norm_hyp_eval]))

            all_refs.append(norm_ref_eval)
            all_hyps.append(norm_hyp_eval)

            sample_records.append({
                "utt_id": row["utt_id"],
                "speaker": row["speaker"],
                "ref_raw": ref_raw,
                "hyp_raw": pred_raw,
                "ref_norm": norm_ref,
                "hyp_norm": norm_hyp,
                "sample_wer": sample_wer,
                "sample_cer": sample_cer,
            })

    decode_time_s = round(time.time() - start_time, 2)
    split_wer = corpus_wer(all_refs, all_hyps)
    split_cer = corpus_cer(all_refs, all_hyps)

    print(f"Split '{split_name}' -> WER: {split_wer*100:.2f}%, CER: {split_cer*100:.2f}%, Decode Time: {decode_time_s:.2f}s")

    # Per-speaker analysis
    speaker_results = {}
    unique_speakers = sorted(list({r["speaker"] for r in sample_records}))
    for spk in unique_speakers:
        spk_recs = [r for r in sample_records if r["speaker"] == spk]
        spk_refs = [r["ref_norm"] for r in spk_recs]
        spk_hyps = [r["hyp_norm"] for r in spk_recs]
        spk_wer = corpus_wer(spk_refs, spk_hyps)
        spk_cer = corpus_cer(spk_refs, spk_hyps)
        speaker_results[spk] = {
            "wer": round(spk_wer * 100, 2),
            "cer": round(spk_cer * 100, 2),
            "n_utts": len(spk_recs),
        }

    # Gender analysis for test split
    gender_results = {}
    m_recs = [r for r in sample_records if r["speaker"] in ["HQTV", "TLV"]]
    f_recs = [r for r in sample_records if r["speaker"] in ["PNV", "THV"]]
    if m_recs:
        gender_results["M (HQTV+TLV)"] = {
            "wer": round(corpus_wer([r["ref_norm"] for r in m_recs], [r["hyp_norm"] for r in m_recs]) * 100, 2),
            "cer": round(corpus_cer([r["ref_norm"] for r in m_recs], [r["hyp_norm"] for r in m_recs]) * 100, 2),
            "n_utts": len(m_recs),
        }
    if f_recs:
        gender_results["F (PNV+THV)"] = {
            "wer": round(corpus_wer([r["ref_norm"] for r in f_recs], [r["hyp_norm"] for r in f_recs]) * 100, 2),
            "cer": round(corpus_cer([r["ref_norm"] for r in f_recs], [r["hyp_norm"] for r in f_recs]) * 100, 2),
            "n_utts": len(f_recs),
        }

    return {
        "split": split_name,
        "n_utts": len(df),
        "decode_time_s": decode_time_s,
        "wer": round(split_wer * 100, 2),
        "cer": round(split_cer * 100, 2),
        "speaker_results": speaker_results,
        "gender_results": gender_results,
        "samples": sample_records,
    }


def generate_eval_report(all_reports: List[Dict]):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_md_path = REPORTS_DIR / "eval_report.md"

    versions = get_package_versions()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Load split stats
    df = load_manifest(None)
    split_stats = {}
    for s in ["train", "val", "test", "suitcase"]:
        sdf = df[df["split"] == s]
        dur_h = sdf["duration_s"].sum() / 3600.0
        split_stats[s] = {"utts": len(sdf), "hours": round(dur_h, 3)}

    lines = [
        "# Speech-to-Text Fine-Tuning Evaluation Report",
        "",
        f"**Date:** {now_str}  ",
        f"**GPU:** {gpu_name}  ",
        f"**Environment:** Python {os.sys.version.split()[0]}, Torch {versions.get('torch')}, Transformers {versions.get('transformers')}, PEFT {versions.get('peft')}, Accelerate {versions.get('accelerate')}, JiWER {versions.get('jiwer')}  ",
        f"**Dataset Splits:** Train: {split_stats['train']['utts']} utts ({split_stats['train']['hours']} h) | Val: {split_stats['val']['utts']} utts ({split_stats['val']['hours']} h) | Test: {split_stats['test']['utts']} utts ({split_stats['test']['hours']} h) | Suitcase: {split_stats['suitcase']['utts']} chunks ({split_stats['suitcase']['hours']} h)",
        "",
        "## Table 1: Model Benchmark Across Splits",
        "",
        "| Model Tag | Model Path / Hub ID | Split | WER (%) | CER (%) | Utterances | Decode Time (s) |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: |",
    ]

    for rep in all_reports:
        tag = rep.get("tag", "unknown")
        model_id = rep.get("model", "unknown")
        for s_name, s_data in rep.get("splits", {}).items():
            lines.append(
                f"| `{tag}` | `{model_id}` | `{s_name}` | **{s_data['wer']:.2f}%** | {s_data['cer']:.2f}% | {s_data['n_utts']} | {s_data['decode_time_s']}s |"
            )

    lines.extend([
        "",
        "## Table 2: Test Set Per-Speaker and Gender Breakdown",
        "",
        "| Model Tag | HQTV (M) | PNV (F) | THV (F) | TLV (M) | Male Avg | Female Avg | Overall Test WER |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for rep in all_reports:
        tag = rep.get("tag", "unknown")
        test_data = rep.get("splits", {}).get("test")
        if not test_data:
            continue
        spk_res = test_data.get("speaker_results", {})
        gen_res = test_data.get("gender_results", {})
        hqtv_wer = spk_res.get("HQTV", {}).get("wer", "N/A")
        pnv_wer = spk_res.get("PNV", {}).get("wer", "N/A")
        thv_wer = spk_res.get("THV", {}).get("wer", "N/A")
        tlv_wer = spk_res.get("TLV", {}).get("wer", "N/A")
        m_wer = gen_res.get("M (HQTV+TLV)", {}).get("wer", "N/A")
        f_wer = gen_res.get("F (PNV+THV)", {}).get("wer", "N/A")
        tot_wer = test_data.get("wer", "N/A")
        lines.append(
            f"| `{tag}` | {hqtv_wer}% | {pnv_wer}% | {thv_wer}% | {tlv_wer}% | {m_wer}% | {f_wer}% | **{tot_wer}%** |"
        )

    # Check for fine-tuned model summary and qualitative examples
    train_summary_file = PROJECT_ROOT / "runs" / "medium-en-lora" / "train_summary.json"
    if train_summary_file.exists():
        with open(train_summary_file, "r", encoding="utf-8") as f:
            train_summary = json.load(f)
        lines.extend([
            "",
            "## Training Summary & Configuration",
            "",
            f"- **Batch Configuration:** batch size = {train_summary.get('per_device_train_batch_size')}, grad accum = {train_summary.get('gradient_accumulation_steps')} (effective batch size = {train_summary.get('effective_batch_size')})",
            f"- **LoRA Parameters:** r = {train_summary.get('lora_r')}, alpha = {train_summary.get('lora_alpha')}, lr = {train_summary.get('learning_rate')}",
            f"- **Epochs & Steps:** {train_summary.get('completed_epochs')} epochs, {train_summary.get('total_steps')} steps",
            f"- **Best Validation WER:** {train_summary.get('best_val_wer', 'N/A')}% (at epoch {train_summary.get('best_epoch', 'N/A')})",
            f"- **Peak VRAM:** {train_summary.get('peak_vram_gb', 'N/A')} GB",
            f"- **Wall-clock Training Time:** {train_summary.get('wall_clock_time_hours', 'N/A')} hours ({train_summary.get('wall_clock_time_sec', 'N/A')} s)",
        ])

    # Qualitative examples from fine-tuned model if available
    ft_rep = next((r for r in all_reports if "finetuned" in r.get("tag", "")), None)
    if ft_rep and "test" in ft_rep.get("splits", {}):
        samples = ft_rep["splits"]["test"].get("samples", [])
        if samples:
            sorted_by_wer = sorted(samples, key=lambda x: x["sample_wer"])
            best_5 = sorted_by_wer[:5]
            worst_5 = sorted_by_wer[-5:][::-1]

            lines.extend([
                "",
                "## Qualitative Examples (Fine-Tuned Model on Test Set)",
                "",
                "### 5 Best Test Examples (Lowest WER)",
                "",
                "| Speaker | Utterance | WER | Reference (Normalized) | Hypothesis (Normalized) |",
                "| :--- | :--- | :---: | :--- | :--- |",
            ])
            for s in best_5:
                lines.append(f"| `{s['speaker']}` | `{s['utt_id']}` | {s['sample_wer']*100:.1f}% | {s['ref_norm']} | {s['hyp_norm']} |")

            lines.extend([
                "",
                "### 5 Worst Test Examples (Highest WER)",
                "",
                "| Speaker | Utterance | WER | Reference (Normalized) | Hypothesis (Normalized) |",
                "| :--- | :--- | :---: | :--- | :--- |",
            ])
            for s in worst_5:
                lines.append(f"| `{s['speaker']}` | `{s['utt_id']}` | {s['sample_wer']*100:.1f}% | {s['ref_norm']} | {s['hyp_norm']} |")

    # Conclusion paragraph
    lines.extend([
        "",
        "## Conclusion",
        "",
    ])

    zs_med = next((r for r in all_reports if r.get("tag") == "zeroshot_medium_en"), None)
    zs_turbo = next((r for r in all_reports if r.get("tag") == "zeroshot_large_v3_turbo"), None)

    if ft_rep and zs_med:
        ft_test_wer = ft_rep.get("splits", {}).get("test", {}).get("wer")
        zs_test_wer = zs_med.get("splits", {}).get("test", {}).get("wer")
        if ft_test_wer is not None and zs_test_wer is not None:
            rel_reduction = ((zs_test_wer - ft_test_wer) / zs_test_wer) * 100.0
            turbo_test_wer = zs_turbo.get("splits", {}).get("test", {}).get("wer") if zs_turbo else None
            turbo_comp = ""
            if turbo_test_wer is not None:
                if ft_test_wer < turbo_test_wer:
                    turbo_comp = f" Furthermore, the fine-tuned medium model surpassed zero-shot `whisper-large-v3-turbo` ({ft_test_wer:.2f}% vs {turbo_test_wer:.2f}%)."
                else:
                    turbo_comp = f" Zero-shot `whisper-large-v3-turbo` achieved {turbo_test_wer:.2f}% WER compared to {ft_test_wer:.2f}% for fine-tuned medium.en."
            lines.append(
                f"Fine-tuning `whisper-medium.en` with LoRA on Vietnamese-accented English achieved a test WER of **{ft_test_wer:.2f}%**, down from **{zs_test_wer:.2f}%** on the zero-shot baseline (a **{rel_reduction:.1f}%** relative WER reduction).{turbo_comp} Additional benchmark comparisons are documented in [future_benchmark.md](file:///d:/Hoai%20Anh/Aalto/Hobbies/S2T%20fine%20tune/future_benchmark.md)."
            )
        else:
            lines.append("Baseline evaluation complete. Fine-tuned model results will be recorded upon training completion.")
    else:
        lines.append("Baseline evaluation complete. Fine-tuned model results will be recorded upon training completion.")

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nGenerated evaluation report at {report_md_path.relative_to(PROJECT_ROOT)}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Whisper WER/CER on dataset splits")
    parser.add_argument("--model", type=str, default="openai/whisper-medium.en", help="Hub model ID or local directory")
    parser.add_argument("--splits", type=str, default="test,suitcase", help="Comma-separated splits to evaluate")
    parser.add_argument("--accent", type=str, default="vietnamese", help="Accent split to evaluate")
    parser.add_argument("--batch_size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--tag", type=str, default=None, help="Tag for report JSON naming")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tag = args.tag
    if not tag:
        tag = args.model.replace("/", "_").replace("\\", "_")

    print(f"Loading model: {args.model} on {args.device}...")
    processor = AutoProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model, dtype=torch.float16, device_map=args.device)
    model.eval()

    splits_to_eval = [s.strip() for s in args.splits.split(",") if s.strip()]
    splits_data = {}

    for s in splits_to_eval:
        result = evaluate_model_on_split(
            model=model,
            processor=processor,
            split_name=s,
            batch_size=args.batch_size,
            device=args.device,
            accent=args.accent,
        )
        splits_data[s] = result

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_json = {
        "tag": tag,
        "model": args.model,
        "timestamp": datetime.datetime.now().isoformat(),
        "splits": splits_data,
    }

    tag_json_path = REPORTS_DIR / f"{tag}.json"
    with open(tag_json_path, "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2)
    print(f"Saved report JSON to {tag_json_path.relative_to(PROJECT_ROOT)}")

    # Collect all report JSONs in reports/
    all_reports = []
    for jf in sorted(REPORTS_DIR.glob("*.json")):
        with open(jf, "r", encoding="utf-8") as f:
            try:
                all_reports.append(json.load(f))
            except Exception as e:
                print(f"Warning: could not parse {jf}: {e}")

    generate_eval_report(all_reports)


if __name__ == "__main__":
    main()
