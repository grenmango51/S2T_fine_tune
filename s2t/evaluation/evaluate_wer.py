import argparse
from s2t.datasets.cmu_arctic import CMU_ACCENT_SPEAKERS
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

from s2t.common import (
    ACCENT_SPEAKERS,
    BASE_MODEL,
    DATA_DIR,
    MANIFEST,
    PROJECT_ROOT,
    SAMPLE_RATE,
    SPEAKERS,
    _splitmerge_edits,
    corpus_cer,
    corpus_wer,
    corpus_wer_splitmerge,
    load_audio_16k,
    load_manifest,
    normalize_speaker,
    normalize_text,
)

REPORTS_DIR = PROJECT_ROOT / "reports"


def _speaker_split_metrics(report: Dict, split: str, speaker: str) -> Optional[Dict]:
    split_data = report.get("splits", {}).get(split, {})
    return split_data.get("speaker_results", {}).get(speaker)


def generate_speaker_eval_report(
    all_reports: List[Dict],
    accent: str,
    speaker: str,
) -> None:
    """Write a focused report with like-for-like metrics for one speaker."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_md_path = REPORTS_DIR / f"eval_report_{accent}_{speaker.lower()}.md"
    df = load_manifest(None, accent=accent, speaker=speaker)

    personalized = next(
        (report for report in all_reports if report.get("speaker") == speaker),
        None,
    )
    general_accent = next(
        (report for report in all_reports if report.get("tag") == "finetuned_medium_en_lora"),
        None,
    )
    zero_shot = next(
        (report for report in all_reports if report.get("tag") == "zeroshot_medium_en"),
        None,
    )

    comparisons = [
        ("Zero-shot Whisper medium.en", zero_shot),
        ("Four-speaker Vietnamese model", general_accent),
        (f"Personalized {speaker} model", personalized),
    ]

    lines = [
        f"# Personalized Vietnamese Whisper Evaluation: {speaker}",
        "",
        f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**GPU:** {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}  ",
        (
            f"**Data:** {len(df[df['split'] == 'train'])} train, "
            f"{len(df[df['split'] == 'val'])} validation, "
            f"{len(df[df['split'] == 'test'])} test, and "
            f"{len(df[df['split'] == 'suitcase'])} spontaneous utterance(s)"
        ),
        "",
        "## Like-for-like HQTV results" if speaker == "HQTV" else f"## Like-for-like {speaker} results",
        "",
        "| Model | Test WER | Test CER | Spontaneous WER | Spontaneous CER |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]

    for label, report in comparisons:
        if report is None:
            continue
        test = _speaker_split_metrics(report, "test", speaker)
        spontaneous = _speaker_split_metrics(report, "suitcase", speaker)
        if test is None:
            continue
        spontaneous_wer = f"{spontaneous['wer']:.2f}%" if spontaneous else "N/A"
        spontaneous_cer = f"{spontaneous['cer']:.2f}%" if spontaneous else "N/A"
        lines.append(
            f"| {label} | **{test['wer']:.2f}%** | {test['cer']:.2f}% | "
            f"{spontaneous_wer} | {spontaneous_cer} |"
        )

    personalized_test = (
        _speaker_split_metrics(personalized, "test", speaker) if personalized else None
    )
    baseline_test = _speaker_split_metrics(zero_shot, "test", speaker) if zero_shot else None
    general_test = (
        _speaker_split_metrics(general_accent, "test", speaker) if general_accent else None
    )

    lines.extend(["", "## Result", ""])
    if personalized_test and baseline_test:
        reduction = 100.0 * (
            baseline_test["wer"] - personalized_test["wer"]
        ) / baseline_test["wer"]
        result = (
            f"The personalized model reached **{personalized_test['wer']:.2f}% test WER**, "
            f"a **{reduction:.1f}% relative reduction** from the speaker-specific "
            f"zero-shot baseline ({baseline_test['wer']:.2f}%)."
        )
        if general_test:
            general_reduction = 100.0 * (
                general_test["wer"] - personalized_test["wer"]
            ) / general_test["wer"]
            result += (
                f" It also improves {general_reduction:.1f}% relative to the existing "
                f"four-speaker Vietnamese model ({general_test['wer']:.2f}%)."
            )
        lines.append(result)

    personalized_spontaneous = (
        _speaker_split_metrics(personalized, "suitcase", speaker) if personalized else None
    )
    baseline_spontaneous = (
        _speaker_split_metrics(zero_shot, "suitcase", speaker) if zero_shot else None
    )
    if personalized_spontaneous and baseline_spontaneous:
        lines.extend([
            "",
            (
                f"The spontaneous result is **{personalized_spontaneous['wer']:.2f}% WER** "
                f"versus {baseline_spontaneous['wer']:.2f}% zero-shot. This split contains "
                "only one short sample, so it should be treated as qualitative rather than "
                "a stable generalization estimate."
            ),
        ])

    train_summary_path = PROJECT_ROOT / "runs" / f"{accent}-{speaker.lower()}-lora" / "train_summary.json"
    if train_summary_path.exists():
        with open(train_summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        lines.extend([
            "",
            "## Training summary",
            "",
            f"- Best validation WER: {summary.get('best_val_wer')}% at epoch {summary.get('best_epoch')}",
            f"- Completed: {summary.get('completed_epochs')} epochs / {summary.get('total_steps')} steps",
            f"- LoRA: r={summary.get('lora_r')}, alpha={summary.get('lora_alpha')}, learning rate={summary.get('learning_rate')}",
            f"- Peak VRAM: {summary.get('peak_vram_gb')} GB",
            f"- Training time: {summary.get('wall_clock_time_hours')} hours",
        ])

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nGenerated speaker evaluation report at {report_md_path.relative_to(PROJECT_ROOT)}")


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
    speaker: Optional[str] = None,
) -> Dict:
    df = load_manifest(split_name, accent=accent, speaker=speaker)
    scope = f"speaker '{speaker}'" if speaker else f"accent '{accent}'"
    print(f"\nEvaluating split '{split_name}' for {scope} ({len(df)} utterances, batch_size={batch_size})...")

    # Whisper stores this flag on GenerationConfig for current multilingual
    # checkpoints (for example large-v3-turbo), not reliably on model.config.
    # Without this check a single short English clip can be misdetected as a
    # different language and translated/transcribed incorrectly.
    is_multilingual = getattr(
        model.generation_config,
        "is_multilingual",
        getattr(model.config, "is_multilingual", False),
    )

    all_refs = []
    all_hyps = []
    all_refs_raw = []
    all_hyps_raw = []
    sample_records = []

    start_time = time.time()

    for i in range(0, len(df), batch_size):
        batch_rows = df.iloc[i : i + batch_size]
        audio_batch = [load_audio_16k(row["path"]) for _, row in batch_rows.iterrows()]

        inputs = processor(
            audio_batch,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_features = inputs.input_features.to(device, dtype=torch.float16)
        attention_mask = inputs.attention_mask.to(device)

        gen_kwargs = {}
        if is_multilingual:
            gen_kwargs["language"] = "en"
            gen_kwargs["task"] = "transcribe"

        with torch.inference_mode():
            pred_ids = model.generate(
                input_features,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

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

            std_e, tol_e, n_ref = _splitmerge_edits(norm_ref_eval, norm_hyp_eval)

            all_refs.append(norm_ref_eval)
            all_hyps.append(norm_hyp_eval)
            all_refs_raw.append(ref_raw)
            all_hyps_raw.append(pred_raw)

            sample_records.append({
                "utt_id": row["utt_id"],
                "speaker": row["speaker"],
                "ref_raw": ref_raw,
                "hyp_raw": pred_raw,
                "ref_norm": norm_ref,
                "hyp_norm": norm_hyp,
                "sample_wer": sample_wer,
                "sample_cer": sample_cer,
                "n_ref_words": n_ref,
                "errors_std": std_e,
                "errors_splitmerge": tol_e,
                "sample_wer_splitmerge": tol_e / max(n_ref, 1),
            })

    decode_time_s = round(time.time() - start_time, 2)
    split_wer = corpus_wer(all_refs, all_hyps)
    split_cer = corpus_cer(all_refs, all_hyps)
    split_wer_sm = corpus_wer_splitmerge(all_refs_raw, all_hyps_raw)

    print(f"Split '{split_name}' -> WER: {split_wer*100:.2f}%, CER: {split_cer*100:.2f}%, Decode Time: {decode_time_s:.2f}s")
    print(f"Split '{split_name}' -> WER: {split_wer*100:.2f}%, WER(sm): {split_wer_sm*100:.2f}%, CER: {split_cer*100:.2f}%, Decode Time: {decode_time_s:.2f}s")

    # Per-speaker analysis
    speaker_results = {}
    unique_speakers = sorted(list({r["speaker"] for r in sample_records}))
    for spk in unique_speakers:
        spk_recs = [r for r in sample_records if r["speaker"] == spk]
        spk_refs = [r["ref_norm"] for r in spk_recs]
        spk_hyps = [r["hyp_norm"] for r in spk_recs]
        spk_wer = corpus_wer(spk_refs, spk_hyps)
        spk_cer = corpus_cer(spk_refs, spk_hyps)
        spk_sm_err = sum(r["errors_splitmerge"] for r in spk_recs)
        spk_sm_ref = sum(r["n_ref_words"] for r in spk_recs)
        spk_wer_sm = spk_sm_err / max(spk_sm_ref, 1)
        speaker_results[spk] = {
            "wer": round(spk_wer * 100, 2),
            "wer_splitmerge": round(spk_wer_sm * 100, 2),
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
        "wer_splitmerge": round(split_wer_sm * 100, 2),
        "cer": round(split_cer * 100, 2),
        "speaker_results": speaker_results,
        "gender_results": gender_results,
        "samples": sample_records,
    }


def generate_eval_report(
    all_reports: List[Dict],
    accent: Optional[str] = None,
    speaker: Optional[str] = None,
):
    if speaker is not None:
        generate_speaker_eval_report(all_reports, accent or "vietnamese", speaker)
        return
    if accent in CMU_ACCENT_SPEAKERS or accent in ("librispeech", "personal"):
        all_reports = [r for r in all_reports if r.get("accent") == accent]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_suffix = f"_{accent}_{speaker.lower()}" if speaker else ""
    if accent and accent not in ("vietnamese",) and not speaker:
        report_suffix = f"_{accent}"
    report_md_path = REPORTS_DIR / f"eval_report{report_suffix}.md"

    versions = get_package_versions()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Load split stats
    df = load_manifest(None, accent=accent, speaker=speaker)
    split_stats = {}
    for s in ["train", "val", "test", "suitcase"]:
        sdf = df[df["split"] == s]
        dur_h = sdf["duration_s"].sum() / 3600.0
        dur_h = sdf["duration_s"].sum() / 3600.0 if len(sdf) > 0 else 0.0
        split_stats[s] = {"utts": len(sdf), "hours": round(dur_h, 3)}

    # Build dataset splits line, only showing non-empty splits
    split_parts = []
    for s, label in [("train", "Train"), ("val", "Val"), ("test", "Test"), ("suitcase", "Suitcase")]:
        if split_stats[s]["utts"] > 0:
            split_parts.append(f"{label}: {split_stats[s]['utts']} utts ({split_stats[s]['hours']} h)")
    dataset_splits_str = " | ".join(split_parts) if split_parts else "No splits found"

    lines = [
        "# Speech-to-Text Fine-Tuning Evaluation Report",
        "",
        f"**Date:** {now_str}  ",
        f"**GPU:** {gpu_name}  ",
        f"**Environment:** Python {os.sys.version.split()[0]}, Torch {versions.get('torch')}, Transformers {versions.get('transformers')}, PEFT {versions.get('peft')}, Accelerate {versions.get('accelerate')}, JiWER {versions.get('jiwer')}  ",
        f"**Dataset Splits:** {dataset_splits_str}",
        "",
        "## Table 1: Model Benchmark Across Splits",
        "",
        "| Model Tag | Model Path / Hub ID | Split | WER (%) | WER split/merge-tolerant (%) | CER (%) | Utterances | Decode Time (s) |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    lines.append("")
    lines.append("The tolerant column ignores only differences where the words concatenate to the same string (e.g. `good bye` vs `goodbye`). The gap between the two WER columns is the amount of split/merge drift; inspect the affected utterances rather than assuming it is harmless.")
    lines.append("")

    for rep in all_reports:
        tag = rep.get("tag", "unknown")
        model_id = rep.get("model", "unknown")
        for s_name, s_data in rep.get("splits", {}).items():
            wer_sm = s_data.get("wer_splitmerge", "N/A")
            wer_sm_str = f"{wer_sm:.2f}%" if isinstance(wer_sm, (int, float)) else str(wer_sm)
            lines.append(
                f"| `{tag}` | `{model_id}` | `{s_name}` | **{s_data['wer']:.2f}%** | {wer_sm_str} | {s_data['cer']:.2f}% | {s_data['n_utts']} | {s_data['decode_time_s']}s |"
            )

    report_speakers = ACCENT_SPEAKERS.get((accent or "").lower(), sorted(df["speaker"].dropna().unique()))
    speaker_columns = [str(spk) for spk in report_speakers]
    if speaker_columns:
        lines.extend([
            "",
            "## Table 2: Test Set Per-Speaker Breakdown",
            "",
            f"| Model Tag | {' | '.join(speaker_columns)} | Overall Test WER |",
            f"| :--- | {' | '.join(':---:' for _ in speaker_columns)} | :---: |",
        ])

        for rep in all_reports:
            tag = rep.get("tag", "unknown")
            test_data = rep.get("splits", {}).get("test")
            if not test_data:
                continue
            spk_res = test_data.get("speaker_results", {})
            tot_wer = test_data.get("wer", "N/A")
            tot_wer_sm = test_data.get("wer_splitmerge", None)
            per_speaker_cells = []
            for spk in speaker_columns:
                spk_data = spk_res.get(spk, {})
                w = spk_data.get("wer", "N/A")
                w_sm = spk_data.get("wer_splitmerge", None)
                if w_sm is not None and w != "N/A":
                    per_speaker_cells.append(f"{w}% / {w_sm}%")
                else:
                    per_speaker_cells.append(f"{w}%")
            tot_cell = f"**{tot_wer}%**" if tot_wer_sm is None else f"**{tot_wer}% / {tot_wer_sm}%**"
            lines.append(f"| `{tag}` | {' | '.join(per_speaker_cells)} | {tot_cell} |")

    # Check for fine-tuned model summary and qualitative examples
    run_name = (
        f"{accent}-lora"
        if accent in CMU_ACCENT_SPEAKERS or accent == "personal"
        else "medium-en-lora"
    )
    train_summary_file = PROJECT_ROOT / "runs" / run_name / "train_summary.json"
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

    # CMU reports use accent-qualified tags so their baselines can coexist with
    # the existing L2-ARCTIC reports. Other accents retain their legacy tags.
    qualified_baseline = accent in CMU_ACCENT_SPEAKERS or accent == "personal"
    zs_med_tag = (
        f"zeroshot_medium_en_{accent}"
        if qualified_baseline
        else "zeroshot_medium_en"
    )
    zs_turbo_tag = (
        f"zeroshot_large_v3_turbo_{accent}"
        if qualified_baseline
        else "zeroshot_large_v3_turbo"
    )
    zs_med = next((r for r in all_reports if r.get("tag") == zs_med_tag), None)
    zs_turbo = next((r for r in all_reports if r.get("tag") == zs_turbo_tag), None)

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
            accent_label = accent.replace("cmu_", "").replace("_", " ").title() if accent else "Vietnamese"
            if rel_reduction >= 0:
                baseline_comparison = (
                    f"down from **{zs_test_wer:.2f}%** on the zero-shot baseline "
                    f"(a **{rel_reduction:.1f}%** relative WER reduction)"
                )
            else:
                baseline_comparison = (
                    f"up from **{zs_test_wer:.2f}%** on the zero-shot baseline "
                    f"(a **{-rel_reduction:.1f}%** relative WER increase)"
                )
            lines.append(
                f"Fine-tuning `whisper-medium.en` with LoRA on {accent_label}-accented English achieved a test WER of **{ft_test_wer:.2f}%**, {baseline_comparison}.{turbo_comp}"
            )
        else:
            lines.append("Baseline evaluation complete. Fine-tuned model results will be recorded upon training completion.")
    else:
        lines.append("Fine-tuned test results are recorded above; no matching baseline comparison is available." if ft_rep else "No fine-tuned test result is available yet.")

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nGenerated evaluation report at {report_md_path.relative_to(PROJECT_ROOT)}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Whisper WER/CER on dataset splits")
    parser.add_argument("--model", type=str, default="openai/whisper-medium.en", help="Hub model ID or local directory")
    parser.add_argument("--splits", type=str, default="test,suitcase", help="Comma-separated splits to evaluate")
    parser.add_argument("--accent", type=str, default="vietnamese", help="Accent split to evaluate")
    parser.add_argument(
        "--speaker",
        type=str.upper,
        default=None,
        help="Evaluate one speaker only (for Vietnamese: HQTV, PNV, THV, or TLV)",
    )
    parser.add_argument("--model_accent", type=str, default=None,
                        help="Accent the model was trained on (for cross-accent leakage guard)")
    parser.add_argument("--allow_leakage", action="store_true",
                        help="Allow evaluation despite text leakage (prefixes tag with 'leaked_')")
    parser.add_argument("--batch_size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--tag", type=str, default=None, help="Tag for report JSON naming")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.speaker is not None:
        try:
            args.speaker = normalize_speaker(args.speaker, accent=args.accent)
        except ValueError as exc:
            parser.error(str(exc))

    # Cross-accent text leakage guard (Step 2.2)
    if args.model_accent and args.model_accent != args.accent:
        try:
            from s2t.evaluation.check_split_overlap import compute_text_overlap
            overlap = compute_text_overlap(args.model_accent, args.accent)
            if overlap > 0:
                if args.allow_leakage:
                    print(f"WARNING: {overlap} test texts overlap with {args.model_accent} train/val. "
                          f"Proceeding with --allow_leakage; tag will be prefixed with 'leaked_'.")
                else:
                    parser.error(
                        f"Text leakage detected: {overlap} of {args.accent}'s test texts appear in "
                        f"{args.model_accent}'s train/val set. Use --allow_leakage to override."
                    )
            else:
                print(f"Leakage check passed: 0 overlap between {args.model_accent} train/val and {args.accent} test.")
        except ImportError:
            print("Warning: check_split_overlap.py not found, skipping leakage guard.")

    tag = args.tag
    if not tag:
        tag = args.model.replace("/", "_").replace("\\", "_")

    # Prefix tag with 'leaked_' when leakage is allowed
    if args.allow_leakage and args.model_accent and args.model_accent != args.accent:
        if not tag.startswith("leaked_"):
            tag = f"leaked_{tag}"

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
            speaker=args.speaker,
        )
        splits_data[s] = result

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_json = {
        "tag": tag,
        "model": args.model,
        "accent": args.accent,
        "speaker": args.speaker,
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
                loaded = json.load(f)
                if isinstance(loaded, dict) and isinstance(loaded.get("splits"), dict):
                    all_reports.append(loaded)
            except Exception as e:
                print(f"Warning: could not parse {jf}: {e}")

    generate_eval_report(all_reports, accent=args.accent, speaker=args.speaker)


if __name__ == "__main__":
    main()
