import datetime
import json
import os
from typing import List, Optional
import torch

from .config import BenchmarkConfig
from .metrics import ModelResults


def generate_benchmark_reports(
    results: List[ModelResults],
    config: BenchmarkConfig,
) -> tuple[str, str]:
    """Generate both JSON and Markdown benchmark reports."""
    os.makedirs(config.output_dir, exist_ok=True)
    json_path = os.path.join(config.output_dir, "benchmark_results.json")
    md_path = os.path.join(config.output_dir, "benchmark_report.md")

    # 1. Generate JSON report
    report_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "config": {
            "manifest_path": config.manifest_path,
            "local_model": config.local_model,
            "openrouter_models": config.openrouter_models,
            "language": config.language,
            "device": config.device,
            "split": config.split,
            "speaker": config.speaker,
            "max_samples": config.max_samples,
        },
        "results": [r.to_dict() for r in results],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    # 2. Generate Markdown report
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else config.device.upper()

    total_utts = results[0].total_utterances if results else 0

    lines = [
        "# Speech-to-Text Model Benchmark Report",
        "",
        f"**Date:** {now_str}  ",
        f"**Device:** {gpu_name}  ",
        f"**Manifest:** `{config.manifest_path}` ({total_utts} utterances evaluated)  ",
    ]
    if config.split:
        lines.append(f"**Split Filter:** `{config.split}`  ")
    if config.speaker:
        lines.append(f"**Speaker Filter:** `{config.speaker}`  ")

    lines.extend([
        "",
        "## Table 1: Model Accuracy & Performance Summary",
        "",
        "| Model | Type | Corpus WER (%) | Corpus CER (%) | Mean Latency (s) | Median Latency (s) | Est. Cost | Success Rate |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    # Find lowest WER for bolding
    valid_wers = [r.corpus_wer for r in results if r.successful_utterances > 0]
    min_wer = min(valid_wers) if valid_wers else None

    for r in results:
        wer_str = f"**{r.corpus_wer:.2f}%**" if (min_wer is not None and r.corpus_wer == min_wer) else f"{r.corpus_wer:.2f}%"
        cost_str = f"${r.total_cost_usd:.4f}" if r.total_cost_usd is not None else ("$0.00" if r.model_type == "local" else "N/A")
        succ_rate = f"{r.successful_utterances}/{r.total_utterances}"
        lines.append(
            f"| `{r.model_name}` | {r.model_type.capitalize()} | {wer_str} | {r.corpus_cer:.2f}% | "
            f"{r.mean_latency_s:.2f}s | {r.median_latency_s:.2f}s | {cost_str} | {succ_rate} |"
        )

    # Table 2: Speaker breakdown (if speakers present)
    all_speakers = sorted(list({spk for r in results for spk in r.speaker_results.keys()}))
    if all_speakers:
        spk_cols = " | ".join([f"{spk} WER" for spk in all_speakers])
        header_align = " | ".join([":---:" for _ in all_speakers])
        lines.extend([
            "",
            "## Table 2: Per-Speaker Accuracy Breakdown",
            "",
            f"| Model | {spk_cols} | Overall WER |",
            f"| :--- | {header_align} | :---: |",
        ])
        for r in results:
            spk_cells = " | ".join([
                f"{r.speaker_results.get(spk, {}).get('wer', 'N/A')}%"
                for spk in all_speakers
            ])
            lines.append(f"| `{r.model_name}` | {spk_cells} | **{r.corpus_wer:.2f}%** |")

    # Relative improvements vs local model
    local_res = next((r for r in results if r.model_type == "local"), None)
    if local_res and len(results) > 1:
        lines.extend([
            "",
            "## Relative Accuracy Comparison",
            "",
        ])
        base_wer = local_res.corpus_wer
        for r in results:
            if r.model_name == local_res.model_name:
                continue
            if base_wer > 0:
                rel_diff = ((base_wer - r.corpus_wer) / base_wer) * 100.0
                if rel_diff > 0:
                    lines.append(
                        f"- **`{r.model_name}`** achieved **{r.corpus_wer:.2f}% WER**, representing a **{rel_diff:.1f}% relative WER reduction** over the local baseline `{local_res.model_name}` ({base_wer:.2f}%)."
                    )
                else:
                    lines.append(
                        f"- **`{r.model_name}`** achieved **{r.corpus_wer:.2f}% WER** (local baseline `{local_res.model_name}` is {abs(rel_diff):.1f}% better at {base_wer:.2f}%)."
                    )

    # Qualitative examples (Best & Worst)
    lines.extend([
        "",
        "## Qualitative Samples",
        "",
    ])

    for r in results:
        valid_utts = [u for u in r.utterance_results if not u.error]
        if not valid_utts:
            continue

        sorted_by_wer = sorted(valid_utts, key=lambda x: x.wer)
        best_samples = sorted_by_wer[:5]
        worst_samples = sorted_by_wer[-5:][::-1]

        lines.extend([
            f"### Model: `{r.model_name}`",
            "",
            "#### Top 5 Most Accurate Transcriptions",
            "",
            "| Utterance ID | Speaker | WER | Reference (Normalized) | Hypothesis (Normalized) |",
            "| :--- | :---: | :---: | :--- | :--- |",
        ])
        for s in best_samples:
            lines.append(
                f"| `{s.utt_id}` | {s.speaker or 'N/A'} | {s.wer:.1f}% | {s.ref_norm} | {s.hyp_norm} |"
            )

        lines.extend([
            "",
            "#### 5 Highest Error Transcriptions",
            "",
            "| Utterance ID | Speaker | WER | Reference (Normalized) | Hypothesis (Normalized) |",
            "| :--- | :---: | :---: | :--- | :--- |",
        ])
        for s in worst_samples:
            lines.append(
                f"| `{s.utt_id}` | {s.speaker or 'N/A'} | {s.wer:.1f}% | {s.ref_norm} | {s.hyp_norm} |"
            )
        lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nSaved benchmark JSON report:     {json_path}")
    print(f"Saved benchmark Markdown report: {md_path}")

    return json_path, md_path
