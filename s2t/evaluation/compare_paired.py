#!/usr/bin/env python3
import argparse
import json
import pathlib
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any

from s2t.common import PROJECT_ROOT, _splitmerge_edits, load_manifest

def load_report(path: pathlib.Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_metrics(sample: Dict[str, Any]) -> None:
    if "errors_splitmerge" not in sample or "errors_std" not in sample or "n_ref_words" not in sample:
        ref_norm = sample.get("ref_norm", "")
        hyp_norm = sample.get("hyp_norm", "")
        std_err, splitmerge_err, n_ref = _splitmerge_edits(ref_norm, hyp_norm)
        sample["errors_std"] = std_err
        sample["errors_splitmerge"] = splitmerge_err
        sample["n_ref_words"] = n_ref

def main():
    parser = argparse.ArgumentParser(description="Compare two evaluation reports with paired bootstrap resampling.")
    parser.add_argument("--accent", required=True, help="Accent used for the evaluation (e.g. cmu_us)")
    parser.add_argument("--baseline", required=True, help="Baseline model tag (e.g. zeroshot_medium_en_cmu_us)")
    parser.add_argument("--candidate", required=True, help="Candidate model tag (e.g. finetuned_medium_en_cmu_us_lora)")
    parser.add_argument("--split", required=True, help="Data split to compare (e.g. test)")
    args = parser.parse_args()

    reports_dir = PROJECT_ROOT / "reports"
    baseline_path = reports_dir / f"{args.baseline}.json"
    candidate_path = reports_dir / f"{args.candidate}.json"

    if not baseline_path.exists():
        print(f"Error: Baseline report not found at {baseline_path}", file=sys.stderr)
        sys.exit(1)
    if not candidate_path.exists():
        print(f"Error: Candidate report not found at {candidate_path}", file=sys.stderr)
        sys.exit(1)

    baseline_data = load_report(baseline_path)
    candidate_data = load_report(candidate_path)

    base_samples = baseline_data.get("splits", {}).get(args.split, {}).get("samples", [])
    cand_samples = candidate_data.get("splits", {}).get(args.split, {}).get("samples", [])

    if not base_samples or not cand_samples:
        print("Error: Missing samples in one or both reports for the given split.", file=sys.stderr)
        sys.exit(1)

    base_dict = {s["utt_id"]: s for s in base_samples}
    cand_dict = {s["utt_id"]: s for s in cand_samples}

    common_utts = sorted(list(set(base_dict.keys()).intersection(set(cand_dict.keys()))))

    if not common_utts:
        print("Error: No common utterances found between the two reports.", file=sys.stderr)
        sys.exit(1)

    try:
        manifest_df = load_manifest(accent=args.accent)
        if "sentence_id" not in manifest_df.columns:
            manifest_df["sentence_id"] = manifest_df["utt_id"]
        utt2sent = dict(zip(manifest_df["utt_id"], manifest_df["sentence_id"]))
    except Exception as e:
        print(f"Warning: Could not load manifest ({e}). Assuming sentence_id = utt_id", file=sys.stderr)
        utt2sent = {}

    merged_data = []
    for uid in common_utts:
        bs = base_dict[uid]
        cs = cand_dict[uid]
        ensure_metrics(bs)
        ensure_metrics(cs)

        sent_id = utt2sent.get(uid, uid)

        merged_data.append({
            "utt_id": uid,
            "sentence_id": sent_id,
            "ref_norm": bs.get("ref_norm", ""),
            "base_hyp_norm": bs.get("hyp_norm", ""),
            "cand_hyp_norm": cs.get("hyp_norm", ""),
            "base_std": bs["errors_std"],
            "base_sm": bs["errors_splitmerge"],
            "cand_std": cs["errors_std"],
            "cand_sm": cs["errors_splitmerge"],
            "n_ref_words": bs["n_ref_words"]
        })

    # 1. Paired regression listing
    regressions = []
    improvements = []

    tot_base_sm = 0
    tot_cand_sm = 0
    affected_base = 0
    affected_cand = 0

    for d in merged_data:
        diff_sm = d["cand_sm"] - d["base_sm"]
        if diff_sm > 0:
            regressions.append((diff_sm, d))
        elif diff_sm < 0:
            improvements.append((-diff_sm, d))

        tot_base_sm += d["base_sm"]
        tot_cand_sm += d["cand_sm"]
        if d["base_sm"] > 0:
            affected_base += 1
        if d["cand_sm"] > 0:
            affected_cand += 1

    regressions.sort(key=lambda x: x[0], reverse=True)
    improvements.sort(key=lambda x: x[0], reverse=True)

    lines = []
    lines.append(f"# Paired Comparison: {args.candidate} vs {args.baseline}")
    lines.append(f"Split: {args.split} | Accent: {args.accent}")
    lines.append("")
    lines.append("## Discounted Edit Counts (Split/Merge Tolerant)")
    lines.append(f"- Baseline: {tot_base_sm} edits, {affected_base} affected utterances")
    lines.append(f"- Candidate: {tot_cand_sm} edits, {affected_cand} affected utterances")
    lines.append("")

    lines.append(f"## Regressions ({len(regressions)} utterances)")
    for diff, d in regressions:
        lines.append(f"### {d['utt_id']} (Diff: +{diff} errors)")
        lines.append(f"- **Ref**:  {d['ref_norm']}")
        lines.append(f"- **Base**: {d['base_hyp_norm']} (err: {d['base_sm']})")
        lines.append(f"- **Cand**: {d['cand_hyp_norm']} (err: {d['cand_sm']})")
        lines.append("")

    lines.append(f"## Improvements ({len(improvements)} utterances)")
    lines.append(f"Showing top 10 of {len(improvements)} improvements.")
    for diff, d in improvements[:10]:
        lines.append(f"### {d['utt_id']} (Diff: -{diff} errors)")
        lines.append(f"- **Ref**:  {d['ref_norm']}")
        lines.append(f"- **Base**: {d['base_hyp_norm']} (err: {d['base_sm']})")
        lines.append(f"- **Cand**: {d['cand_hyp_norm']} (err: {d['cand_sm']})")
        lines.append("")

    # 2. Paired bootstrap by sentence
    sentences = {}
    for d in merged_data:
        sid = d["sentence_id"]
        if sid not in sentences:
            sentences[sid] = []
        sentences[sid].append(d)

    sent_keys = list(sentences.keys())
    n_sents = len(sent_keys)

    np.random.seed(42)
    boot_diffs_std = []
    boot_diffs_sm = []

    for _ in range(2000):
        sample_idxs = np.random.randint(0, n_sents, size=n_sents)

        sum_base_std = 0
        sum_cand_std = 0
        sum_base_sm = 0
        sum_cand_sm = 0
        sum_n_ref = 0

        for idx in sample_idxs:
            sid = sent_keys[idx]
            for d in sentences[sid]:
                sum_base_std += d["base_std"]
                sum_cand_std += d["cand_std"]
                sum_base_sm += d["base_sm"]
                sum_cand_sm += d["cand_sm"]
                sum_n_ref += d["n_ref_words"]

        if sum_n_ref == 0:
            sum_n_ref = 1

        base_wer_std = sum_base_std / sum_n_ref
        cand_wer_std = sum_cand_std / sum_n_ref
        base_wer_sm = sum_base_sm / sum_n_ref
        cand_wer_sm = sum_cand_sm / sum_n_ref

        boot_diffs_std.append(cand_wer_std - base_wer_std)
        boot_diffs_sm.append(cand_wer_sm - base_wer_sm)

    boot_diffs_std = np.array(boot_diffs_std)
    boot_diffs_sm = np.array(boot_diffs_sm)

    total_ref_words = max(1, sum(d["n_ref_words"] for d in merged_data))
    pt_base_std = sum(d["base_std"] for d in merged_data) / total_ref_words
    pt_cand_std = sum(d["cand_std"] for d in merged_data) / total_ref_words
    pt_base_sm = sum(d["base_sm"] for d in merged_data) / total_ref_words
    pt_cand_sm = sum(d["cand_sm"] for d in merged_data) / total_ref_words

    pt_diff_std = pt_cand_std - pt_base_std
    pt_diff_sm = pt_cand_sm - pt_base_sm

    lines.append("## Paired Bootstrap by Sentence (Candidate - Baseline)")
    lines.append(f"**Standard WER Diff**: {pt_diff_std:.4f} "
                 f"(95% CI: [{np.percentile(boot_diffs_std, 2.5):.4f}, {np.percentile(boot_diffs_std, 97.5):.4f}])")
    lines.append(f"**Split/Merge Tolerant WER Diff**: {pt_diff_sm:.4f} "
                 f"(95% CI: [{np.percentile(boot_diffs_sm, 2.5):.4f}, {np.percentile(boot_diffs_sm, 97.5):.4f}])")
    lines.append("")

    out_text = "\n".join(lines)
    print(out_text)

    reports_dir.mkdir(parents=True, exist_ok=True)
    out_file = reports_dir / f"paired_{args.accent}_{args.candidate}_vs_{args.baseline}.md"
    out_file.write_text(out_text, encoding="utf-8")
    print(f"\nReport written to {out_file}")

if __name__ == "__main__":
    main()
