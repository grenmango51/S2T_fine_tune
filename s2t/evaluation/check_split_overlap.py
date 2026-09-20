#!/usr/bin/env python3

import argparse
import csv
import sys
import os
from glob import glob
from collections import defaultdict
from s2t.common import PROJECT_ROOT, DATA_DIR, normalize_text

def load_texts(accent: str, splits=None):
    """Load normalized texts for an accent and specific splits."""
    if splits is None:
        splits = ["train", "val", "test"]

    manifest_path = DATA_DIR / accent / "manifest.csv"
    if not manifest_path.exists():
        return {s: set() for s in splits}

    texts_by_split = defaultdict(set)
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                split = row.get("split")
                text = row.get("text", "")
                if split in splits:
                    texts_by_split[split].add(normalize_text(text))
    except Exception as e:
        print(f"Warning: Failed to read {manifest_path}: {e}", file=sys.stderr)

    return {s: texts_by_split[s] for s in splits}

def compute_text_overlap(model_accent: str, eval_accent: str) -> int:
    """Returns number of overlapping normalized texts between (train_model | val_model) and test_eval."""
    model_texts = load_texts(model_accent, ["train", "val"])
    eval_texts = load_texts(eval_accent, ["test"])

    model_train_val = model_texts["train"] | model_texts["val"]
    eval_test = eval_texts["test"]

    return len(model_train_val & eval_test)

def default_mode():
    accent_paths = sorted(glob(str(DATA_DIR / "cmu_*" / "manifest.csv")))
    accents = [os.path.basename(os.path.dirname(p)) for p in accent_paths]

    if not accents:
        print("No cmu_* accents found.")
        return

    # Within-accent
    accent_data = {}
    for acc in accents:
        texts = load_texts(acc, ["train", "val", "test"])
        accent_data[acc] = texts
        train = texts["train"]
        val = texts["val"]
        test = texts["test"]

        print(f"Accent: {acc}")
        print(f"  Sizes - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        print(f"  Overlaps - train&test: {len(train & test)}, train&val: {len(train & val)}, val&test: {len(val & test)}")
        print()

    # Cross-accent matrix
    print("Cross-accent test coverage (|(train_A | val_A) & test_B| / |test_B|):")
    for a in accents:
        for b in accents:
            if a != b:
                a_texts = accent_data[a]
                b_texts = accent_data[b]

                a_train_val = a_texts["train"] | a_texts["val"]
                b_test = b_texts["test"]

                if not b_test:
                    overlap_ratio = "N/A (test_B empty)"
                else:
                    overlap = len(a_train_val & b_test)
                    overlap_ratio = f"{overlap}/{len(b_test)}"
                print(f"  {a} -> {b}: {overlap_ratio}")

def main():
    parser = argparse.ArgumentParser(description="Check split overlaps")
    parser.add_argument("--model_accent", type=str, help="Model accent (e.g., cmu_us)")
    parser.add_argument("--eval_accent", type=str, help="Eval accent (e.g., cmu_german or librispeech)")
    args = parser.parse_args()

    if args.model_accent and args.eval_accent:
        overlap = compute_text_overlap(args.model_accent, args.eval_accent)
        print(f"Overlap between {args.model_accent} (train|val) and {args.eval_accent} (test): {overlap}")
        if overlap > 0:
            sys.exit(1)
        sys.exit(0)
    elif args.model_accent or args.eval_accent:
        print("Error: Must provide both --model_accent and --eval_accent for guard mode.")
        sys.exit(1)
    else:
        default_mode()
        sys.exit(0)

if __name__ == "__main__":
    main()
