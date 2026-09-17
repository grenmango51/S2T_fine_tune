# Whisper Multi-Accent Speech-to-Text Fine-Tuning

Adapted speech-to-text models for non-native, accented English speech based on [`openai/whisper-medium.en`](https://huggingface.co/openai/whisper-medium.en).

This repository contains the complete end-to-end pipeline: data preprocessing, LoRA parameter-efficient fine-tuning, standalone weight merging, WER/CER benchmarking, and direct deployment to Hugging Face Hub.

---

## Published Models on Hugging Face

All 6 fine-tuned accent models are published under [**@Grenmango**](https://huggingface.co/Grenmango) as standalone merged FP16 models. They can be loaded directly with standard Hugging Face `transformers` without requiring `peft`:

| Accent | Hugging Face Model | Test Set WER | Test Set CER | Relative WER Improvement |
| :--- | :--- | :---: | :---: | :---: |
| 🇻🇳 **Vietnamese** | [`Grenmango/whisper-medium-en-vi-accent`](https://huggingface.co/Grenmango/whisper-medium-en-vi-accent) | **11.42%** | **6.05%** | **+39.4%** *(vs 18.85% base)* |
| 🇸🇦 **Arabic** | [`Grenmango/whisper-medium-en-arabic-accent`](https://huggingface.co/Grenmango/whisper-medium-en-arabic-accent) | **4.44%** | **2.04%** | **+45.9%** *(vs 8.21% base)* |
| 🇨🇳 **Chinese** | [`Grenmango/whisper-medium-en-chinese-accent`](https://huggingface.co/Grenmango/whisper-medium-en-chinese-accent) | **7.09%** | **3.57%** | **+37.4%** *(vs 11.33% base)* |
| 🇮🇳 **Hindi** | [`Grenmango/whisper-medium-en-hindi-accent`](https://huggingface.co/Grenmango/whisper-medium-en-hindi-accent) | **2.64%** | **1.30%** | **+41.9%** *(vs 4.54% base)* |
| 🇰🇷 **Korean** | [`Grenmango/whisper-medium-en-korean-accent`](https://huggingface.co/Grenmango/whisper-medium-en-korean-accent) | **4.84%** | **2.21%** | **+34.5%** *(vs 7.39% base)* |
| 🇪🇸 **Spanish** | [`Grenmango/whisper-medium-en-spanish-accent`](https://huggingface.co/Grenmango/whisper-medium-en-spanish-accent) | **5.69%** | **2.49%** | **+40.8%** *(vs 9.61% base)* |

---

## Quickstart

### 1. Transcribe with Hugging Face `pipeline`

```python
from transformers import pipeline

transcriber = pipeline(
    "automatic-speech-recognition",
    model="Grenmango/whisper-medium-en-vi-accent",
    chunk_length_s=30,
    device="cuda",  # or "cpu"
)

result = transcriber("audio.wav")
print("Transcription:", result["text"])
```

### 2. Local CLI Transcription

```bash
python transcribe.py path/to/sample.wav --model models/whisper-medium-en-vi-accent
```

---

## Project Structure

```
.
├── corpus/                 # Original L2-ARCTIC corpus recordings & phoneme annotations
│   ├── ABA/, HQTV/, ...    # 24 individual speaker folders
│   ├── suitcase_corpus/    # Spontaneous speech story subset
│   └── README_L2_ARCTIC.md # Original L2-ARCTIC documentation & license
├── data/                   # Resampled 16kHz mono audio & train/val/test split manifests
│   ├── manifest.csv        # Master manifest
│   ├── arabic/, chinese/, ...
│   └── wav16k/
├── docs/                   # Implementation specifications, hardware specs, & guides
│   ├── fine_tune_whisper_guide.md
│   ├── finetune_detailed_plan.md
│   └── future_benchmark.md
├── models/                 # Standalone merged FP16 models (ready for inference)
│   ├── whisper-medium-en-arabic-accent/
│   ├── whisper-medium-en-chinese-accent/
│   ├── whisper-medium-en-hindi-accent/
│   ├── whisper-medium-en-korean-accent/
│   ├── whisper-medium-en-spanish-accent/
│   └── whisper-medium-en-vi-accent/
├── reports/                # Evaluation benchmarks, WER reports, and qualitative samples
├── runs/                   # Checkpoints, LoRA adapters, & TensorBoard logs
├── common.py               # Shared data collator, dataset loader, and text normalizer
├── prepare_data.py         # Audio resampling (16kHz), chunking, & manifest generation
├── train_lora.py           # PEFT / LoRA fine-tuning engine
├── merge_lora.py           # LoRA-to-FP16 weight merging utility
├── evaluate_wer.py         # WER & CER benchmark evaluation suite
├── run_baseline_benchmarks.py # Zero-shot baseline evaluation across accents
├── train_all_accents.py    # Multi-accent end-to-end orchestration pipeline
├── transcribe.py           # Standalone CLI inference entrypoint
├── requirements.txt        # Python package dependencies
└── requirements.lock.txt   # Locked package versions
```

---

## Methodology & Training Details

* **Base Model:** `openai/whisper-medium.en` (769M parameters).
* **Dataset:** [L2-ARCTIC](https://psi.engr.tamu.edu/l2-arctic-corpus/) (non-native English speech corpus).
* **Acoustic Input:** 80-channel log-Mel spectrograms resampled to 16,000 Hz.
* **LoRA Configuration:**
  * Rank: $r = 32$, $\alpha = 64$, Dropout: 0.05.
  * Target modules: `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc2`.
* **Hardware:** NVIDIA GeForce RTX 5070 (12 GB VRAM) using FP16 mixed-precision and gradient checkpointing.
* **Metric Normalization:** Text normalized using official Whisper English text normalizer prior to WER/CER computation via `jiwer`.

---

## Reproduction & Pipeline Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare data for an accent (e.g. Vietnamese)
python prepare_data.py --accent vietnamese

# 3. Fine-tune LoRA adapter
python train_lora.py --accent vietnamese --output runs/vi-lora --batch_size 8 --grad_accum 2 --epochs 10

# 4. Merge adapter into a standalone model
python merge_lora.py --adapter runs/vi-lora/best_adapter --out models/whisper-medium-en-vi-accent

# 5. Evaluate WER & CER
python evaluate_wer.py --model models/whisper-medium-en-vi-accent --accent vietnamese --splits test --tag eval_vi

# 6. Run the complete pipeline across all accents
python train_all_accents.py
```
