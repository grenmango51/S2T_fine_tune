# L2-ARCTIC Multi-Accent Whisper Fine-Tuning Summary Report

**Generated:** 2026-09-17 06:57:57  
**Hardware:** NVIDIA GeForce RTX 5070 (12 GB VRAM)  
**Base Model:** `openai/whisper-medium.en` (LoRA, r=32, alpha=64)  

## Fine-Tuned Model Weights & Benchmark Results Across All Regions

| Region / Accent | Standalone Merged Model Directory | Weights Size | Test Set WER | Test Set CER | Suitcase (Spontaneous) WER |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Vietnamese** | `whisper-medium-en-vi-accent` | 1.53 GB | **11.42%** | 6.05% | 20.53% |
| **Arabic** | `whisper-medium-en-arabic-accent` | 1.53 GB | **4.44%** | 2.04% | 11.23% |
| **Chinese** | `whisper-medium-en-chinese-accent` | 1.53 GB | **7.09%** | 3.57% | 16.54% |
| **Hindi** | `whisper-medium-en-hindi-accent` | 1.53 GB | **2.64%** | 1.3% | 5.17% |
| **Korean** | `whisper-medium-en-korean-accent` | 1.53 GB | **4.84%** | 2.21% | 8.26% |
| **Spanish** | `whisper-medium-en-spanish-accent` | 1.53 GB | **5.69%** | 2.49% | 9.98% |


All models are standalone merged FP16 models compatible with `WhisperForConditionalGeneration.from_pretrained()`.
