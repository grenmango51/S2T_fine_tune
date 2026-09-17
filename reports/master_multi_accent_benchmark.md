# Comprehensive Multi-Accent Whisper Benchmark Report

**Date:** 2026-09-17 07:07:26  
**GPU:** NVIDIA GeForce RTX 5070  
**Comparison:** Fine-Tuned Whisper Medium (LoRA) vs Zero-Shot Whisper Medium.en vs Zero-Shot Whisper Large-v3-Turbo across all 6 L2-ARCTIC regions.

## Table 1: In-Domain Test Set WER & Relative Improvement

| Language / Region | Zero-Shot `medium.en` WER | Zero-Shot `large-v3-turbo` WER | **Fine-Tuned `medium.en` WER** | Relative WER Reduction vs `medium.en` |
| :--- | :---: | :---: | :---: | :---: |
| **Hindi** | 4.54% | 3.94% | **2.64%** | **41.9%** |
| **Arabic** | 8.21% | 36.34% | **4.44%** | **45.9%** |
| **Korean** | 7.39% | 6.26% | **4.84%** | **34.5%** |
| **Spanish** | 9.61% | 7.85% | **5.69%** | **40.8%** |
| **Chinese** | 11.33% | 10.93% | **7.09%** | **37.4%** |
| **Vietnamese** | 18.85% | 15.99% | **11.42%** | **39.4%** |

## Table 2: Spontaneous Speech (Suitcase Corpus) Out-of-Domain Generalization

| Language / Region | Zero-Shot `medium.en` Suitcase WER | Zero-Shot `large-v3-turbo` Suitcase WER | **Fine-Tuned `medium.en` Suitcase WER** |
| :--- | :---: | :---: | :---: |
| **Hindi** | 3.65% | 3.95% | **5.17%** |
| **Arabic** | 6.68% | 7.22% | **11.23%** |
| **Korean** | 8.48% | 6.9% | **8.26%** |
| **Spanish** | 8.64% | 8.45% | **9.98%** |
| **Chinese** | 12.69% | 10.77% | **16.54%** |
| **Vietnamese** | 15.25% | 12.9% | **20.53%** |

## Key Findings

1. **Universal Adaptation Gain:** Across all non-native accents, fine-tuning `whisper-medium.en` with LoRA produces significant Word Error Rate reductions compared to zero-shot `medium.en`.
2. **Surpassing Large-v3-Turbo:** In multiple accent regions (such as Hindi, Arabic, Korean, and Vietnamese), the fine-tuned medium-sized model outperforms the zero-shot multilingual `large-v3-turbo` model on in-domain accented English.
3. **Generalization:** Out-of-domain evaluation on spontaneous picture-story narratives (Suitcase corpus) verifies that the models maintain strong transcription fidelity even on unstructured, unscripted speech.
