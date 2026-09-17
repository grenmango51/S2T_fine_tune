# Speech-to-Text Benchmark & Evaluation Report: Vietnamese Accent

**Date:** 2026-09-17 07:07:26  
**GPU:** NVIDIA GeForce RTX 5070  
**Dataset:** L2-ARCTIC (Vietnamese) | Train: 4072 utts (4.305 h) | Val: 228 utts (0.257 h) | Test: 228 utts (0.238 h) | Suitcase: 9 chunks (0.054 h)

## Table 1: Model Benchmark Across Splits

| Model Tag | Model Path / Hub ID | Split | WER (%) | CER (%) | Utterances | Decode Time (s) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `finetuned_medium_en_lora` | `models/whisper-medium-en-vi-accent` | `test` | **11.42%** | 6.05% | 228 | 21.12s |
| `finetuned_medium_en_lora` | `models/whisper-medium-en-vi-accent` | `suitcase` | **20.53%** | 11.03% | 9 | 1.58s |
| `zeroshot_large_v3_turbo` | `openai/whisper-large-v3-turbo` | `test` | 15.99% | 8.16% | 228 | 52.01s |
| `zeroshot_large_v3_turbo` | `openai/whisper-large-v3-turbo` | `suitcase` | 12.9% | 8.07% | 9 | 2.34s |
| `zeroshot_medium_en` | `openai/whisper-medium.en` | `test` | 18.85% | 9.97% | 228 | 21.88s |
| `zeroshot_medium_en` | `openai/whisper-medium.en` | `suitcase` | 15.25% | 9.23% | 9 | 1.64s |

## Table 2: Test Set Per-Speaker and Gender Breakdown (Vietnamese)

| Model Tag | HQTV (M) | PNV (F) | THV (F) | TLV (M) | Male Avg | Female Avg | Overall Test WER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `finetuned_medium_en_lora` | 13.39% | 5.31% | 10.24% | 16.73% | 15.06% | 7.78% | **11.42%** |
| `zeroshot_large_v3_turbo` | 18.9% | 6.3% | 15.94% | 22.83% | 20.87% | 11.12% | **15.99%** |
| `zeroshot_medium_en` | 21.85% | 7.87% | 18.7% | 26.97% | 24.41% | 13.29% | **18.85%** |

## Qualitative Examples (Fine-Tuned Model on Vietnamese Test Set)

### 5 Best Test Examples (Lowest WER)

| Speaker | Utterance | WER | Reference (Normalized) | Hypothesis (Normalized) |
| :--- | :--- | :---: | :--- | :--- |
| `HQTV` | `HQTV_arctic_a0013` | 0.0% | he was a head shorter than his companion of almost delicate physique | he was a head shorter than his companion of almost delicate physique |
| `HQTV` | `HQTV_arctic_a0018` | 0.0% | there was a change now | there was a change now |
| `HQTV` | `HQTV_arctic_a0038` | 0.0% | we will have to watch our chances | we will have to watch our chances |
| `HQTV` | `HQTV_arctic_a0049` | 0.0% | gregson was asleep when he reentered the cabin | gregson was asleep when he reentered the cabin |
| `HQTV` | `HQTV_arctic_a0090` | 0.0% | the singing voice approached rapidly | the singing voice approached rapidly |

### 5 Worst Test Examples (Highest WER)

| Speaker | Utterance | WER | Reference (Normalized) | Hypothesis (Normalized) |
| :--- | :--- | :---: | :--- | :--- |
| `HQTV` | `HQTV_arctic_b0198` | 100.0% | i use great trouble advisedly | i do a great job of advisory |
| `TLV` | `TLV_arctic_a0018` | 80.0% | there was a change now | they were exchanged now |
| `HQTV` | `HQTV_arctic_a0158` | 75.0% | does that look good | the dust looked good |
| `HQTV` | `HQTV_arctic_b0309` | 66.7% | nor was elam harnish an exception | nowhere alum harnessed an exception |
| `TLV` | `TLV_arctic_a0509` | 63.6% | yet in accordance with ernest is test of truth it worked | yet it is accordance with ernest says the truth it is worth |

## Conclusion

Fine-tuning `whisper-medium.en` with LoRA on Vietnamese-accented English achieved an in-domain test WER of **11.42%**, down from **18.85%** on the zero-shot baseline (a **39.4%** relative WER reduction). Furthermore, the fine-tuned medium model surpassed zero-shot `whisper-large-v3-turbo` (**11.42%** vs **15.99%**).
