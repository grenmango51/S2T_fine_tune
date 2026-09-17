# Speech-to-Text Benchmark & Evaluation Report: Korean Accent

**Date:** 2026-09-17 07:07:26  
**GPU:** NVIDIA GeForce RTX 5070  
**Dataset:** L2-ARCTIC (Korean) | Train: 4068 utts (4.068 h) | Val: 228 utts (0.247 h) | Test: 228 utts (0.232 h) | Suitcase: 20 chunks (0.141 h)

## Table 1: Model Benchmark Across Splits

| Model Tag | Model Path / Hub ID | Split | WER (%) | CER (%) | Utterances | Decode Time (s) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `finetuned_medium_en_korean_lora` | `/m/home/home2/21/nguyena42/data/Downloads/Hobbies/S2T fine tune/models/whisper-medium-en-korean-accent` | `test` | **4.84%** | 2.21% | 228 | 11.5s |
| `finetuned_medium_en_korean_lora` | `/m/home/home2/21/nguyena42/data/Downloads/Hobbies/S2T fine tune/models/whisper-medium-en-korean-accent` | `suitcase` | **8.26%** | 5.5% | 20 | 1.72s |
| `zeroshot_large_v3_turbo_korean` | `openai/whisper-large-v3-turbo` | `test` | 6.26% | 2.84% | 228 | 23.65s |
| `zeroshot_large_v3_turbo_korean` | `openai/whisper-large-v3-turbo` | `suitcase` | 6.9% | 4.56% | 20 | 2.35s |
| `zeroshot_medium_en_korean` | `openai/whisper-medium.en` | `test` | 7.39% | 3.5% | 228 | 14.69s |
| `zeroshot_medium_en_korean` | `openai/whisper-medium.en` | `suitcase` | 8.48% | 6.12% | 20 | 1.82s |

## Table 2: Test Set Per-Speaker and Gender Breakdown (Korean)

| Model Tag | HJK (F) | HKK (M) | YDCK (F) | YKWK (M) | Male Avg | Female Avg | Overall Test WER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `finetuned_medium_en_korean_lora` | 2.54% | 6.85% | 5.28% | 4.7% | 5.77% | 3.91% | **4.84%** |
| `zeroshot_large_v3_turbo_korean` | 4.31% | 8.02% | 6.07% | 6.65% | 7.34% | 5.19% | **6.26%** |
| `zeroshot_medium_en_korean` | 4.11% | 10.37% | 7.24% | 7.83% | 9.1% | 5.68% | **7.39%** |

## Qualitative Examples (Fine-Tuned Model on Korean Test Set)

### 5 Best Test Examples (Lowest WER)

| Speaker | Utterance | WER | Reference (Normalized) | Hypothesis (Normalized) |
| :--- | :--- | :---: | :--- | :--- |
| `HJK` | `HJK_arctic_a0018` | 0.0% | there was a change now | there was a change now |
| `HJK` | `HJK_arctic_a0023` | 0.0% | a combination of canadian capital quickly organized and petitioned for the same privileges | a combination of canadian capital quickly organized and petitioned for the same privileges |
| `HJK` | `HJK_arctic_a0038` | 0.0% | we will have to watch our chances | we will have to watch our chances |
| `HJK` | `HJK_arctic_a0049` | 0.0% | gregson was asleep when he reentered the cabin | gregson was asleep when he reentered the cabin |
| `HJK` | `HJK_arctic_a0050` | 0.0% | in spite of their absurdity the words affected philip curiously | in spite of their absurdity the words affected philip curiously |

### 5 Worst Test Examples (Highest WER)

| Speaker | Utterance | WER | Reference (Normalized) | Hypothesis (Normalized) |
| :--- | :--- | :---: | :--- | :--- |
| `YKWK` | `YKWK_arctic_a0150` | 50.0% | goodbye pierre he shouted | good bye pierre he shouted |
| `YDCK` | `YDCK_arctic_a0150` | 50.0% | goodbye pierre he shouted | good bye pierre he shouted |
| `HKK` | `HKK_arctic_b0376` | 50.0% | thought i and a worthy fool he proved | thought i and the world be full he proved |
| `HKK` | `HKK_arctic_a0150` | 50.0% | goodbye pierre he shouted | good bye pierre he shouted |
| `HJK` | `HJK_arctic_a0150` | 50.0% | goodbye pierre he shouted | good bye pierre he shouted |

## Conclusion

Fine-tuning `whisper-medium.en` with LoRA on Korean-accented English achieved an in-domain test WER of **4.84%**, down from **7.39%** on the zero-shot baseline (a **34.5%** relative WER reduction). Furthermore, the fine-tuned medium model surpassed zero-shot `whisper-large-v3-turbo` (**4.84%** vs **6.26%**).
