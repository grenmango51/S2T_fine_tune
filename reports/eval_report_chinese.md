# Speech-to-Text Benchmark & Evaluation Report: Chinese Accent

**Date:** 2026-09-17 07:07:26  
**GPU:** NVIDIA GeForce RTX 5070  
**Dataset:** L2-ARCTIC (Chinese) | Train: 4071 utts (4.297 h) | Val: 228 utts (0.255 h) | Test: 225 utts (0.235 h) | Suitcase: 12 chunks (0.075 h)

## Table 1: Model Benchmark Across Splits

| Model Tag | Model Path / Hub ID | Split | WER (%) | CER (%) | Utterances | Decode Time (s) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `finetuned_medium_en_chinese_lora` | `/m/home/home2/21/nguyena42/data/Downloads/Hobbies/S2T fine tune/models/whisper-medium-en-chinese-accent` | `test` | **7.09%** | 3.57% | 225 | 11.3s |
| `finetuned_medium_en_chinese_lora` | `/m/home/home2/21/nguyena42/data/Downloads/Hobbies/S2T fine tune/models/whisper-medium-en-chinese-accent` | `suitcase` | **16.54%** | 10.75% | 12 | 0.99s |
| `zeroshot_large_v3_turbo_chinese` | `openai/whisper-large-v3-turbo` | `test` | 10.93% | 5.45% | 225 | 23.71s |
| `zeroshot_large_v3_turbo_chinese` | `openai/whisper-large-v3-turbo` | `suitcase` | 10.77% | 6.89% | 12 | 1.43s |
| `zeroshot_medium_en_chinese` | `openai/whisper-medium.en` | `test` | 11.33% | 5.65% | 225 | 12.37s |
| `zeroshot_medium_en_chinese` | `openai/whisper-medium.en` | `suitcase` | 12.69% | 9.11% | 12 | 1.23s |

## Table 2: Test Set Per-Speaker and Gender Breakdown (Chinese)

| Model Tag | BWC (M) | LXC (F) | NCC (F) | TXHC (M) | Male Avg | Female Avg | Overall Test WER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `finetuned_medium_en_chinese_lora` | 9.22% | 6.01% | 8.23% | 4.92% | 7.05% | 7.12% | **7.09%** |
| `zeroshot_large_v3_turbo_chinese` | 13.63% | 13.03% | 10.24% | 6.89% | 10.23% | 11.63% | **10.93%** |
| `zeroshot_medium_en_chinese` | 12.22% | 14.63% | 10.84% | 7.68% | 9.93% | 12.74% | **11.33%** |

## Qualitative Examples (Fine-Tuned Model on Chinese Test Set)

### 5 Best Test Examples (Lowest WER)

| Speaker | Utterance | WER | Reference (Normalized) | Hypothesis (Normalized) |
| :--- | :--- | :---: | :--- | :--- |
| `BWC` | `BWC_arctic_a0018` | 0.0% | there was a change now | there was a change now |
| `BWC` | `BWC_arctic_a0038` | 0.0% | we will have to watch our chances | we will have to watch our chances |
| `BWC` | `BWC_arctic_a0049` | 0.0% | gregson was asleep when he reentered the cabin | gregson was asleep when he reentered the cabin |
| `BWC` | `BWC_arctic_a0093` | 0.0% | for a full minute he crouched and listened | for a full minute he crouched and listened |
| `BWC` | `BWC_arctic_a0108` | 0.0% | he waded into the edge of the water and began scrubbing himself | he waded into the edge of the water and began scrubbing himself |

### 5 Worst Test Examples (Highest WER)

| Speaker | Utterance | WER | Reference (Normalized) | Hypothesis (Normalized) |
| :--- | :--- | :---: | :--- | :--- |
| `TXHC` | `TXHC_arctic_a0150` | 50.0% | goodbye pierre he shouted | good bye pierre he shouted |
| `NCC` | `NCC_arctic_b0376` | 50.0% | thought i and a worthy fool he proved | thorpe eye and a worth full he proved |
| `NCC` | `NCC_arctic_b0309` | 50.0% | nor was elam harnish an exception | nor was eileen is harness an exception |
| `NCC` | `NCC_arctic_a0150` | 50.0% | goodbye pierre he shouted | good bye pierre he shouted |
| `BWC` | `BWC_arctic_b0309` | 50.0% | nor was elam harnish an exception | nor was eileen is harness an exception |

## Conclusion

Fine-tuning `whisper-medium.en` with LoRA on Chinese-accented English achieved an in-domain test WER of **7.09%**, down from **11.33%** on the zero-shot baseline (a **37.4%** relative WER reduction). Furthermore, the fine-tuned medium model surpassed zero-shot `whisper-large-v3-turbo` (**7.09%** vs **10.93%**).
