# Speech-to-Text Benchmark & Evaluation Report: Spanish Accent

**Date:** 2026-09-17 07:07:26  
**GPU:** NVIDIA GeForce RTX 5070  
**Dataset:** L2-ARCTIC (Spanish) | Train: 3955 utts (4.205 h) | Val: 224 utts (0.251 h) | Test: 223 utts (0.234 h) | Suitcase: 12 chunks (0.078 h)

## Table 1: Model Benchmark Across Splits

| Model Tag | Model Path / Hub ID | Split | WER (%) | CER (%) | Utterances | Decode Time (s) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `finetuned_medium_en_spanish_lora` | `/m/home/home2/21/nguyena42/data/Downloads/Hobbies/S2T fine tune/models/whisper-medium-en-spanish-accent` | `test` | **5.69%** | 2.49% | 223 | 13.19s |
| `finetuned_medium_en_spanish_lora` | `/m/home/home2/21/nguyena42/data/Downloads/Hobbies/S2T fine tune/models/whisper-medium-en-spanish-accent` | `suitcase` | **9.98%** | 5.55% | 12 | 1.29s |
| `zeroshot_large_v3_turbo_spanish` | `openai/whisper-large-v3-turbo` | `test` | 7.85% | 3.64% | 223 | 23.16s |
| `zeroshot_large_v3_turbo_spanish` | `openai/whisper-large-v3-turbo` | `suitcase` | 8.45% | 5.09% | 12 | 1.45s |
| `zeroshot_medium_en_spanish` | `openai/whisper-medium.en` | `test` | 9.61% | 4.47% | 223 | 11.85s |
| `zeroshot_medium_en_spanish` | `openai/whisper-medium.en` | `suitcase` | 8.64% | 4.79% | 12 | 1.1s |

## Table 2: Test Set Per-Speaker and Gender Breakdown (Spanish)

| Model Tag | EBVS (M) | ERMS (M) | MBMPS (F) | NJS (F) | Male Avg | Female Avg | Overall Test WER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `finetuned_medium_en_spanish_lora` | 8.9% | 4.92% | 5.71% | 3.41% | 6.84% | 4.57% | **5.69%** |
| `zeroshot_large_v3_turbo_spanish` | 13.35% | 7.68% | 6.1% | 4.61% | 10.41% | 5.36% | **7.85%** |
| `zeroshot_medium_en_spanish` | 15.68% | 9.25% | 7.09% | 6.81% | 12.35% | 6.95% | **9.61%** |

## Qualitative Examples (Fine-Tuned Model on Spanish Test Set)

### 5 Best Test Examples (Lowest WER)

| Speaker | Utterance | WER | Reference (Normalized) | Hypothesis (Normalized) |
| :--- | :--- | :---: | :--- | :--- |
| `EBVS` | `EBVS_arctic_a0018` | 0.0% | there was a change now | there was a change now |
| `EBVS` | `EBVS_arctic_a0023` | 0.0% | a combination of capital quickly organized and petitioned for the same privileges | a combination of capital quickly organized and petitioned for the same privileges |
| `EBVS` | `EBVS_arctic_a0038` | 0.0% | we will have to watch our chances | we will have to watch our chances |
| `EBVS` | `EBVS_arctic_a0050` | 0.0% | in spite of their absurdity the words affected philip curiously | in spite of their absurdity the words affected philip curiously |
| `EBVS` | `EBVS_arctic_a0108` | 0.0% | he waded into the edge of the water and began scrubbing himself | he waded into the edge of the water and began scrubbing himself |

### 5 Worst Test Examples (Highest WER)

| Speaker | Utterance | WER | Reference (Normalized) | Hypothesis (Normalized) |
| :--- | :--- | :---: | :--- | :--- |
| `MBMPS` | `MBMPS_arctic_b0309` | 66.7% | nor was elam harnish an exception | nor was a lamb harish an exemption |
| `NJS` | `NJS_arctic_a0150` | 50.0% | goodbye pierre he shouted | good bye pierre he shouted |
| `MBMPS` | `MBMPS_arctic_a0158` | 50.0% | does that look good | that looked good |
| `ERMS` | `ERMS_arctic_b0198` | 40.0% | i use great trouble advisedly | i used great trouble advicely |
| `EBVS` | `EBVS_arctic_b0198` | 40.0% | i use great trouble advisedly | i used great trouble admittedly |

## Conclusion

Fine-tuning `whisper-medium.en` with LoRA on Spanish-accented English achieved an in-domain test WER of **5.69%**, down from **9.61%** on the zero-shot baseline (a **40.8%** relative WER reduction). Furthermore, the fine-tuned medium model surpassed zero-shot `whisper-large-v3-turbo` (**5.69%** vs **7.85%**).
