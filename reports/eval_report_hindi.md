# Speech-to-Text Benchmark & Evaluation Report: Hindi Accent

**Date:** 2026-09-17 07:07:26  
**GPU:** NVIDIA GeForce RTX 5070  
**Dataset:** L2-ARCTIC (Hindi) | Train: 4071 utts (3.473 h) | Val: 228 utts (0.203 h) | Test: 225 utts (0.188 h) | Suitcase: 7 chunks (0.044 h)

## Table 1: Model Benchmark Across Splits

| Model Tag | Model Path / Hub ID | Split | WER (%) | CER (%) | Utterances | Decode Time (s) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `finetuned_medium_en_hindi_lora` | `/m/home/home2/21/nguyena42/data/Downloads/Hobbies/S2T fine tune/models/whisper-medium-en-hindi-accent` | `test` | **2.64%** | 1.3% | 225 | 11.69s |
| `finetuned_medium_en_hindi_lora` | `/m/home/home2/21/nguyena42/data/Downloads/Hobbies/S2T fine tune/models/whisper-medium-en-hindi-accent` | `suitcase` | **5.17%** | 3.4% | 7 | 0.83s |
| `zeroshot_large_v3_turbo_hindi` | `openai/whisper-large-v3-turbo` | `test` | 3.94% | 2.07% | 225 | 23.28s |
| `zeroshot_large_v3_turbo_hindi` | `openai/whisper-large-v3-turbo` | `suitcase` | 3.95% | 1.93% | 7 | 0.91s |
| `zeroshot_medium_en_hindi` | `openai/whisper-medium.en` | `test` | 4.54% | 2.07% | 225 | 14.26s |
| `zeroshot_medium_en_hindi` | `openai/whisper-medium.en` | `suitcase` | 3.65% | 1.76% | 7 | 0.96s |

## Table 2: Test Set Per-Speaker and Gender Breakdown (Hindi)

| Model Tag | ASI (M) | RRBI (M) | SVBI (F) | TNI (F) | Male Avg | Female Avg | Overall Test WER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `finetuned_medium_en_hindi_lora` | 1.8% | 2.81% | 2.56% | 3.41% | 2.3% | 2.98% | **2.64%** |
| `zeroshot_large_v3_turbo_hindi` | 4.41% | 3.41% | 3.54% | 4.42% | 3.91% | 3.98% | **3.94%** |
| `zeroshot_medium_en_hindi` | 3.81% | 4.61% | 4.13% | 5.62% | 4.21% | 4.87% | **4.54%** |

## Qualitative Examples (Fine-Tuned Model on Hindi Test Set)

### 5 Best Test Examples (Lowest WER)

| Speaker | Utterance | WER | Reference (Normalized) | Hypothesis (Normalized) |
| :--- | :--- | :---: | :--- | :--- |
| `ASI` | `ASI_arctic_a0013` | 0.0% | he was a head shorter than his companion of almost delicate physique | he was a head shorter than his companion of almost delicate physique |
| `ASI` | `ASI_arctic_a0018` | 0.0% | there was a change now | there was a change now |
| `ASI` | `ASI_arctic_a0023` | 0.0% | a combination of canadian capital quickly organized and petitioned for the same privileges | a combination of canadian capital quickly organized and petitioned for the same privileges |
| `ASI` | `ASI_arctic_a0034` | 0.0% | men of selden is stamp do not stop at women and children | men of selden is stamp do not stop at women and children |
| `ASI` | `ASI_arctic_a0038` | 0.0% | we will have to watch our chances | we will have to watch our chances |

### 5 Worst Test Examples (Highest WER)

| Speaker | Utterance | WER | Reference (Normalized) | Hypothesis (Normalized) |
| :--- | :--- | :---: | :--- | :--- |
| `TNI` | `TNI_arctic_b0356` | 33.3% | he loved to play chinese lottery | he loved to play chinese law trick |
| `ASI` | `ASI_arctic_b0309` | 33.3% | nor was elam harnish an exception | nor was elam harness an expectation |
| `SVBI` | `SVBI_arctic_a0582` | 30.0% | daughtry elaborated on the counting trick by bringing cocky along | daughter he elaborated on the counting trick by bringing cockey along |
| `ASI` | `ASI_arctic_b0330` | 28.6% | they only had a little $30000 fire | they only had a little thirsty $8000 fire |
| `SVBI` | `SVBI_arctic_b0445` | 22.2% | mops sir eagerly answered the sailor at the wheel | mop is sir eagerly answered the sailor at the wheel |

## Conclusion

Fine-tuning `whisper-medium.en` with LoRA on Hindi-accented English achieved an in-domain test WER of **2.64%**, down from **4.54%** on the zero-shot baseline (a **41.9%** relative WER reduction). Furthermore, the fine-tuned medium model surpassed zero-shot `whisper-large-v3-turbo` (**2.64%** vs **3.94%**).
