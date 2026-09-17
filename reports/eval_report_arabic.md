# Speech-to-Text Benchmark & Evaluation Report: Arabic Accent

**Date:** 2026-09-17 07:07:25  
**GPU:** NVIDIA GeForce RTX 5070  
**Dataset:** L2-ARCTIC (Arabic) | Train: 3927 utts (3.934 h) | Val: 221 utts (0.238 h) | Test: 217 utts (0.213 h) | Suitcase: 7 chunks (0.044 h)

## Table 1: Model Benchmark Across Splits

| Model Tag | Model Path / Hub ID | Split | WER (%) | CER (%) | Utterances | Decode Time (s) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `finetuned_medium_en_arabic_lora` | `/m/home/home2/21/nguyena42/data/Downloads/Hobbies/S2T fine tune/models/whisper-medium-en-arabic-accent` | `test` | **4.44%** | 2.04% | 217 | 10.68s |
| `finetuned_medium_en_arabic_lora` | `/m/home/home2/21/nguyena42/data/Downloads/Hobbies/S2T fine tune/models/whisper-medium-en-arabic-accent` | `suitcase` | **11.23%** | 7.65% | 7 | 0.81s |
| `zeroshot_large_v3_turbo_arabic` | `openai/whisper-large-v3-turbo` | `test` | 36.34% | 27.08% | 217 | 24.08s |
| `zeroshot_large_v3_turbo_arabic` | `openai/whisper-large-v3-turbo` | `suitcase` | 7.22% | 5.27% | 7 | 0.91s |
| `zeroshot_medium_en_arabic` | `openai/whisper-medium.en` | `test` | 8.21% | 3.95% | 217 | 13.16s |
| `zeroshot_medium_en_arabic` | `openai/whisper-medium.en` | `suitcase` | 6.68% | 4.82% | 7 | 0.95s |

## Table 2: Test Set Per-Speaker and Gender Breakdown (Arabic)

| Model Tag | ABA (M) | SKA (F) | YBAA (M) | ZHAA (F) | Male Avg | Female Avg | Overall Test WER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `finetuned_medium_en_arabic_lora` | 4.85% | 6.9% | 3.61% | 2.76% | 4.23% | 4.67% | **4.44%** |
| `zeroshot_large_v3_turbo_arabic` | 80.2% | 13.56% | 44.69% | 4.92% | 62.37% | 8.91% | **36.34%** |
| `zeroshot_medium_en_arabic` | 8.08% | 14.48% | 4.61% | 6.5% | 6.34% | 10.18% | **8.21%** |

## Qualitative Examples (Fine-Tuned Model on Arabic Test Set)

### 5 Best Test Examples (Lowest WER)

| Speaker | Utterance | WER | Reference (Normalized) | Hypothesis (Normalized) |
| :--- | :--- | :---: | :--- | :--- |
| `ABA` | `ABA_arctic_a0013` | 0.0% | he was a head shorter than his companion of almost delicate physique | he was a head shorter than his companion of almost delicate physique |
| `ABA` | `ABA_arctic_a0018` | 0.0% | there was a change now | there was a change now |
| `ABA` | `ABA_arctic_a0023` | 0.0% | a combination of canadian capital quickly organized and petitioned for the same privileges | a combination of canadian capital quickly organized and petitioned for the same privileges |
| `ABA` | `ABA_arctic_a0034` | 0.0% | men of selden is stamp do not stop at women and children | men of selden is stamp do not stop at women and children |
| `ABA` | `ABA_arctic_a0038` | 0.0% | we will have to watch our chances | we will have to watch our chances |

### 5 Worst Test Examples (Highest WER)

| Speaker | Utterance | WER | Reference (Normalized) | Hypothesis (Normalized) |
| :--- | :--- | :---: | :--- | :--- |
| `ZHAA` | `ZHAA_arctic_a0150` | 50.0% | goodbye pierre he shouted | good bye pierre he shouted |
| `ABA` | `ABA_arctic_b0309` | 50.0% | nor was elam harnish an exception | nor was the lamb harness an exception |
| `SKA` | `SKA_arctic_b0309` | 33.3% | nor was elam harnish an exception | nor was alhamb harness an exception |
| `YBAA` | `YBAA_arctic_a0582` | 30.0% | daughtry elaborated on the counting trick by bringing cocky along | daughtry elaborated on counting tricks by bringing cookie along |
| `SKA` | `SKA_arctic_a0511` | 30.0% | there is more behind this than a mere university ideal | there is no mehind sense than a mere university ideal |

## Conclusion

Fine-tuning `whisper-medium.en` with LoRA on Arabic-accented English achieved an in-domain test WER of **4.44%**, down from **8.21%** on the zero-shot baseline (a **45.9%** relative WER reduction). Furthermore, the fine-tuned medium model surpassed zero-shot `whisper-large-v3-turbo` (**4.44%** vs **36.34%**).
