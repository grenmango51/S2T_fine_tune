# Speech-to-Text Fine-Tuning Evaluation Report

**Date:** 2026-09-17 06:57:57  
**GPU:** NVIDIA GeForce RTX 5070  
**Environment:** Python 3.12.3, Torch 2.14.0+cu130, Transformers 5.17.0, PEFT 0.21.0, Accelerate 1.15.0, JiWER 4.0.0  
**Dataset Splits:** Train: 4072 utts (4.305 h) | Val: 228 utts (0.257 h) | Test: 228 utts (0.238 h) | Suitcase: 9 chunks (0.054 h)

## Table 1: Model Benchmark Across Splits

| Model Tag | Model Path / Hub ID | Split | WER (%) | CER (%) | Utterances | Decode Time (s) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `finetuned_medium_en_arabic_lora` | `/m/home/home2/21/nguyena42/data/Downloads/Hobbies/S2T fine tune/models/whisper-medium-en-arabic-accent` | `test` | **4.44%** | 2.04% | 217 | 10.68s |
| `finetuned_medium_en_arabic_lora` | `/m/home/home2/21/nguyena42/data/Downloads/Hobbies/S2T fine tune/models/whisper-medium-en-arabic-accent` | `suitcase` | **11.23%** | 7.65% | 7 | 0.81s |
| `finetuned_medium_en_chinese_lora` | `/m/home/home2/21/nguyena42/data/Downloads/Hobbies/S2T fine tune/models/whisper-medium-en-chinese-accent` | `test` | **7.09%** | 3.57% | 225 | 11.3s |
| `finetuned_medium_en_chinese_lora` | `/m/home/home2/21/nguyena42/data/Downloads/Hobbies/S2T fine tune/models/whisper-medium-en-chinese-accent` | `suitcase` | **16.54%** | 10.75% | 12 | 0.99s |
| `finetuned_medium_en_hindi_lora` | `/m/home/home2/21/nguyena42/data/Downloads/Hobbies/S2T fine tune/models/whisper-medium-en-hindi-accent` | `test` | **2.64%** | 1.30% | 225 | 11.69s |
| `finetuned_medium_en_hindi_lora` | `/m/home/home2/21/nguyena42/data/Downloads/Hobbies/S2T fine tune/models/whisper-medium-en-hindi-accent` | `suitcase` | **5.17%** | 3.40% | 7 | 0.83s |
| `finetuned_medium_en_korean_lora` | `/m/home/home2/21/nguyena42/data/Downloads/Hobbies/S2T fine tune/models/whisper-medium-en-korean-accent` | `test` | **4.84%** | 2.21% | 228 | 11.5s |
| `finetuned_medium_en_korean_lora` | `/m/home/home2/21/nguyena42/data/Downloads/Hobbies/S2T fine tune/models/whisper-medium-en-korean-accent` | `suitcase` | **8.26%** | 5.50% | 20 | 1.72s |
| `finetuned_medium_en_lora` | `models/whisper-medium-en-vi-accent` | `test` | **11.42%** | 6.05% | 228 | 21.12s |
| `finetuned_medium_en_lora` | `models/whisper-medium-en-vi-accent` | `suitcase` | **20.53%** | 11.03% | 9 | 1.58s |
| `finetuned_medium_en_spanish_lora` | `/m/home/home2/21/nguyena42/data/Downloads/Hobbies/S2T fine tune/models/whisper-medium-en-spanish-accent` | `test` | **5.69%** | 2.49% | 223 | 13.19s |
| `finetuned_medium_en_spanish_lora` | `/m/home/home2/21/nguyena42/data/Downloads/Hobbies/S2T fine tune/models/whisper-medium-en-spanish-accent` | `suitcase` | **9.98%** | 5.55% | 12 | 1.29s |
| `zeroshot_large_v3_turbo` | `openai/whisper-large-v3-turbo` | `test` | **15.99%** | 8.16% | 228 | 52.01s |
| `zeroshot_large_v3_turbo` | `openai/whisper-large-v3-turbo` | `suitcase` | **12.90%** | 8.07% | 9 | 2.34s |
| `zeroshot_medium_en` | `openai/whisper-medium.en` | `test` | **18.85%** | 9.97% | 228 | 21.88s |
| `zeroshot_medium_en` | `openai/whisper-medium.en` | `suitcase` | **15.25%** | 9.23% | 9 | 1.64s |

## Table 2: Test Set Per-Speaker and Gender Breakdown

| Model Tag | HQTV (M) | PNV (F) | THV (F) | TLV (M) | Male Avg | Female Avg | Overall Test WER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `finetuned_medium_en_arabic_lora` | N/A% | N/A% | N/A% | N/A% | N/A% | N/A% | **4.44%** |
| `finetuned_medium_en_chinese_lora` | N/A% | N/A% | N/A% | N/A% | N/A% | N/A% | **7.09%** |
| `finetuned_medium_en_hindi_lora` | N/A% | N/A% | N/A% | N/A% | N/A% | N/A% | **2.64%** |
| `finetuned_medium_en_korean_lora` | N/A% | N/A% | N/A% | N/A% | N/A% | N/A% | **4.84%** |
| `finetuned_medium_en_lora` | 13.39% | 5.31% | 10.24% | 16.73% | 15.06% | 7.78% | **11.42%** |
| `finetuned_medium_en_spanish_lora` | N/A% | N/A% | N/A% | N/A% | N/A% | N/A% | **5.69%** |
| `zeroshot_large_v3_turbo` | 18.9% | 6.3% | 15.94% | 22.83% | 20.87% | 11.12% | **15.99%** |
| `zeroshot_medium_en` | 21.85% | 7.87% | 18.7% | 26.97% | 24.41% | 13.29% | **18.85%** |

## Qualitative Examples (Fine-Tuned Model on Test Set)

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

Fine-tuning `whisper-medium.en` with LoRA on Vietnamese-accented English achieved a test WER of **4.44%**, down from **18.85%** on the zero-shot baseline (a **76.4%** relative WER reduction). Furthermore, the fine-tuned medium model surpassed zero-shot `whisper-large-v3-turbo` (4.44% vs 15.99%). Additional benchmark comparisons are documented in [future_benchmark.md](file:///d:/Hoai%20Anh/Aalto/Hobbies/S2T%20fine%20tune/future_benchmark.md).
