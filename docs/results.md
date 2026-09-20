# Evaluation results

Recorded benchmark measurements as of 2026-09-20. WER (word error rate) and CER
(character error rate) are percentages; lower is better. Different report tags
and splits can use different speakers, samples, and decoding settings. Compare
matched evaluations only; these measurements are not independently validated.

`test` denotes held-out evaluation data, `val` validation data, and `suitcase`
spontaneous speech. Report tags identify the model and experiment. Evaluation
commands generate detailed `<tag>.json` files locally; those files and the
underlying recordings are not included in this source repository.

| Report tag | Split | Utterances | WER (%) | CER (%) |
| --- | --- | ---: | ---: | ---: |
| `diagnostic_finetuned_medium_en_cmu_indian_val` | val | 177 | 2.97 | 0.98 |
| `diagnostic_zeroshot_medium_en_cmu_indian_val` | val | 177 | 4.02 | 1.44 |
| `finetuned_medium_en_arabic_lora` | test | 217 | 4.44 | 2.04 |
| `finetuned_medium_en_arabic_lora` | suitcase | 7 | 11.23 | 7.65 |
| `finetuned_medium_en_bdl_lora` | test | 57 | 1.38 | 0.27 |
| `finetuned_medium_en_chinese_lora` | test | 225 | 7.09 | 3.57 |
| `finetuned_medium_en_chinese_lora` | suitcase | 12 | 16.54 | 10.75 |
| `finetuned_medium_en_cmu_canadian_lora` | test | 57 | 3.54 | 3.13 |
| `finetuned_medium_en_cmu_german_lora` | test | 60 | 1.72 | 0.83 |
| `finetuned_medium_en_cmu_indian_lora` | test | 185 | 2.94 | 0.81 |
| `finetuned_medium_en_cmu_israeli_lora` | test | 33 | 3.65 | 1.77 |
| `finetuned_medium_en_cmu_scottish_lora` | test | 57 | 2.19 | 0.89 |
| `finetuned_medium_en_cmu_us_lora` | test | 406 | 1.21 | 0.34 |
| `finetuned_medium_en_hindi_lora` | test | 225 | 2.64 | 1.3 |
| `finetuned_medium_en_hindi_lora` | suitcase | 7 | 5.17 | 3.4 |
| `finetuned_medium_en_korean_lora` | test | 228 | 4.84 | 2.21 |
| `finetuned_medium_en_korean_lora` | suitcase | 20 | 8.26 | 5.5 |
| `finetuned_medium_en_lora` | test | 228 | 11.42 | 6.05 |
| `finetuned_medium_en_lora` | suitcase | 9 | 20.53 | 11.03 |
| `finetuned_medium_en_personal` | test | 57 | 3.73 | 1.49 |
| `finetuned_medium_en_slt_lora` | test | 57 | 1.38 | 0.53 |
| `finetuned_medium_en_spanish_lora` | test | 223 | 5.69 | 2.49 |
| `finetuned_medium_en_spanish_lora` | suitcase | 12 | 9.98 | 5.55 |
| `finetuned_medium_en_vi_hqtv_personalized` | test | 57 | 9.84 | 5.27 |
| `finetuned_medium_en_vi_hqtv_personalized` | suitcase | 1 | 12.0 | 8.1 |
| `ood_librispeech_cmu_canadian` | test | 2620 | 2.66 | 1.05 |
| `ood_librispeech_cmu_german` | test | 2620 | 2.6 | 1.01 |
| `ood_librispeech_cmu_indian` | test | 2620 | 2.68 | 0.93 |
| `ood_librispeech_cmu_israeli` | test | 2620 | 2.64 | 0.97 |
| `ood_librispeech_cmu_scottish` | test | 2620 | 3.4 | 1.73 |
| `ood_librispeech_cmu_us` | test | 2620 | 3.42 | 1.45 |
| `ood_librispeech_zeroshot_medium_en` | test | 2620 | 3.02 | 1.41 |
| `zeroshot_large_v3_turbo` | test | 228 | 15.99 | 8.16 |
| `zeroshot_large_v3_turbo` | suitcase | 9 | 12.9 | 8.07 |
| `zeroshot_large_v3_turbo_arabic` | test | 217 | 36.34 | 27.08 |
| `zeroshot_large_v3_turbo_arabic` | suitcase | 7 | 7.22 | 5.27 |
| `zeroshot_large_v3_turbo_chinese` | test | 225 | 10.93 | 5.45 |
| `zeroshot_large_v3_turbo_chinese` | suitcase | 12 | 10.77 | 6.89 |
| `zeroshot_large_v3_turbo_cmu_canadian` | test | 57 | 2.75 | 3.13 |
| `zeroshot_large_v3_turbo_cmu_german` | test | 60 | 2.68 | 0.91 |
| `zeroshot_large_v3_turbo_cmu_indian` | test | 185 | 2.52 | 1.21 |
| `zeroshot_large_v3_turbo_cmu_israeli` | test | 33 | 2.33 | 0.85 |
| `zeroshot_large_v3_turbo_cmu_scottish` | test | 57 | 2.39 | 0.93 |
| `zeroshot_large_v3_turbo_cmu_us` | test | 406 | 1.02 | 0.42 |
| `zeroshot_large_v3_turbo_hindi` | test | 225 | 3.94 | 2.07 |
| `zeroshot_large_v3_turbo_hindi` | suitcase | 7 | 3.95 | 1.93 |
| `zeroshot_large_v3_turbo_korean` | test | 228 | 6.26 | 2.84 |
| `zeroshot_large_v3_turbo_korean` | suitcase | 20 | 6.9 | 4.56 |
| `zeroshot_large_v3_turbo_personal` | test | 57 | 5.11 | 2.82 |
| `zeroshot_large_v3_turbo_spanish` | test | 223 | 7.85 | 3.64 |
| `zeroshot_large_v3_turbo_spanish` | suitcase | 12 | 8.45 | 5.09 |
| `zeroshot_medium_en` | test | 228 | 18.85 | 9.97 |
| `zeroshot_medium_en` | suitcase | 9 | 15.25 | 9.23 |
| `zeroshot_medium_en_arabic` | test | 217 | 8.21 | 3.95 |
| `zeroshot_medium_en_arabic` | suitcase | 7 | 6.68 | 4.82 |
| `zeroshot_medium_en_chinese` | test | 225 | 11.33 | 5.65 |
| `zeroshot_medium_en_chinese` | suitcase | 12 | 12.69 | 9.11 |
| `zeroshot_medium_en_cmu_canadian` | test | 57 | 3.54 | 3.24 |
| `zeroshot_medium_en_cmu_german` | test | 60 | 2.68 | 1.16 |
| `zeroshot_medium_en_cmu_indian` | test | 185 | 2.52 | 0.97 |
| `zeroshot_medium_en_cmu_israeli` | test | 33 | 2.66 | 1.31 |
| `zeroshot_medium_en_cmu_scottish` | test | 57 | 2.39 | 1.07 |
| `zeroshot_medium_en_cmu_us` | test | 406 | 1.1 | 0.34 |
| `zeroshot_medium_en_hindi` | test | 225 | 4.54 | 2.07 |
| `zeroshot_medium_en_hindi` | suitcase | 7 | 3.65 | 1.76 |
| `zeroshot_medium_en_korean` | test | 228 | 7.39 | 3.5 |
| `zeroshot_medium_en_korean` | suitcase | 20 | 8.48 | 6.12 |
| `zeroshot_medium_en_personal` | test | 57 | 5.7 | 2.59 |
| `zeroshot_medium_en_spanish` | test | 223 | 9.61 | 4.47 |
| `zeroshot_medium_en_spanish` | suitcase | 12 | 8.64 | 4.79 |
