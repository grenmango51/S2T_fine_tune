# CMU ARCTIC accent grouping

Speaker groups follow the [official FestVox index](http://festvox.org/cmu_arctic/).
The recorder and downloader use canonical prompts in `cmu_arctic_sources/cmuarctic.data`.

| Training target | Official accent/dialect | Speakers |
| --- | --- | --- |
| `cmu_us` | US English | AEW, BDL, CLB, EEY, LJM, LNH, RMS, SLT |
| `cmu_indian` | Indian English | AUP, AXB, GKA, KSP, SLP |
| `cmu_german` | German-accented English | AHW, FEM |
| `cmu_canadian` | Canadian English | JMK |
| `cmu_scottish` | Scottish English | AWB |
| `cmu_israeli` | Israeli-accented English | RXR |

All 18 speakers occur exactly once. Labels follow the official index; no regional dialect or first-language inference is added. The `us` archive filename prefix describes the English database and is not evidence that every speaker has a US accent.

These targets are separate from all six L2-ARCTIC language groups. Each CMU accent pools every member into one manifest, one LoRA run and one merged model. The Canadian, Scottish and Israeli groups contain only one available speaker each. Evaluation is on held-out prompts spoken by the training speakers; it does not establish generalization to unseen speakers.

Archives are retained under `corpus/cmu_arctic/archives/`. The downloader checks bzip2 integrity, records SHA-256 hashes and source URLs, safely extracts audio and metadata, and uses each archive's own prompt table. If a WAV lacks a prompt (including BDL a0507), the official canonical prompt list supplies it, with each substitution logged. SHA-256 values record the downloaded bytes; they are not publisher-authenticated checksums.

Preparation resamples to 16 kHz and uses the existing seed-42, 90/5/5 sentence-ID split, shared across all speakers within an accent. Every speaker must appear in every split. CMU has no Suitcase split.

Run `python -m s2t.workflows.run_cmu_arctic` to download/verify all archives, prepare each group, train up to 10 epochs with the existing LoRA and early-stopping settings, merge, smoke-test, and evaluate test WER/CER. The runner uses an exclusive lock, resumes training checkpoints, continues other groups after a group failure, and records its final outcome in `runs/cmu_arctic_queue/status.json`. A local desktop notification is attempted on completion or failure; notification delivery errors are recorded in that status file.

Model paths: `models/whisper-medium-en-cmu_<dialect>-accent/`. Runs: `runs/cmu_<dialect>-lora/`. Reports: `reports/finetuned_medium_en_cmu_<dialect>_lora.json`. Nothing is published automatically.

## Evaluation rules (text leakage)

ARCTIC prompts are shared across speakers and accents; each accent group has its own split of the same sentence pool. A model trained on accent A may share prompt text between A's train/val set and B's test set. This is **text leakage**: the model has seen the exact sentence text during training, inflating its apparent accuracy on B's test split.

**Rules**:
1. A fine-tuned model may only be evaluated on its **own accent's test split** or on a test split whose text overlap with the model's train+val text is **zero**.
2. `python -m s2t.evaluation.check_split_overlap` prints the full cross-accent overlap matrix. Guard mode (`python -m s2t.evaluation.check_split_overlap --model_accent cmu_us --eval_accent cmu_german`) exits non-zero if overlap exists.
3. `python -m s2t.evaluation.evaluate_wer --model_accent cmu_us --accent cmu_german` refuses to run when leakage is detected, unless `--allow_leakage` is also passed (which prefixes the report tag with `leaked_`).
4. `s2t.workflows.train_all_accents` always evaluates in-accent, so no change is needed there.

Run `python -m s2t.evaluation.check_split_overlap` to view the current cross-accent overlap matrix.
