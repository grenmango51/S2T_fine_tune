# Fine-tune Whisper medium.en (LoRA) for Vietnamese-accented English

> Self-contained execution spec. Any capable coding agent with shell access to this machine can execute it top-to-bottom. Follow the steps in order; each step has **Accept** criteria that must hold before starting the next. Do not modify the original corpus files or the reference scripts.

## Problem
Build a speech-to-text model adapted to English spoken with a Vietnamese accent, trained on the 4 Vietnamese speakers of L2-ARCTIC (~4.8 h), on an 8 GB RTX 4060 Laptop, finishing well within 20 h. Deliver a standalone merged model, a CLI, and a before/after WER report.

## Current state
* Data (already unzipped in the project root): `HQTV/`, `PNV/`, `THV/`, `TLV/` each with `wav/` (1132 files, 44.1 kHz mono 16-bit, ~3.6 s), `transcript/` (one plain-text English sentence per file, cased, no punctuation), `textgrid/`, `annotation/`. All 4 speakers read the same 1132 CMU ARCTIC sentence IDs (`arctic_a0001`..`arctic_b0539`). Total ≈ 4528 utterances ≈ 4.8 h.
* `suitcase_corpus/`: 4 spontaneous recordings (~1.2 min each, >30 s) with `transcript/*.txt` and `annotation/*.TextGrid` (word-level timings).
* `Unused_voices/*.zip`: other L1s, not used.
* `fine_tune_whisper.py`: stock HF Colab script (Hindi / Common Voice / `whisper-small` / batch 16 ≈ 16 GB VRAM). Contains `!pip` and `!nvidia-smi` notebook magics, so it is not runnable Python as-is. Kept as reference only.
* `fine_tune_whisper_guide.md`: spec notes; confirms 8–10 GB class GPUs need small batch + grad accumulation, and PEFT/LoRA for models ≥ medium.
* Machine: Windows, pwsh, Python 3.11.9 (no venv), **no torch/transformers installed**, RTX 4060 Laptop 8 GB (driver 610.78), 16 GB RAM, 556 GB free on D:.

## Decisions (settled with the user — do not revisit)
* Task: Vietnamese-accented **English** transcription (data is English).
* Base model: `openai/whisper-medium.en` from the Hugging Face Hub (public, auto-downloaded by `from_pretrained`, no login). Method: LoRA via `peft`; full FT of medium does not fit in 8 GB.
* Split: by sentence ID, seeded shuffle (seed 42), 90/5/5 → 1018/57/57 sentence IDs → 4072/228/228 utterances; all 4 speakers in every split; assignment written to `data/manifest.csv`. Suitcase recordings chunked into ≤30 s segments at word boundaries (from TextGrid) as a separate out-of-domain test set, never trained on.
* Labels: transcripts as-is (unpunctuated). WER computed after Whisper's English text normalizer on both hypothesis and reference.
* Storage: local only under the project folder. No Hub push.
* Laptop left alone while training; tune batch size for the full 8 GB.
* Deliverables: merged standalone model, `transcribe.py` CLI, `reports/eval_report.md` with zero-shot `medium.en`, zero-shot `large-v3-turbo`, and fine-tuned `medium.en` WER on test and suitcase sets.

## Executor notes (read first)
* Everything runs **locally on this laptop** (the GPU is here); work in the project root `D:\Hoai Anh\Aalto\Hobbies\S2T fine tune`, shell is pwsh 5.1. Do not modify `fine_tune_whisper.py`, `fine_tune_whisper_guide.md`, `README*`, `PROMPTS`, `LICENSE`, or anything inside the speaker folders.
* Package versions at plan time (2026-09): torch 2.14.0 (cu126/cu130 Windows cp311 wheels exist; cu128 tops out at 2.11), transformers 5.17.0, peft 0.21.0, accelerate 1.15.0, jiwer 4.0.0, soundfile 0.14.0, praatio 6.2.2. Install latest stable; treat the **API-drift checklist** below as required reading; after step 1 passes, write `pip freeze` to `requirements.lock.txt`.
* Do not use the `datasets` library or `evaluate`. Audio I/O is `soundfile`, resampling is `soxr`, WER is `jiwer`. No FFmpeg anywhere.
* Every script has a `main()` under `if __name__ == "__main__":`, uses `argparse`, and `dataloader_num_workers=0` (Windows).
* Each step below ends with **Accept** criteria. Do not start the next step until they hold. If a criterion cannot be met after two honest attempts, consult the stop-and-ask list.

## File contracts (new files, project root)
* `requirements.txt` — `transformers peft accelerate jiwer soundfile soxr praatio tensorboard numpy pandas tqdm` (torch installed separately from the PyTorch index).
* `common.py`
    * `PROJECT_ROOT`, `DATA_DIR = data/`, `MANIFEST = data/manifest.csv`, `SPEAKERS = ["HQTV","PNV","THV","TLV"]`, `BASE_MODEL = "openai/whisper-medium.en"`, `SAMPLE_RATE = 16000`, `SEED = 42`.
    * `load_manifest(split: str|None) -> pandas.DataFrame`.
    * `load_audio_16k(path) -> np.ndarray float32` (soundfile read, mono mixdown, `soxr.resample` if rate != 16000).
    * `normalize_text(text, tokenizer) -> str` using the Whisper English normalizer (see drift checklist), plus `corpus_wer(refs, hyps) -> float` and `corpus_cer` via jiwer over the normalized strings; empty normalized hyp is replaced by a single placeholder token so jiwer does not crash.
    * `WhisperSpeechDataset(df, feature_extractor, tokenizer)` — `__getitem__` returns `{"input_features": (80,3000) float32 array, "labels": list[int]}`; labels are `tokenizer(text).input_ids` (tokenizer already configured with `task="transcribe"`, no language for `.en` models).
    * `DataCollatorSpeechSeq2SeqWithPadding` — identical logic to the one in `fine_tune_whisper.py` lines 332–358 (pad features, pad labels with -100, strip leading decoder-start token).
* `prepare_data.py` — writes `data/wav16k/<SPK>/<id>.wav` (16 kHz mono PCM16), `data/wav16k/suitcase/<spk>_<k>.wav`, `data/manifest.csv`, `data/split_ids.json`; prints a summary table. Idempotent (skips existing wavs unless `--force`).
* `train_lora.py` — flags: `--batch_size 8 --grad_accum 2 --epochs 10 --lr 2e-4 --lora_r 32 --lora_alpha 64 --no_spec_augment --pilot --resume --precision {fp16,bf16} --output runs/medium-en-lora`. Writes checkpoints under `--output`, `best_adapter/` (PEFT adapter + `adapter_config.json`), `train_summary.json` (steps, wall time, best val WER, epoch of best, peak VRAM, chosen batch config).
* `merge_lora.py --adapter runs/medium-en-lora/best_adapter --out models/whisper-medium-en-vi-accent` — saves merged fp16 safetensors + processor + `generation_config.json`; then reloads the saved dir once and transcribes one train wav as a smoke test.
* `evaluate_wer.py --model <hub id | local dir> --splits test,suitcase [--batch_size 16] [--tag NAME]` — writes `reports/<tag>.json` and regenerates `reports/eval_report.md` from all JSONs present.
* `transcribe.py <audio_path> [--model models/whisper-medium-en-vi-accent]` — prints plain text; long audio via `pipeline(..., chunk_length_s=30, batch_size=8)`.

## Data contract
* Utterance pairing: `<SPK>/wav/<id>.wav` ↔ `<SPK>/transcript/<id>.txt`; text = file content stripped with whitespace collapsed. Skip and log any pair with missing file or empty text (expect 0 skips).
* `data/manifest.csv` columns: `utt_id` (`<SPK>_<id>`), `speaker`, `sentence_id`, `path` (relative, forward slashes, points to the 16 kHz copy), `text`, `split` (`train|val|test|suitcase`), `duration_s`.
* Split algorithm: `ids = sorted(set(sentence_id))` (expect 1132); `random.Random(42).shuffle(ids)`; `n_test = n_val = round(0.05*len(ids))` (=57); `test = ids[:57]`, `val = ids[57:114]`, `train = ids[114:]` (=1018). Persist to `data/split_ids.json`. All 4 speakers' utterances of a sentence id share its split.
* Suitcase chunking: for each `suitcase_corpus/annotation/<spk>.TextGrid`, read the `words` tier with praatio; keep intervals whose label is non-empty and not in `{"sp","sil","spn",""}`; greedily group consecutive words so a chunk spans ≤ 28.0 s from its first word start; cut points are midpoints between the last kept word's end and the next word's start; chunk text = space-joined word labels; write chunk audio from the 16 kHz resampled recording. Sanity: `corpus_wer(transcript_txt, join(all chunk texts))` per speaker must be < 10% (labels may contain fillers/tags; report the number).
* **Accept (step 2):** manifest has 4528 rows across the 4 speakers (1132 each) + ≥ 8 suitcase rows; per split ≈ 4072 / 228 / 228 utterances; each of train/val/test contains all 4 speakers with equal counts; total train+val+test duration 4.5–5.2 h; every `path` exists; 3 random rows print text that matches the original transcript file.

## Hyperparameters (exact)
* Base: `openai/whisper-medium.en`, loaded in fp16 with `attn_implementation="sdpa"`; `model.config.use_cache=False` during training; `model.generation_config.forced_decoder_ids=None`; do not set `language` (English-only checkpoint).
* SpecAugment (default on): `config.apply_spec_augment=True, mask_time_prob=0.05, mask_time_length=10, mask_feature_prob=0.05, mask_feature_length=10`.
* Gradient checkpointing on with `gradient_checkpointing_kwargs={"use_reentrant": False}`; call `model.enable_input_require_grads()` before wrapping with PEFT.
* LoRA: `LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, bias="none", target_modules=["q_proj","k_proj","v_proj","out_proj","fc1","fc2"])`, **no `task_type`** (Whisper takes `input_features`, not `input_ids`). Adapters stay fp32 (peft default). Print `print_trainable_parameters()`; expect roughly 1.5–3.5% trainable.
* `Seq2SeqTrainingArguments`: `per_device_train_batch_size=8, gradient_accumulation_steps=2` (effective 16; pilot may change), `per_device_eval_batch_size=16, learning_rate=2e-4, weight_decay=0.01, warmup_steps=100, lr_scheduler_type="linear", max_grad_norm=1.0, num_train_epochs=10, fp16=True, eval_strategy="epoch", save_strategy="epoch", save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="wer", greater_is_better=False, predict_with_generate=True, generation_max_length=128, generation_num_beams=1, logging_steps=25, report_to=["tensorboard"], remove_unused_columns=False, label_names=["labels"], dataloader_num_workers=0, seed=42`. Callbacks: `EarlyStoppingCallback(early_stopping_patience=3)` plus a small callback logging `torch.cuda.max_memory_allocated()/2**30` at each log step.
* `compute_metrics`: decode with `skip_special_tokens=True`, normalize both sides, return `{"wer": 100*corpus_wer}`.
* Steps per epoch ≈ 4072/16 ≈ 255; 10 epochs ≈ 2550 optimizer steps max.

## Decision tables
* Pilot (`--pilot`: 30 optimizer steps + eval on the first 64 val utterances, then exit and print `sec_per_step`, `peak_vram_gb`, `eval_sec_per_utt`):
    * OOM at 8×2 → retry 4×4; OOM at 4×4 → retry 2×8; OOM at 2×8 → stop-and-ask.
    * peak < 5.5 GB at 8×2 → try 16×1 once; keep it only if peak < 7.2 GB and sec/step improves ≥ 10%.
    * projected_hours = (255 × epochs × sec_per_step + epochs × 228 × eval_sec_per_utt) / 3600. If projected_hours(10) > 14 → set `epochs = max(4, floor(12h budget))`. If projected_hours(4) > 14 → stop-and-ask.
* Training health (check after each epoch from `trainer_state.json` / TensorBoard):
    * Train loss not below 90% of its initial value after epoch 1 → lr may be too low; continue but flag in summary.
    * Val WER after epoch 1 higher than zero-shot medium.en test WER by > 2 abs points, or loss NaN/inf → kill, restart with `--lr 1e-4 --lora_r 16`; if it happens again → stop-and-ask.
    * Early stopping fires → normal; proceed.
    * Process died (power, crash) → rerun the identical command with `--resume`.
* Precision: if a `ValueError: Attempting to unscale FP16 gradients` or similar occurs → rerun with `--precision bf16` (base loaded in bf16, `bf16=True`). Ada GPUs support bf16.
* Eval sanity: zero-shot medium.en test WER expected roughly 5–15%. If < 2% or > 40% → suspect normalization/label bug; inspect 10 (ref, hyp) pairs before continuing.

## Step-by-step execution with acceptance criteria
1. **Environment.** `python -m venv .venv`; activate; `python -m pip install -U pip`; `pip install torch --index-url https://download.pytorch.org/whl/cu130` (fallback `cu126` if CUDA is unavailable at runtime); `pip install -r requirements.txt`. **Accept:** `python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"` prints `True` and a name containing `4060`; `import transformers, peft, jiwer, soundfile, soxr, praatio` succeeds; write `requirements.lock.txt`.
2. **Data prep.** Write `common.py`, `prepare_data.py`; run it. **Accept:** see Data contract.
3. **Baselines.** Write `evaluate_wer.py`; run with `--model openai/whisper-medium.en --tag zeroshot_medium_en` and `--model openai/whisper-large-v3-turbo --tag zeroshot_large_v3_turbo --batch_size 8` (turbo is multilingual: pass `language="en", task="transcribe"` to generate; use 128 mel bins automatically via its own processor). **Accept:** two JSONs in `reports/`, `eval_report.md` renders a table with WER/CER per split per model, medium.en test WER within the sanity band above, suitcase WER reported.
4. **Power settings + pilot.** Record `powercfg /q SCHEME_CURRENT SUB_SLEEP` output to `reports/powercfg_before.txt`; then `powercfg /change standby-timeout-ac 0` and `powercfg /change hibernate-timeout-ac 0`. Write `train_lora.py`; run `--pilot`. **Accept:** pilot completes, prints sec/step, peak VRAM, eval sec/utt, and the chosen `(batch_size, grad_accum, epochs)` per the decision table; a val WER number is produced (any value).
5. **Full training.** Run `train_lora.py` with the chosen config (long-running; launch in a way that survives the agent session, e.g. `Start-Process` with stdout/stderr redirected to `runs/medium-en-lora/train.log`, and poll the log / `trainer_state.json` every ~15 min). **Accept:** `best_adapter/` and `train_summary.json` exist; best val WER < zero-shot medium.en val-equivalent (use test WER from step 3 as reference) — if not, see stop-and-ask.
6. **Merge + final eval.** Run `merge_lora.py`; run `evaluate_wer.py --model models/whisper-medium-en-vi-accent --tag finetuned_medium_en_lora`. **Accept:** merged dir loads via `WhisperForConditionalGeneration.from_pretrained`; merged test WER is within 0.5 abs points of the best checkpoint's val WER trend (i.e. no merge/dtype mistake); `eval_report.md` now lists three models.
7. **CLI + wrap-up.** Write `transcribe.py`; run it on one test wav and one suitcase chunk. Restore power settings to the recorded values. **Accept:** both transcriptions are sensible; final `eval_report.md` includes the results table, per-speaker WER for the fine-tuned model on test, 5 worst and 5 best test examples, the chosen training config, wall-clock time, and peak VRAM.

## API-drift checklist (transformers 5.x / peft 0.2x) — items marked VERIFIED were checked against transformers `main` source/docs on 2026-09-15
* VERIFIED: `Seq2SeqTrainer` still exists and takes `processing_class=` (pass `processor.feature_extractor`); `Seq2SeqTrainingArguments` still has `predict_with_generate`, `generation_max_length`, `generation_num_beams`, `generation_config`. Use `eval_strategy` (not `evaluation_strategy`).
* VERIFIED: model load kwarg is `dtype=torch.float16` in v5 (`torch_dtype=` is the legacy name).
* VERIFIED: `WhisperConfig` still has `apply_spec_augment, mask_time_prob, mask_time_length, mask_time_min_masks, mask_feature_prob, mask_feature_length, mask_feature_min_masks`. Note the config class is a strict dataclass in v5; set attributes after loading (`model.config.apply_spec_augment = True`), do not pass unknown kwargs.
* VERIFIED: `WhisperTokenizer.normalize(text)` is public API (uses `EnglishTextNormalizer` + the checkpoint's `normalizer.json`). **Gotcha (open transformers issue #48142 / PR #48143, Aug 2026):** in v5 `save_pretrained` does not write `normalizer.json`, so `normalize()` raises `AttributeError` on a tokenizer reloaded from a local dir. Mitigation (required): in `common.py`, always build the normalizer from the hub tokenizer `WhisperTokenizer.from_pretrained("openai/whisper-medium.en").normalize`, never from `models/...`; additionally `merge_lora.py` copies `normalizer.json` from the base checkpoint's HF cache into the merged output dir so the merged model is self-contained.
* VERIFIED: jiwer API is `jiwer.wer(refs, hyps)` / `jiwer.cer(refs, hyps)` accepting lists of strings; `jiwer.process_words(...)` gives alignments for the worst/best-example listing.
* VERIFIED: praatio usage is `from praatio import textgrid; tg = textgrid.openTextgrid(path, includeEmptyIntervals=False)`; the tier is `tg.getTier("words")` (or `tg.tierDict["words"]`), whose `.entries` are `(start, end, label)` tuples.
* VERIFIED: v5 removed slow tokenizers; `WhisperTokenizer` is the fast tokenizer backend. `AutoProcessor`/`WhisperProcessor.from_pretrained` unchanged.
* If `Trainer` refuses `predict_with_generate` with a `PeftModel`, wrap generation in `compute_metrics`-side manual loop as fallback (`trainer.model.generate(input_features=...)`).
* `WhisperForConditionalGeneration.generate` for `.en` models: do not pass `language`/`task`; for `large-v3-turbo` pass `language="en", task="transcribe"`.
* peft (not re-verified against 0.21 source; high confidence, adapt on error): `get_peft_model(model, LoraConfig(...))`, `PeftModel.from_pretrained(base, adapter_dir)`, `merged = peft_model.merge_and_unload()`, `merged.save_pretrained(out, safe_serialization=True)`; `processor.save_pretrained(out)`; `merged.generation_config.save_pretrained(out)`; then copy `normalizer.json` (see gotcha above).
* On any `TypeError`/`AttributeError` from these libraries: read `inspect.signature` / `help()` of the offending symbol and adapt the call; do **not** downgrade packages blindly. Pin only after a passing smoke test.

## Stop-and-ask list (pause and report to the user instead of improvising)
* CUDA not available after trying both cu130 and cu126 wheels.
* Pilot OOMs even at 2×8, or projected time for 4 epochs exceeds 14 h.
* Zero-shot medium.en test WER outside 2–40% after checking 10 examples (likely data/normalization bug).
* Suitcase chunk-text vs transcript WER ≥ 10% for any speaker.
* Training restarted once with the fallback lr/rank and still diverges or does not beat zero-shot after 3 epochs.
* Fine-tuned test WER is not lower than zero-shot medium.en (report the numbers; do not iterate on hyperparameters without approval).
* Any step would require deleting or altering files outside `data/`, `runs/`, `models/`, `reports/`, `.venv/`, or the new scripts.

## eval_report.md format
* Header: date, GPU, package versions (from lock file), split sizes and hours.
* Table 1: model × split (test, suitcase) → WER %, CER %, n_utts, decode time.
* Table 2: fine-tuned model per-speaker WER on test (HQTV, PNV, THV, TLV) and by gender (M: HQTV+TLV, F: PNV+THV).
* Training config + `train_summary.json` contents; 5 worst / 5 best test examples for the fine-tuned model (ref vs hyp, normalized).
* One-paragraph conclusion: relative WER reduction vs zero-shot medium.en; whether it beats zero-shot large-v3-turbo. Deferred benchmarks are listed in `future_benchmark.md`.

## Risks / notes
* VRAM: medium.en fp16 weights ≈ 1.5 GB; the rest is activations. The pilot sets the batch config before any long run.
* Overfitting on 4.8 h: LoRA + early stopping on val WER + SpecAugment; fallback lr 1e-4 / r 16 per the decision table.
* Zero-shot `large-v3-turbo` may already beat fine-tuned `medium.en`; the report makes that visible for a possible follow-up (LoRA on turbo).
* Success criterion: fine-tuned test WER clearly below zero-shot `medium.en` (target ≥ 25% relative reduction), no regression on the suitcase set.
