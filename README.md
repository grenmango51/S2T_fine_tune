# Whisper speech-to-text fine-tuning

Adapt Whisper to accented English and individual speakers using LoRA fine-tuning.
This project includes dataset preparation, training, adapter merging, transcription,
and WER/CER evaluation, plus a browser recorder and local/cloud benchmark app.

The main training model is `openai/whisper-medium.en`. Supported datasets include
L2-ARCTIC, CMU ARCTIC, and personal recordings; LibriSpeech provides an additional
evaluation dataset.

## Layout

```text
s2t/                  Python pipeline and shared utilities
  datasets/           ARCTIC, personal recordings, and LibriSpeech preparation
  training/           LoRA training and adapter merging
  evaluation/         WER/CER, baselines, paired comparisons, and cloud benchmarks
  workflows/          Complete accent, speaker, and personal training pipelines
  publishing/         Hugging Face upload utility
personal_recorder/    Browser recorder and its static assets
stt_benchmark/        Configurable local/cloud benchmark application
tests/               Automated tests
docs/                Current guides and aggregate results
reports/             Local generated results (ignored except its README)
```

Datasets and trained weights are not bundled with the source. `corpus/` holds
original datasets; `data/` holds prepared audio and split manifests. Personal
recordings go in `recordings/`, training checkpoints in `runs/`, and merged
models in `models/`. These directories are excluded from Git.

## Setup and common commands

Run commands from the repository root with Python 3.10 or newer. Install a PyTorch
build suitable for your machine, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
python -m s2t.datasets.prepare_data --accent vietnamese
python -m s2t.training.train_lora --accent vietnamese --output runs/vi-lora --batch_size 8 --grad_accum 2 --epochs 10
python -m s2t.training.merge_lora --adapter runs/vi-lora/best_adapter --out models/whisper-medium-en-vi-accent
python -m s2t.evaluation.evaluate_wer --model models/whisper-medium-en-vi-accent --accent vietnamese --splits test --tag eval_vi
python -m s2t.transcribe path/to/sample.wav --model models/whisper-medium-en-vi-accent
```

The preparation command requires an existing L2-ARCTIC corpus in `corpus/`.
The commands above cover the training pipeline. See
[CMU ARCTIC instructions](docs/cmu_arctic_accents.md) for the downloadable corpus.

For complete pipelines and applications:

```bash
python -m s2t.workflows.train_all_accents --help
python -m s2t.workflows.train_one_speaker --speaker HQTV
python -m s2t.workflows.run_cmu_arctic
python -m s2t.workflows.run_personal_finetune --skip-openrouter
python -m personal_recorder --help
python -m stt_benchmark --help
python -m pytest -q
```

Personal training requires completed, verified recordings; see
[the recorder guide](docs/personal_arctic_recorder.md).
Copy `.env.example` to `.env` and fill in credentials only when using cloud
benchmarks or Hugging Face publishing. Cloud benchmark commands send audio to
the selected provider and may incur charges.

## Results

[Aggregate evaluation snapshot](docs/results.md) contains existing measurements.
Evaluation commands write detailed JSON and Markdown reports to `reports/`.
WER and CER measure transcription errors; lower values are better. Compare
models on matching test samples. Results on prompts from training speakers
do not establish performance on unseen speakers.
