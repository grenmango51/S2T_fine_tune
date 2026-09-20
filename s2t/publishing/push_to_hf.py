#!/usr/bin/env python3
"""
Upload fine-tuned Whisper medium accent and personalized models to Hugging Face Hub under Grenmango.
Generates comprehensive model cards with benchmark results and usage guides for each model.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from huggingface_hub import HfApi

from s2t.paths import PROJECT_ROOT
MODELS_DIR = PROJECT_ROOT / "models"

MODELS_SPEC = [
    {
        "dir": "whisper-medium-en-vi-accent",
        "type": "accent",
        "accent_name": "Vietnamese",
        "lang_code": "vi",
        "speakers": ["HQTV (Male)", "PNV (Female)", "THV (Female)", "TLV (Male)"],
        "train_utts": "4,072 utterances (~4.3h)",
        "test_wer": "11.42%",
        "test_cer": "6.05%",
        "base_wer": "18.85%",
        "base_cer": "9.97%",
        "turbo_wer": "15.99%",
        "turbo_cer": "8.16%",
        "rel_improvement": "39.4%",
    },
    {
        "dir": "whisper-medium-en-arabic-accent",
        "type": "accent",
        "accent_name": "Arabic",
        "lang_code": "ar",
        "speakers": ["ABA (Male)", "SKA (Female)", "YBAA (Male)", "ZHAA (Female)"],
        "train_utts": "3,927 utterances (~3.9h)",
        "test_wer": "4.44%",
        "test_cer": "2.04%",
        "base_wer": "8.21%",
        "base_cer": "3.95%",
        "turbo_wer": "36.34%",
        "turbo_cer": "27.08%",
        "rel_improvement": "45.9%",
    },
    {
        "dir": "whisper-medium-en-chinese-accent",
        "type": "accent",
        "accent_name": "Chinese",
        "lang_code": "zh",
        "speakers": ["BWC (Male)", "LXC (Female)", "NCC (Female)", "TXHC (Male)"],
        "train_utts": "4,071 utterances (~4.1h)",
        "test_wer": "7.09%",
        "test_cer": "3.57%",
        "base_wer": "11.33%",
        "base_cer": "5.65%",
        "turbo_wer": "10.93%",
        "turbo_cer": "6.89%",
        "rel_improvement": "37.4%",
    },
    {
        "dir": "whisper-medium-en-hindi-accent",
        "type": "accent",
        "accent_name": "Hindi",
        "lang_code": "hi",
        "speakers": ["ASI (Male)", "RRBI (Male)", "SVBI (Female)", "TNI (Female)"],
        "train_utts": "4,071 utterances (~3.5h)",
        "test_wer": "2.64%",
        "test_cer": "1.30%",
        "base_wer": "4.54%",
        "base_cer": "2.07%",
        "turbo_wer": "3.94%",
        "turbo_cer": "2.07%",
        "rel_improvement": "41.9%",
    },
    {
        "dir": "whisper-medium-en-korean-accent",
        "type": "accent",
        "accent_name": "Korean",
        "lang_code": "ko",
        "speakers": ["HJK (Female)", "HKK (Male)", "YDCK (Female)", "YKWK (Male)"],
        "train_utts": "4,068 utterances (~4.1h)",
        "test_wer": "4.84%",
        "test_cer": "2.21%",
        "base_wer": "7.39%",
        "base_cer": "3.50%",
        "turbo_wer": "6.26%",
        "turbo_cer": "2.84%",
        "rel_improvement": "34.5%",
    },
    {
        "dir": "whisper-medium-en-spanish-accent",
        "type": "accent",
        "accent_name": "Spanish",
        "lang_code": "es",
        "speakers": ["EBVS (Male)", "ERMS (Male)", "MBMPS (Female)", "NJS (Female)"],
        "train_utts": "3,955 utterances (~4.2h)",
        "test_wer": "5.69%",
        "test_cer": "2.49%",
        "base_wer": "9.61%",
        "base_cer": "4.47%",
        "turbo_wer": "7.85%",
        "turbo_cer": "3.64%",
        "rel_improvement": "40.8%",
    },
    {
        "dir": "whisper-medium-en-vi-hqtv-personalized",
        "type": "personalized",
        "speaker": "HQTV",
        "gender": "Male",
        "accent_name": "Vietnamese",
        "lang_code": "vi",
        "train_utts": "1,018 utterances (~1.1h)",
        "test_wer": "9.84%",
        "test_cer": "5.27%",
        "base_wer": "21.85%",
        "base_cer": "11.67%",
        "general_accent_wer": "13.39%",
        "general_accent_cer": "6.87%",
        "turbo_wer": "18.90%",
        "rel_improvement_base": "55.0%",
        "rel_improvement_accent": "26.5%",
    },
]


def create_accent_model_card(spec: dict, username: str) -> str:
    speakers_list = ", ".join(spec["speakers"])
    return f"""---
language:
- en
license: apache-2.0
tags:
- whisper
- audio
- speech
- automatic-speech-recognition
- hf-asr-leaderboard
- l2-arctic
- accented-speech
- english
- {spec['accent_name'].lower()}-accent
base_model: openai/whisper-medium.en
pipeline_tag: automatic-speech-recognition
inference: true
---

# Whisper Medium (English) - Fine-Tuned for {spec['accent_name']}-Accented English

This model is a fine-tuned, standalone merged version of [`openai/whisper-medium.en`](https://huggingface.co/openai/whisper-medium.en) specifically adapted for English speech spoken with a **{spec['accent_name']} accent**.

It was trained on native {spec['accent_name']} speakers from the [L2-ARCTIC speech corpus](https://psi.engr.tamu.edu/l2-arctic-corpus/) using Parameter-Efficient Fine-Tuning (LoRA rank=32, alpha=64), with the learned adapters permanently merged into the base model weights.

---

## Benchmark & Performance Evaluation

Evaluated against the baseline `whisper-medium.en` and `whisper-large-v3-turbo` on clean held-out read speech (`test` split):

| Model | Test Split WER | Test Split CER |
| :--- | :---: | :---: |
| **This Model (`{spec['dir']}`)** | **{spec['test_wer']}** | **{spec['test_cer']}** |
| `openai/whisper-medium.en` (Zero-Shot) | {spec['base_wer']} | {spec['base_cer']} |
| `openai/whisper-large-v3-turbo` (Zero-Shot) | {spec['turbo_wer']} | {spec['turbo_cer']} |

* **Relative WER reduction on held-out {spec['accent_name']}-accented English:** **~{spec['rel_improvement']} improvement** over zero-shot `whisper-medium.en`.

---

## Training Data & Configuration

* **Base Model:** `openai/whisper-medium.en` (769M parameters)
* **Dataset:** L2-ARCTIC ({spec['accent_name']} subset)
* **Training Utterances:** {spec['train_utts']}
* **Speakers in Corpus:** {speakers_list}
* **Acoustic Input:** 80-channel log-Mel spectrogram, 16 kHz mono audio
* **Training Method:** LoRA ($r=32, \\alpha=64$, targeting `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc2`)
* **Precision:** FP16 merged weights (compatible with standard `WhisperForConditionalGeneration`)

---

## Quickstart & Usage

This is a standalone model. You can load and use it directly with Hugging Face `transformers` without needing `peft` or any extra setup.

### 1. Using `pipeline` (Recommended)

```python
from transformers import pipeline

# Initialize the pipeline
transcriber = pipeline(
    "automatic-speech-recognition",
    model="{username}/{spec['dir']}",
    chunk_length_s=30,
    device="cuda",  # or "cpu"
)

# Transcribe an audio file (automatically resampled to 16kHz)
result = transcriber("path/to/audio.wav")
print(result["text"])
```

### 2. Direct Model & Processor Usage

```python
import torch
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration

model_id = "{username}/{spec['dir']}"
processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

# Load 16kHz audio
audio_data, sample_rate = sf.read("path/to/audio.wav")
if sample_rate != 16000:
    import soxr
    audio_data = soxr.resample(audio_data, sample_rate, 16000)

input_features = processor(audio_data, sampling_rate=16000, return_tensors="pt").input_features.to("cuda", torch.float16)

# Generate transcription
predicted_ids = model.generate(input_features, max_new_tokens=128)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print("Transcription:", transcription)
```

---

## Multi-Accent & Personalized Collection

This model is part of a complete multi-accent & personalized English fine-tuning suite on Hugging Face:

* 🇻🇳 [whisper-medium-en-vi-accent](https://huggingface.co/{username}/whisper-medium-en-vi-accent) (Vietnamese)
* 🇸🇦 [whisper-medium-en-arabic-accent](https://huggingface.co/{username}/whisper-medium-en-arabic-accent) (Arabic)
* 🇨🇳 [whisper-medium-en-chinese-accent](https://huggingface.co/{username}/whisper-medium-en-chinese-accent) (Chinese)
* 🇮🇳 [whisper-medium-en-hindi-accent](https://huggingface.co/{username}/whisper-medium-en-hindi-accent) (Hindi)
* 🇰🇷 [whisper-medium-en-korean-accent](https://huggingface.co/{username}/whisper-medium-en-korean-accent) (Korean)
* 🇪🇸 [whisper-medium-en-spanish-accent](https://huggingface.co/{username}/whisper-medium-en-spanish-accent) (Spanish)
* 👤 [whisper-medium-en-vi-hqtv-personalized](https://huggingface.co/{username}/whisper-medium-en-vi-hqtv-personalized) (Personalized - HQTV)
"""


def create_personalized_model_card(spec: dict, username: str) -> str:
    return f"""---
language:
- en
license: apache-2.0
tags:
- whisper
- audio
- speech
- automatic-speech-recognition
- hf-asr-leaderboard
- l2-arctic
- accented-speech
- personalized-speech
- english
- {spec['accent_name'].lower()}-accent
- speaker-adaptation
base_model: openai/whisper-medium.en
pipeline_tag: automatic-speech-recognition
inference: true
---

# Whisper Medium (English) - Personalized for Speaker {spec['speaker']} ({spec['accent_name']} Accent)

This model is a fine-tuned, standalone merged version of [`openai/whisper-medium.en`](https://huggingface.co/openai/whisper-medium.en) specifically adapted for **Speaker {spec['speaker']}** ({spec['gender']}, native {spec['accent_name']} accent) from the [L2-ARCTIC speech corpus](https://psi.engr.tamu.edu/l2-arctic-corpus/).

It was trained using Parameter-Efficient Fine-Tuning (LoRA rank=32, alpha=64), with the learned speaker-adapted adapters permanently merged into the base model weights.

---

## Benchmark & Performance Evaluation

Evaluated on clean held-out read speech (`test` split) specifically spoken by {spec['speaker']}:

| Model | Test Split WER | Test Split CER | Description |
| :--- | :---: | :---: | :--- |
| **This Model (`{spec['dir']}`)** | **{spec['test_wer']}** | **{spec['test_cer']}** | **Personalized to Speaker {spec['speaker']}** |
| `Grenmango/whisper-medium-en-vi-accent` | {spec['general_accent_wer']} | {spec['general_accent_cer']} | 4-Speaker Vietnamese Accent Model |
| `openai/whisper-large-v3-turbo` (Zero-Shot) | {spec['turbo_wer']} | - | Zero-Shot Large-v3-Turbo Baseline |
| `openai/whisper-medium.en` (Zero-Shot) | {spec['base_wer']} | {spec['base_cer']} | Zero-Shot Medium.en Baseline |

* **Relative WER reduction over zero-shot baseline (`whisper-medium.en`):** **{spec['rel_improvement_base']} improvement** ({spec['base_wer']} -> {spec['test_wer']}).
* **Relative WER reduction over multi-speaker accent model:** **{spec['rel_improvement_accent']} improvement** ({spec['general_accent_wer']} -> {spec['test_wer']}).
* **Outperforms zero-shot `whisper-large-v3-turbo`** by {(float(spec['turbo_wer'].rstrip('%')) - float(spec['test_wer'].rstrip('%'))):.2f} percentage points on this speaker.

---

## Training Data & Configuration

* **Base Model:** `openai/whisper-medium.en` (769M parameters)
* **Dataset:** L2-ARCTIC (Speaker `{spec['speaker']}`, {spec['accent_name']} English)
* **Training Utterances:** {spec['train_utts']}
* **Target Speaker:** {spec['speaker']} ({spec['gender']}, native {spec['accent_name']} accent)
* **Acoustic Input:** 80-channel log-Mel spectrogram, 16 kHz mono audio
* **Training Method:** LoRA ($r=32, \\alpha=64$, targeting `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc2`)
* **Precision:** FP16 merged weights (compatible with standard `WhisperForConditionalGeneration`)

---

## Quickstart & Usage

This is a standalone model. You can load and use it directly with Hugging Face `transformers` without needing `peft` or any extra setup.

### 1. Using `pipeline` (Recommended)

```python
from transformers import pipeline

# Initialize the pipeline
transcriber = pipeline(
    "automatic-speech-recognition",
    model="{username}/{spec['dir']}",
    chunk_length_s=30,
    device="cuda",  # or "cpu"
)

# Transcribe an audio file (automatically resampled to 16kHz)
result = transcriber("path/to/audio.wav")
print(result["text"])
```

### 2. Direct Model & Processor Usage

```python
import torch
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration

model_id = "{username}/{spec['dir']}"
processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

# Load 16kHz audio
audio_data, sample_rate = sf.read("path/to/audio.wav")
if sample_rate != 16000:
    import soxr
    audio_data = soxr.resample(audio_data, sample_rate, 16000)

input_features = processor(audio_data, sampling_rate=16000, return_tensors="pt").input_features.to("cuda", torch.float16)

# Generate transcription
predicted_ids = model.generate(input_features, max_new_tokens=128)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print("Transcription:", transcription)
```

---

## Multi-Accent & Personalized Collection

This model is part of a complete multi-accent & personalized English fine-tuning suite on Hugging Face:

* 🇻🇳 [whisper-medium-en-vi-accent](https://huggingface.co/{username}/whisper-medium-en-vi-accent) (Vietnamese)
* 🇸🇦 [whisper-medium-en-arabic-accent](https://huggingface.co/{username}/whisper-medium-en-arabic-accent) (Arabic)
* 🇨🇳 [whisper-medium-en-chinese-accent](https://huggingface.co/{username}/whisper-medium-en-chinese-accent) (Chinese)
* 🇮🇳 [whisper-medium-en-hindi-accent](https://huggingface.co/{username}/whisper-medium-en-hindi-accent) (Hindi)
* 🇰🇷 [whisper-medium-en-korean-accent](https://huggingface.co/{username}/whisper-medium-en-korean-accent) (Korean)
* 🇪🇸 [whisper-medium-en-spanish-accent](https://huggingface.co/{username}/whisper-medium-en-spanish-accent) (Spanish)
* 👤 [whisper-medium-en-vi-hqtv-personalized](https://huggingface.co/{username}/whisper-medium-en-vi-hqtv-personalized) (Personalized - HQTV)
"""


def main():
    parser = argparse.ArgumentParser(description="Upload L2-Arctic fine-tuned models to Hugging Face Hub")
    parser.add_argument("--model", type=str, default="all", help="Specific model dir name to push, or 'all'")
    parser.add_argument("--dry-run", action="store_true", help="Only generate model cards and check paths without uploading")
    parser.add_argument("--private", action="store_true", help="Create private repositories instead of public")
    args = parser.parse_args()

    # Read HF token from .env
    token = None
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if line.startswith("HF_TOKEN="):
                    token = line.strip().split("=", 1)[1].strip()
                elif not token and line.startswith("HUGGING_FACE_HUB_TOKEN="):
                    token = line.strip().split("=", 1)[1].strip()

    if not token:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if not token:
        raise ValueError("Could not find HF_TOKEN in .env or environment variables.")

    api = HfApi(token=token)
    user = api.whoami()
    username = user["name"]
    print(f"Authenticated as Hugging Face user: {username}")


    specs = MODELS_SPEC
    if args.model != "all":
        specs = [s for s in MODELS_SPEC if s["dir"] == args.model]
        if not specs:
            print(f"Error: Model '{args.model}' not found in MODELS_SPEC.")
            print("Available models:")
            for s in MODELS_SPEC:
                print(f"  - {s['dir']}")
            sys.exit(1)

    print(f"Target account: '{username}'")
    print(f"Models to process: {len(specs)}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'UPLOAD'}\n")

    for i, spec in enumerate(specs, 1):
        model_name = spec["dir"]
        model_path = MODELS_DIR / model_name
        repo_id = f"{username}/{model_name}"

        if not model_path.exists():
            print(f"[{i}/{len(specs)}] Warning: Directory {model_path} not found! Skipping.")
            continue

        print("=" * 65)
        print(f"[{i}/{len(specs)}] Processing: {repo_id}")
        print("=" * 65)

        # 1. Write README.md (Model Card)
        readme_path = model_path / "README.md"
        if spec.get("type") == "personalized":
            card_content = create_personalized_model_card(spec, username)
        else:
            card_content = create_accent_model_card(spec, username)

        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(card_content)
        print(f"  [✓] Generated Model Card: {readme_path.relative_to(PROJECT_ROOT)}")

        if args.dry_run:
            # Check required files
            required = ["config.json", "generation_config.json", "tokenizer_config.json", "model.safetensors", "tokenizer.json", "processor_config.json", "normalizer.json"]
            missing = [f for f in required if not (model_path / f).exists()]
            if missing:
                print(f"  [!] Missing files in {model_name}: {missing}")
            else:
                print(f"  [✓] All required model files present in {model_name}")
            continue

        # 2. Create Public / Private Repo
        print(f"  [...] Ensuring repo exists: https://huggingface.co/{repo_id} (private={args.private})")
        repo_url = api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=args.private,
            exist_ok=True,
        )
        print(f"  [✓] Repo active: {repo_url}")

        # 3. Upload Folder
        print(f"  [...] Uploading model files from {model_path.name}...")
        start_time = time.time()
        commit_desc = (
            f"Upload standalone fine-tuned {spec['accent_name']} speaker {spec.get('speaker', '')} personalized Whisper medium model"
            if spec.get("type") == "personalized"
            else f"Upload standalone fine-tuned {spec['accent_name']} accent Whisper medium model"
        )
        api.upload_folder(
            folder_path=str(model_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_desc,
        )
        elapsed = time.time() - start_time
        print(f"  [✓] Successfully uploaded {repo_id} in {elapsed:.1f}s!\n")

    print("=" * 65)
    action = "CHECKED" if args.dry_run else "PUSHED TO HUGGING FACE"
    print(f"ALL {len(specs)} MODEL(S) SUCCESSFULLY {action}!")
    print("=" * 65)


if __name__ == "__main__":
    main()
