# Whisper Fine-Tuning Technical Specification & Execution Guide

> **Document Type:** Machine-Readable Implementation Spec / AI Developer Reference  
> **Target Models:** `openai/whisper-{tiny, base, small, medium, large, large-v2, large-v3}`  
> **Frameworks:** 🤗 `transformers`, `datasets`, `accelerate`, `evaluate`, PyTorch  
> **Source Baseline:** Hugging Face Multilingual ASR Guide (Sanchit Gandhi)  

---

## 1. Architectural Invariants & Tensor Specifications

### 1.1 Model Topology
* **Architecture:** Sequence-to-Sequence (Encoder-Decoder) Transformer with internal deep fusion language modeling.
* **Objective:** Autoregressive token classification via Cross-Entropy Loss over Byte-Pair Encoded (BPE) vocabulary.
* **Acoustic Input:** 80-channel log-Mel spectrogram (128-channel for `large-v3`), computed over 25ms windows with 10ms hop size.
* **Temporal Window:** Fixed 30.0-second chunking. Shorter audio is right-padded with zeros; longer audio is truncated to 30s.
* **Attention Mask:** **None**. The encoder receives a static tensor shape and learns silence/pad boundaries directly from input features without attention masks.

### 1.2 Tensor Shapes & Types

| Stage | Tensor Identifier | Dimensions | Data Type | Value Range / Properties |
| :--- | :--- | :--- | :--- | :--- |
| **Raw Audio** | `waveform` | `(N,)` | `float32` | `[-1.0, 1.0]`, Sample Rate = 16,000 Hz |
| **Encoder Input** | `input_features` | `(B, 80, 3000)` | `float32` | Normalized log-Mel spectrogram (`large-v3` has shape `(B, 128, 3000)`) |
| **Decoder Target** | `labels` | `(B, T)` | `int64` | Token IDs padded with `-100` (ignored by Cross-Entropy Loss) |
| **Decoder Input** | `decoder_input_ids`| `(B, T)` | `int64` | Right-shifted labels prepended with `decoder_start_token_id` |

### 1.3 Checkpoint Specifications

| Checkpoint | Layers (Enc/Dec) | Hidden Dim ($d_{\text{model}}$) | Attention Heads | Parameters | Mel Bins | FP16 Training Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `openai/whisper-tiny` | 4 / 4 | 384 | 6 | 39 M | 80 | ~4 GB |
| `openai/whisper-base` | 6 / 6 | 512 | 8 | 74 M | 80 | ~6 GB |
| `openai/whisper-small` | 12 / 12 | 768 | 12 | 244 M | 80 | ~14 - 16 GB |
| `openai/whisper-medium` | 24 / 24 | 1024 | 16 | 769 M | 80 | ~24 - 32 GB |
| `openai/whisper-large-v3` | 32 / 32 | 1280 | 20 | 1550 M | 128 | >32 GB (Use LoRA/PEFT if <32GB) |

---

## 2. Tokenizer Control Sequence Grammar

Whisper conditions decoding using a deterministic sequence of special prefix tokens:

```
<|startoftranscript|> -> <|{language}|> -> <|{task}|> -> <|notimestamps|> -> [Target Text Tokens] -> <|endoftext|>
```

* `<|startoftranscript|>` (`50258`): Initializes decoder autoregression.
* `<|{language}|>` (e.g., `<|hi|>`: `50276`, `<|es|>`: `50262`, `<|en|>`: `50259`): Explicit language conditioning token.
* `<|{task}|>`: `<|transcribe|>` (`50359`) for ASR; `<|translate|>` (`50358`) for direct speech-to-English translation.
* `<|notimestamps|>` (`50363`): Enforces pure text generation without segment timestamp predictions.
* `<|endoftext|>` (`50257`): EOS token terminating autoregression.

---

## 3. Hardware Profiles & Memory Optimization

To avoid CUDA Out-Of-Memory (OOM) errors during full-parameter fine-tuning:

| Available GPU VRAM | Target Model | `per_device_train_batch_size` | `gradient_accumulation_steps` | Effective Batch Size | Notes |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **8 GB - 10 GB** | `small` | 2 | 8 | 16 | Enable `fp16`, `gradient_checkpointing` |
| **12 GB - 16 GB** | `small` | 4 or 8 | 4 or 2 | 16 | Enable `fp16`, `gradient_checkpointing` |
| **24 GB (3090/4090/A10G)**| `small` | 16 | 1 | 16 | Baseline recommended configuration |
| **24 GB** | `medium` | 4 | 4 | 16 | Enable `fp16`, `gradient_checkpointing` |
| **< 16 GB** | `large-v3` | 2 or 4 | 8 or 4 | 16 | **Must** use PEFT / LoRA (Rank 16/32) |

---

## 4. End-to-End Implementation Protocol

### 4.1 Dependency Setup
```bash
pip install --upgrade pip
pip install --upgrade "datasets[audio]>=2.14.0" "transformers>=4.34.0" accelerate evaluate jiwer soundfile
```

### 4.2 Data Ingestion & Resampling Protocol
* Every input must be resampled to **16,000 Hz**.
* High-sampling datasets (e.g. 48 kHz in Common Voice) must be cast dynamically using `cast_column("audio", Audio(sampling_rate=16000))` to prevent high-frequency aliasing or tempo-distortion.
* Drop extraneous dataset columns prior to batch collation to conserve RAM.

```python
from datasets import load_dataset, DatasetDict, Audio

common_voice = DatasetDict()
common_voice["train"] = load_dataset(
    "mozilla-foundation/common_voice_11_0", "hi", split="train+validation", use_auth_token=True
)
common_voice["test"] = load_dataset(
    "mozilla-foundation/common_voice_11_0", "hi", split="test", use_auth_token=True
)

# Strip non-essential columns
common_voice = common_voice.remove_columns([
    "accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"
])

# Enforce 16kHz resampling
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
```

### 4.3 Feature Extractor, Tokenizer, & Processor Initialization

```python
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor

model_id = "openai/whisper-small"
target_language = "Hindi"   # Case-sensitive language name
target_task = "transcribe"  # "transcribe" or "translate"

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
tokenizer = WhisperTokenizer.from_pretrained(model_id, language=target_language, task=target_task)
processor = WhisperProcessor.from_pretrained(model_id, language=target_language, task=target_task)
```

### 4.4 Mapping Pipeline

```python
def prepare_dataset(batch):
    audio = batch["audio"]
    # Feature extractor converts raw 1D float array into normalized 80-channel log-mel spectrogram
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    # Tokenize target text transcription to integer IDs
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

common_voice = common_voice.map(
    prepare_dataset,
    remove_columns=common_voice.column_names["train"],
    num_proc=2  # Set num_proc=1 if running on Windows to prevent multi-processing deadlock
)
```

---

## 5. Sequence-to-Sequence Data Collator

The data collator must resolve two asymmetric dimensional constraints:
1. **Audio dimension:** Statically padded to 3000 frames $(80 \times 3000)$.
2. **Text label dimension:** Dynamically padded to the maximum token sequence in the current batch; all padding positions masked to `-100`.
3. **BOS token removal:** If the tokenizer prepended `<|startoftranscript|>`, strip the first token to avoid doubling it with the decoder's internally generated start token.

```python
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Batch audio input features
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Batch and pad labels
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Mask padding tokens with -100 to omit from Cross-Entropy Loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Truncate leading BOS if tokenizer prepended it
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
```

---

## 6. Evaluation Metric (Word Error Rate)

```python
import evaluate

metric = evaluate.load("wer")

def compute_metrics(pred, tokenizer):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Unmask -100 tokens back to pad_token_id for correct decoding
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}
```

---

## 7. Model Instantiation & Training Execution

```python
from transformers import WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer

# 1. Model Loading
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# 2. Generation Config Override
model.generation_config.language = "hindi"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

# 3. Data Collator Initialization
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# 4. Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-hi",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

# 5. Persist processor configuration
processor.save_pretrained(training_args.output_dir)

# 6. Trainer Instantiation & Launch
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=lambda pred: compute_metrics(pred, tokenizer),
    tokenizer=processor.feature_extractor,
)

trainer.train()
```

---

## 8. Invariants & Common Failure Modes

### Pitfall 1: Sample Rate Mismatch
* **Symptom:** Divergent loss, repetitive nonsense generation, WER > 90%.
* **Root Cause:** Input audio passed at 44.1 kHz or 48 kHz without resampling. Whisper models strictly require 16,000 Hz.
* **Fix:** `dataset.cast_column("audio", Audio(sampling_rate=16000))` before extracting spectrogram features.

### Pitfall 2: Double BOS Token Injection
* **Symptom:** Poor generation initial tokens, early EOS prediction, truncated transcripts.
* **Root Cause:** Tokenizer appends `<|startoftranscript|>` to labels, and the decoder prepends `decoder_start_token_id` during teacher-forcing.
* **Fix:** Data collator check: `if (labels[:, 0] == decoder_start_token_id).all(): labels = labels[:, 1:]`.

### Pitfall 3: Ineffective Attention Mask on Encoder
* **Symptom:** Attempting to pass `attention_mask` to Whisper encoder throws errors or creates degraded features.
* **Root Cause:** Whisper is designed without input attention masking. Mel-spectrogram inputs are fixed length (30s = 3000 frames).

### Pitfall 4: Neglecting `-100` Label Masking
* **Symptom:** Model learns to emit pad tokens repeatedly; validation loss explodes.
* **Root Cause:** Padding tokens left as `tokenizer.pad_token_id` in `labels` instead of `-100`, forcing the loss function to penalize non-pad predictions on padded frames.

### Pitfall 5: Eval OOM during Generation
* **Symptom:** Training steps succeed, but crash occurs immediately when reaching `eval_steps`.
* **Root Cause:** `predict_with_generate=True` uses autoregressive beam/greedy decoding, which stores KV-cache in GPU memory.
* **Fix:** Reduce `per_device_eval_batch_size` (e.g. from 16 to 4 or 8) and cap `generation_max_length=225`.

---

## 9. Parameter-Efficient Fine-Tuning (PEFT / LoRA Extension)

For fine-tuning on GPUs with $\le$ 12 GB VRAM or targeting `medium` / `large-v3`:

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Target attention projections in both encoder and decoder
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# Typical output: trainable params: ~3.5M || all params: ~244M || trainable%: ~1.4%
```

---

## 10. Post-Training Inference Pipeline

```python
from transformers import pipeline

pipe = pipeline(
    task="automatic-speech-recognition",
    model="./whisper-small-hi",
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30,  # Enables sliding-window chunking for audio > 30s
    device="cuda:0",
)

transcription = pipe("test_audio_sample.wav")["text"]
print(transcription)
```
