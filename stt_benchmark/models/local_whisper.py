import time
from typing import List, Optional
import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration

import s2t.common as common
from .base import STTModel, TranscriptionResult


class LocalWhisperModel(STTModel):
    """Local Whisper speech-to-text model loaded via Hugging Face Transformers."""

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        language: str = "en",
    ):
        super().__init__(name=model_name_or_path, model_type="local")
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.language = language

        print(f"Loading local Whisper model '{model_name_or_path}' on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)

        dtype = torch.float16 if "cuda" in self.device and torch.cuda.is_available() else torch.float32
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            device_map=self.device,
        )
        self.model.eval()

        self.is_multilingual = getattr(self.model.config, "is_multilingual", False)

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        results = self.transcribe_batch([audio_path])
        return results[0]

    def transcribe_batch(self, audio_paths: List[str]) -> List[TranscriptionResult]:
        if not audio_paths:
            return []

        start_time = time.time()
        try:
            audio_batch = [common.load_audio_16k(p) for p in audio_paths]
            dtype = torch.float16 if "cuda" in self.device and torch.cuda.is_available() else torch.float32

            inputs = self.processor(
                audio_batch,
                sampling_rate=common.SAMPLE_RATE,
                return_tensors="pt",
                return_attention_mask=True,
            )
            input_features = inputs.input_features.to(self.device, dtype=dtype)
            attention_mask = inputs.attention_mask.to(self.device)

            gen_kwargs = {}
            if self.is_multilingual and self.language:
                gen_kwargs["language"] = self.language
                gen_kwargs["task"] = "transcribe"

            with torch.inference_mode():
                pred_ids = self.model.generate(
                    input_features,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )

            preds = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
            elapsed = time.time() - start_time
            per_item_latency = round(elapsed / len(audio_paths), 4)

            return [
                TranscriptionResult(
                    text=p.strip(),
                    latency_s=per_item_latency,
                    cost_usd=0.0,
                    error=None,
                )
                for p in preds
            ]
        except Exception as e:
            elapsed = time.time() - start_time
            per_item_latency = round(elapsed / len(audio_paths), 4)
            return [
                TranscriptionResult(
                    text="",
                    latency_s=per_item_latency,
                    cost_usd=0.0,
                    error=str(e),
                )
                for _ in audio_paths
            ]
