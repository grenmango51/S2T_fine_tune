import base64
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional
import requests

from .base import STTModel, TranscriptionResult


class OpenRouterSTTModel(STTModel):
    """Cloud Speech-to-Text model via OpenRouter Audio Transcriptions API."""

    def __init__(
        self,
        model_slug: str,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        language: str = "en",
        timeout_s: float = 90.0,
        max_retries: int = 5,
        concurrency: int = 5,
    ):
        super().__init__(name=model_slug, model_type="cloud")
        self.model_slug = model_slug
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.language = language
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.concurrency = concurrency

        self.endpoint = f"{self.base_url}/audio/transcriptions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Grenmango/stt_benchmark",
            "X-Title": "STT Benchmark App",
        }

    def _prepare_payload(self, audio_path: str) -> Dict[str, Any]:
        ext = os.path.splitext(audio_path)[1].lstrip(".").lower()
        if not ext:
            ext = "wav"

        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        if file_size_mb > 25.0:
            print(f"Warning: Audio file '{audio_path}' is {file_size_mb:.1f} MB, which exceeds 25 MB limit.")

        with open(audio_path, "rb") as f:
            audio_data = f.read()

        b64_audio = base64.b64encode(audio_data).decode("utf-8")

        payload: Dict[str, Any] = {
            "model": self.model_slug,
            "input_audio": {
                "data": b64_audio,
                "format": ext,
            },
        }
        if self.language:
            payload["language"] = self.language

        return payload

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Transcribe an audio file with retry logic and latency measurement."""
        if not os.path.exists(audio_path):
            return TranscriptionResult(
                text="",
                latency_s=0.0,
                error=f"File not found: {audio_path}",
            )

        start_time = time.time()
        payload = self._prepare_payload(audio_path)

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(
                    self.endpoint,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout_s,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    elapsed = round(time.time() - start_time, 3)

                    # Extract transcription text
                    transcribed_text = ""
                    if "text" in data:
                        transcribed_text = data["text"]
                    elif "transcription" in data:
                        transcribed_text = data["transcription"]
                    elif "choices" in data and len(data["choices"]) > 0:
                        choice = data["choices"][0]
                        transcribed_text = (
                            choice.get("message", {}).get("content")
                            or choice.get("text")
                            or ""
                        )

                    # Extract cost if provided in usage
                    cost = None
                    usage = data.get("usage")
                    if usage and isinstance(usage, dict):
                        cost = usage.get("cost")
                        if cost is None and "total_cost" in usage:
                            cost = usage.get("total_cost")

                    return TranscriptionResult(
                        text=transcribed_text.strip(),
                        latency_s=elapsed,
                        cost_usd=cost,
                        error=None,
                    )

                # Handle rate limiting (429) or server errors (5xx)
                if resp.status_code in (429, 500, 502, 503, 504):
                    backoff = (2 ** (attempt - 1)) * 1.5
                    print(f"[{self.model_slug}] API HTTP {resp.status_code}. Retrying in {backoff:.1f}s (attempt {attempt}/{self.max_retries})...")
                    time.sleep(backoff)
                    last_error = f"HTTP {resp.status_code}: {resp.text}"
                    continue

                # Client error (4xx except 429)
                last_error = f"HTTP {resp.status_code}: {resp.text}"
                break

            except requests.RequestException as e:
                backoff = (2 ** (attempt - 1)) * 1.5
                print(f"[{self.model_slug}] Network error: {e}. Retrying in {backoff:.1f}s...")
                time.sleep(backoff)
                last_error = str(e)

        elapsed = round(time.time() - start_time, 3)
        return TranscriptionResult(
            text="",
            latency_s=elapsed,
            cost_usd=None,
            error=last_error or "Unknown error",
        )

    def transcribe_batch(self, audio_paths: List[str]) -> List[TranscriptionResult]:
        """Transcribe multiple audio files concurrently using a thread pool."""
        if not audio_paths:
            return []

        results: List[Optional[TranscriptionResult]] = [None] * len(audio_paths)

        with ThreadPoolExecutor(max_workers=min(self.concurrency, len(audio_paths))) as executor:
            future_to_idx = {
                executor.submit(self.transcribe, path): idx
                for idx, path in enumerate(audio_paths)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    results[idx] = TranscriptionResult(
                        text="",
                        latency_s=0.0,
                        error=str(exc),
                    )

        return [r for r in results if r is not None]


def list_available_openrouter_stt_models(api_key: Optional[str] = None, base_url: str = "https://openrouter.ai/api/v1") -> List[Dict[str, Any]]:
    """Fetch list of speech-to-text / audio transcription models supported by OpenRouter."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        url = f"{base_url.rstrip('/')}/models?output_modalities=transcription"
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code != 200:
            # Fallback to general models query
            url = f"{base_url.rstrip('/')}/models"
            resp = requests.get(url, headers=headers, timeout=30)

        if resp.status_code == 200:
            data = resp.json()
            models = data.get("data", [])
            # Filter for audio/transcription capable models
            stt_models = []
            for m in models:
                arch = m.get("architecture", {})
                output_mods = arch.get("output_modalities", [])
                input_mods = arch.get("input_modalities", [])
                mid = m.get("id", "").lower()
                name = m.get("name", "").lower()
                if (
                    "transcription" in output_mods
                    or "audio" in input_mods
                    or "whisper" in mid
                    or "transcribe" in mid
                    or "whisper" in name
                ):
                    stt_models.append(m)
            return stt_models
        else:
            print(f"Error fetching OpenRouter models: HTTP {resp.status_code} {resp.text}")
            return []
    except Exception as e:
        print(f"Exception listing OpenRouter models: {e}")
        return []
