"""Model implementations for STT benchmarking."""

from .base import STTModel, TranscriptionResult
from .local_whisper import LocalWhisperModel
from .openrouter import OpenRouterSTTModel

__all__ = ["STTModel", "TranscriptionResult", "LocalWhisperModel", "OpenRouterSTTModel"]
