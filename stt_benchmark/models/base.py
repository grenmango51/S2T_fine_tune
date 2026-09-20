from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TranscriptionResult:
    text: str
    latency_s: float
    cost_usd: Optional[float] = None
    error: Optional[str] = None


class STTModel(ABC):
    """Abstract base class for STT models (local or cloud)."""

    def __init__(self, name: str, model_type: str):
        self.name = name
        self.model_type = model_type  # "local" or "cloud"

    @abstractmethod
    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Transcribe a single audio file."""
        pass

    def transcribe_batch(self, audio_paths: List[str]) -> List[TranscriptionResult]:
        """Transcribe a batch of audio files. Subclasses may optimize this."""
        results = []
        for path in audio_paths:
            results.append(self.transcribe(path))
        return results
