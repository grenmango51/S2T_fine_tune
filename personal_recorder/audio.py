"""Audio validation, atomic file saving, and 16 kHz resampling for Personal ARCTIC Recorder."""

from dataclasses import dataclass
import hashlib
import io
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf
import soxr

TARGET_SAMPLE_RATE = 16000
MAX_TAKE_DURATION_S = 65.0  # 60s recording limit + 5s buffer


class AudioValidationError(Exception):
    """Raised when uploaded audio fails validation."""


@dataclass(frozen=True)
class AudioInfo:
    samples: np.ndarray
    sample_rate: int
    duration_s: float
    sha256: str
    num_channels: int


def compute_bytes_sha256(data: bytes) -> str:
    """Compute SHA-256 checksum of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def atomic_write_bytes(target_path: Path, data: bytes) -> None:
    """Atomically write bytes to target_path with sync."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.parent / f".tmp_{target_path.name}_{os.getpid()}"
    try:
        with open(tmp_path, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def validate_wav_bytes(
    wav_bytes: bytes,
    max_duration_s: float = MAX_TAKE_DURATION_S,
) -> AudioInfo:
    """Validate WAV bytes for completeness, finite samples, and duration.

    Converts stereo to mono if needed.
    """
    if not wav_bytes:
        raise AudioValidationError("Audio data is empty (0 bytes).")

    sha256 = compute_bytes_sha256(wav_bytes)

    try:
        bio = io.BytesIO(wav_bytes)
        samples, sr = sf.read(bio, dtype="float32", always_2d=False)
    except Exception as e:
        raise AudioValidationError(f"Failed to decode WAV audio: {e}")

    if samples.size == 0:
        raise AudioValidationError("Decoded audio contains 0 samples.")

    # Check finite numbers
    if not np.all(np.isfinite(samples)):
        raise AudioValidationError("Audio contains non-finite samples (NaN or Inf).")

    num_channels = 1
    if samples.ndim > 1:
        num_channels = samples.shape[1]
        samples = np.mean(samples, axis=1)

    duration_s = round(float(len(samples)) / float(sr), 4)
    if duration_s <= 0.0:
        raise AudioValidationError("Audio duration must be greater than 0 seconds.")

    if duration_s > max_duration_s:
        raise AudioValidationError(
            f"Audio duration ({duration_s:.2f}s) exceeds maximum allowed duration ({max_duration_s:.2f}s)."
        )

    return AudioInfo(
        samples=samples,
        sample_rate=sr,
        duration_s=duration_s,
        sha256=sha256,
        num_channels=num_channels,
    )


def resample_and_save_16k(
    samples: np.ndarray,
    orig_sr: int,
    output_path: Path,
) -> AudioInfo:
    """Resample waveform to 16 kHz mono PCM16 and save atomically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if orig_sr != TARGET_SAMPLE_RATE:
        audio_16k = soxr.resample(samples, orig_sr, TARGET_SAMPLE_RATE)
    else:
        audio_16k = samples

    # Ensure finite
    if not np.all(np.isfinite(audio_16k)):
        raise AudioValidationError("Resampled audio contains non-finite samples.")

    duration_16k = round(float(len(audio_16k)) / float(TARGET_SAMPLE_RATE), 4)

    # Write to memory buffer first to compute sha256 and write atomically
    buf = io.BytesIO()
    sf.write(buf, audio_16k, TARGET_SAMPLE_RATE, format="WAV", subtype="PCM_16")
    wav_16k_bytes = buf.getvalue()
    sha256_16k = compute_bytes_sha256(wav_16k_bytes)

    atomic_write_bytes(output_path, wav_16k_bytes)

    return AudioInfo(
        samples=audio_16k,
        sample_rate=TARGET_SAMPLE_RATE,
        duration_s=duration_16k,
        sha256=sha256_16k,
        num_channels=1,
    )
