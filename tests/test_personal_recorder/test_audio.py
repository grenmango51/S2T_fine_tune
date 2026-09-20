"""Tests for audio validation, 16 kHz resampling, and atomic file saving."""

import io
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from personal_recorder.audio import (
    AudioValidationError,
    atomic_write_bytes,
    compute_bytes_sha256,
    resample_and_save_16k,
    validate_wav_bytes,
)


def _make_wav_bytes(
    duration_s: float = 1.0,
    sample_rate: int = 44100,
    channels: int = 1,
    nan_samples: bool = False,
) -> bytes:
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    sig = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    if channels > 1:
        sig = np.column_stack([sig] * channels)
    if nan_samples:
        sig[10] = np.nan

    subtype = "FLOAT" if nan_samples else "PCM_16"
    buf = io.BytesIO()
    sf.write(buf, sig, sample_rate, format="WAV", subtype=subtype)
    return buf.getvalue()


def test_validate_valid_mono_wav():
    wav_bytes = _make_wav_bytes(duration_s=1.5, sample_rate=48000)
    info = validate_wav_bytes(wav_bytes)
    assert info.sample_rate == 48000
    assert info.duration_s == 1.5
    assert info.num_channels == 1
    assert len(info.samples) == int(48000 * 1.5)
    assert info.sha256 == compute_bytes_sha256(wav_bytes)


def test_validate_stereo_downmix_to_mono():
    wav_bytes = _make_wav_bytes(duration_s=1.0, sample_rate=44100, channels=2)
    info = validate_wav_bytes(wav_bytes)
    assert info.num_channels == 2
    assert info.samples.ndim == 1  # Successfully downmixed to 1D mono
    assert info.duration_s == 1.0


def test_reject_empty_bytes():
    with pytest.raises(AudioValidationError) as exc:
        validate_wav_bytes(b"")
    assert "empty" in str(exc.value).lower()


def test_reject_corrupt_wav_bytes():
    with pytest.raises(AudioValidationError) as exc:
        validate_wav_bytes(b"RIFFrandomgarbagebytesnotawavfile")
    assert "failed to decode" in str(exc.value).lower()


def test_reject_non_finite_samples():
    wav_bytes = _make_wav_bytes(duration_s=0.5, nan_samples=True)
    with pytest.raises(AudioValidationError) as exc:
        validate_wav_bytes(wav_bytes)
    assert "non-finite" in str(exc.value).lower()


def test_reject_oversized_audio():
    wav_bytes = _make_wav_bytes(duration_s=70.0, sample_rate=8000)
    with pytest.raises(AudioValidationError) as exc:
        validate_wav_bytes(wav_bytes, max_duration_s=65.0)
    assert "exceeds maximum allowed duration" in str(exc.value).lower()


def test_resample_and_save_16k(tmp_path: Path):
    wav_bytes = _make_wav_bytes(duration_s=2.0, sample_rate=44100)
    info = validate_wav_bytes(wav_bytes)
    out_16k = tmp_path / "processed" / "take_test.wav"

    info_16k = resample_and_save_16k(info.samples, info.sample_rate, out_16k)
    assert out_16k.is_file()
    assert info_16k.sample_rate == 16000
    assert info_16k.duration_s == 2.0

    # Verify saved file with soundfile
    saved_samples, saved_sr = sf.read(str(out_16k))
    assert saved_sr == 16000
    assert len(saved_samples) == 32000


def test_atomic_write_bytes(tmp_path: Path):
    target = tmp_path / "subdir" / "atomic.bin"
    payload = b"test payload for atomic write"
    atomic_write_bytes(target, payload)

    assert target.is_file()
    assert target.read_bytes() == payload
    # Ensure no leftover temp files
    assert len(list(target.parent.glob(".tmp_*"))) == 0
