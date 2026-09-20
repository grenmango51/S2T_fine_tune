"""Tests for DatasetStorage persistence, concurrency, and manifest generation."""

import csv
import io
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from personal_recorder.audio import resample_and_save_16k, validate_wav_bytes
from personal_recorder.prompts import PromptCatalog, PromptEntry
from personal_recorder.storage import ConflictError, DatasetStorage, StorageError


@pytest.fixture
def mock_catalog() -> PromptCatalog:
    return PromptCatalog(
        source_sha256="mock_sha256_hash",
        split_provenance="mock_split_provenance",
        prompts=[
            PromptEntry("arctic_a0001", "Author of the danger trail, Philip Steels, etc.", 0, "train"),
            PromptEntry("arctic_a0002", "Not at this particular case, Tom, apologized Whittemore.", 1, "val"),
            PromptEntry("arctic_a0003", "For the twentieth time that evening the two men shook hands.", 2, "test"),
        ],
    )


def _make_wav_bytes(duration_s: float = 0.5, sr: int = 16000, freq: float = 440.0) -> bytes:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    sig = (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, sig, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_storage_init_and_resumption(tmp_path: Path, mock_catalog: PromptCatalog):
    storage = DatasetStorage(tmp_path, mock_catalog, speaker="TEST_SPK", project_root=tmp_path)
    meta = storage.get_dataset_metadata()
    assert meta["speaker"] == "TEST_SPK"
    assert meta["prompt_sha256"] == "mock_sha256_hash"
    assert meta["schema_version"] == 1

    # Resume with matching catalog and speaker
    resumed = DatasetStorage(tmp_path, mock_catalog, speaker="TEST_SPK", project_root=tmp_path)
    assert resumed.get_dataset_metadata()["speaker"] == "TEST_SPK"

    # Reject mismatched speaker
    with pytest.raises(ConflictError) as exc:
        DatasetStorage(tmp_path, mock_catalog, speaker="OTHER_SPK", project_root=tmp_path)
    assert "cannot resume with speaker 'OTHER_SPK'" in str(exc.value)


def test_save_raw_take_idempotency_and_conflict(tmp_path: Path, mock_catalog: PromptCatalog):
    storage = DatasetStorage(tmp_path, mock_catalog, speaker="PERSONAL", project_root=tmp_path)
    wav1 = _make_wav_bytes(duration_s=0.5, freq=440.0)
    wav2 = _make_wav_bytes(duration_s=0.5, freq=880.0)

    # 1. Save new take
    rec, is_new = storage.save_raw_take(
        take_id="take_001",
        session_id="sess_1",
        sentence_id="arctic_a0001",
        wav_bytes=wav1,
    )
    assert is_new is True
    assert rec.take_id == "take_001"
    assert rec.conversion_status == "pending"
    assert Path(rec.original_path).is_file()

    # 2. Idempotent retry with identical data
    rec_retry, is_new_retry = storage.save_raw_take(
        take_id="take_001",
        session_id="sess_1",
        sentence_id="arctic_a0001",
        wav_bytes=wav1,
    )
    assert is_new_retry is False
    assert rec_retry.take_id == rec.take_id

    # 3. Conflicting take_id reuse with different audio
    with pytest.raises(ConflictError) as exc:
        storage.save_raw_take(
            take_id="take_001",
            session_id="sess_1",
            sentence_id="arctic_a0001",
            wav_bytes=wav2,
        )
    assert "already exists with different sentence or audio content" in str(exc.value)

    # 4. Unknown sentence ID rejected
    with pytest.raises(StorageError) as exc:
        storage.save_raw_take(
            take_id="take_999",
            session_id="sess_1",
            sentence_id="arctic_nonexistent",
            wav_bytes=wav1,
        )
    assert "not in frozen prompt catalog" in str(exc.value)


def test_keep_decision_and_superseding(tmp_path: Path, mock_catalog: PromptCatalog):
    storage = DatasetStorage(tmp_path, mock_catalog, speaker="PERSONAL", project_root=tmp_path)
    wav = _make_wav_bytes(duration_s=0.5)

    # Record two takes for the same sentence
    storage.save_raw_take("take_1a", "sess_1", "arctic_a0001", wav)
    storage.save_raw_take("take_1b", "sess_1", "arctic_a0001", wav)

    # Select take_1a
    storage.set_kept_take("arctic_a0001", "take_1a")
    assert storage.get_take("take_1a").keep_decision == 1
    assert storage.get_take("take_1b").keep_decision == 0

    # Supersede with take_1b
    storage.set_kept_take("arctic_a0001", "take_1b")
    assert storage.get_take("take_1a").keep_decision == 0
    assert storage.get_take("take_1b").keep_decision == 1


def test_manifest_generation_and_columns(tmp_path: Path, mock_catalog: PromptCatalog):
    storage = DatasetStorage(tmp_path, mock_catalog, speaker="PERSONAL", project_root=tmp_path)
    wav = _make_wav_bytes(duration_s=1.0, sr=16000)

    # Save takes for sentence 1 and sentence 2
    rec1, _ = storage.save_raw_take("take_1", "sess_1", "arctic_a0001", wav)
    rec2, _ = storage.save_raw_take("take_2", "sess_1", "arctic_a0002", wav)

    # Mark take 1 ready and keep it
    wav16k_path1 = Path(rec1.processed_path)
    resample_and_save_16k(validate_wav_bytes(wav).samples, 16000, wav16k_path1)
    storage.mark_conversion_ready("take_1", 1.0, wav16k_path1)
    storage.set_kept_take("arctic_a0001", "take_1")

    # Mark take 2 ready but do NOT keep it
    wav16k_path2 = Path(rec2.processed_path)
    resample_and_save_16k(validate_wav_bytes(wav).samples, 16000, wav16k_path2)
    storage.mark_conversion_ready("take_2", 1.0, wav16k_path2)

    assert storage.manifest_path.is_file()
    with open(storage.manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Only kept take_1 should be in manifest
    assert len(rows) == 1
    row = rows[0]
    assert row["utt_id"] == "PERSONAL_arctic_a0001"
    assert row["speaker"] == "PERSONAL"
    assert row["sentence_id"] == "arctic_a0001"
    assert row["text"] == "Author of the danger trail, Philip Steels, etc."
    assert row["split"] == "train"
    assert float(row["duration_s"]) == 1.0
    assert row["take_id"] == "take_1"
    assert row["session_id"] == "sess_1"
    assert row["review_status"] == "unreviewed"
    assert row["prompt_sha256"] == "mock_sha256_hash"


def test_writer_lease_concurrency(tmp_path: Path, mock_catalog: PromptCatalog):
    storage = DatasetStorage(tmp_path, mock_catalog, speaker="PERSONAL", project_root=tmp_path)

    # Client A acquires lease
    assert storage.acquire_or_renew_lease("client_A", ttl_seconds=10.0) is True

    # Client B fails while A's lease is valid
    assert storage.acquire_or_renew_lease("client_B", ttl_seconds=10.0) is False

    # Client A renews
    assert storage.acquire_or_renew_lease("client_A", ttl_seconds=10.0) is True

    # Client A releases
    storage.release_lease("client_A")

    # Client B can now acquire
    assert storage.acquire_or_renew_lease("client_B", ttl_seconds=10.0) is True


def test_startup_reconciliation_cleans_tmp_and_detects_missing_file(
    tmp_path: Path, mock_catalog: PromptCatalog
):
    storage = DatasetStorage(tmp_path, mock_catalog, speaker="PERSONAL", project_root=tmp_path)
    wav = _make_wav_bytes(duration_s=0.5)

    # Create stale .tmp file
    stale_tmp = storage.raw_dir / ".tmp_stale_123"
    stale_tmp.write_bytes(b"stale data")

    rec, _ = storage.save_raw_take("take_test", "sess_1", "arctic_a0001", wav)

    # Reconcile cleans stale tmp
    storage.reconcile_startup()
    assert not stale_tmp.exists()

    # If raw file is removed, startup reconciliation must raise StorageError
    Path(rec.original_path).unlink()
    with pytest.raises(StorageError) as exc:
        DatasetStorage(tmp_path, mock_catalog, speaker="PERSONAL", project_root=tmp_path)
    assert "missing on disk" in str(exc.value)
