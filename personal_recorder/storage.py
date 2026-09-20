"""SQLite persistence, writer leases, file management, and manifest generation."""

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from personal_recorder.audio import (
    AudioValidationError,
    atomic_write_bytes,
    compute_bytes_sha256,
    validate_wav_bytes,
)
from personal_recorder.prompts import PromptCatalog

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


class StorageError(Exception):
    """General storage error."""


class ConflictError(StorageError):
    """Raised when an operation conflicts with existing stored data."""


class LeaseConflictError(StorageError):
    """Raised when another client holds an active writer lease."""


@dataclass
class TakeRecord:
    take_id: str
    session_id: str
    sentence_id: str
    speaker: str
    prompt_text: str
    prompt_sha256: str
    original_path: str
    processed_path: str
    original_sr: int
    processed_sr: int
    duration_s: float
    audio_sha256: str
    mic_settings: str
    capture_time: str
    conversion_status: str  # "pending", "ready", "failed"
    conversion_error: Optional[str]
    keep_decision: int  # 0 or 1
    review_status: str  # "unreviewed"
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        try:
            d["mic_settings"] = json.loads(self.mic_settings)
        except Exception:
            pass
        return d


class DatasetStorage:
    """Authoritative SQLite-backed storage manager for a personal ARCTIC dataset."""

    def __init__(
        self,
        output_dir: Path,
        catalog: PromptCatalog,
        speaker: str,
        project_root: Optional[Path] = None,
    ):
        self.output_dir = output_dir.resolve()
        self.catalog = catalog
        self.speaker = speaker
        self.project_root = (
            project_root.resolve()
            if project_root is not None
            else Path(__file__).resolve().parent.parent
        )

        self.db_path = self.output_dir / "session.sqlite"
        self.raw_dir = self.output_dir / "raw"
        self.wav16k_dir = self.output_dir / "wav16k"
        self.manifest_path = self.output_dir / "manifest.csv"

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.wav16k_dir.mkdir(parents=True, exist_ok=True)

        self._init_db()
        self.reconcile_startup()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a SQLite connection configured for safe network-filesystem operation."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=5.0,
            isolation_level=None,  # Autocommit mode, we manage transactions explicitly
        )
        conn.row_factory = sqlite3.Row
        # Avoid WAL mode on potentially network-mounted filesystems; use DELETE journal
        conn.execute("PRAGMA journal_mode = DELETE;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self) -> None:
        """Initialize SQLite schema and verify/store dataset identity."""
        with self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dataset_metadata (
                    dataset_id TEXT PRIMARY KEY,
                    speaker TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    prompt_sha256 TEXT NOT NULL,
                    split_provenance TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    user_agent TEXT,
                    mic_label TEXT,
                    mic_settings TEXT
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS takes (
                    take_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    sentence_id TEXT NOT NULL,
                    speaker TEXT NOT NULL,
                    prompt_text TEXT NOT NULL,
                    prompt_sha256 TEXT NOT NULL,
                    original_path TEXT NOT NULL,
                    processed_path TEXT NOT NULL,
                    original_sr INTEGER NOT NULL,
                    processed_sr INTEGER NOT NULL,
                    duration_s REAL NOT NULL,
                    audio_sha256 TEXT NOT NULL,
                    mic_settings TEXT,
                    capture_time TEXT NOT NULL,
                    conversion_status TEXT NOT NULL,
                    conversion_error TEXT,
                    keep_decision INTEGER NOT NULL DEFAULT 0,
                    review_status TEXT NOT NULL DEFAULT 'unreviewed',
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_takes_sentence_id ON takes(sentence_id);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_takes_conversion_status ON takes(conversion_status);
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS leases (
                    lease_id TEXT PRIMARY KEY,
                    client_id TEXT NOT NULL,
                    acquired_at TEXT NOT NULL,
                    last_heartbeat TEXT NOT NULL,
                    expires_at REAL NOT NULL
                );
                """
            )

            # Check existing metadata
            row = conn.execute("SELECT * FROM dataset_metadata LIMIT 1;").fetchone()
            now_iso = datetime.now(timezone.utc).isoformat()
            if row is None:
                dataset_id = f"dataset_{self.speaker}_{int(datetime.now(timezone.utc).timestamp())}"
                conn.execute(
                    """
                    INSERT INTO dataset_metadata (
                        dataset_id, speaker, created_at, schema_version, prompt_sha256, split_provenance
                    ) VALUES (?, ?, ?, ?, ?, ?);
                    """,
                    (
                        dataset_id,
                        self.speaker,
                        now_iso,
                        SCHEMA_VERSION,
                        self.catalog.source_sha256,
                        self.catalog.split_provenance,
                    ),
                )
            else:
                if row["speaker"] != self.speaker:
                    conn.execute("ROLLBACK;")
                    raise ConflictError(
                        f"Existing dataset was created for speaker '{row['speaker']}', "
                        f"cannot resume with speaker '{self.speaker}'."
                    )
                if row["prompt_sha256"] != self.catalog.source_sha256:
                    conn.execute("ROLLBACK;")
                    raise ConflictError(
                        f"Existing dataset prompt checksum '{row['prompt_sha256']}' does not match "
                        f"current catalog '{self.catalog.source_sha256}'."
                    )
                if row["split_provenance"] != self.catalog.split_provenance:
                    conn.execute("ROLLBACK;")
                    raise ConflictError(
                        f"Existing dataset split provenance '{row['split_provenance']}' does not match "
                        f"current catalog '{self.catalog.split_provenance}'."
                    )

            conn.execute("COMMIT;")

    def get_dataset_metadata(self) -> Dict[str, Any]:
        """Fetch dataset metadata record."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM dataset_metadata LIMIT 1;").fetchone()
            if row is None:
                raise StorageError("Dataset metadata missing from database.")
            return dict(row)

    # -------------------------------------------------------------------------
    # Writer Lease Management
    # -------------------------------------------------------------------------

    def acquire_or_renew_lease(self, client_id: str, ttl_seconds: float = 15.0) -> bool:
        """Attempt to acquire or renew single-writer lease for client_id."""
        now = datetime.now(timezone.utc).timestamp()
        now_iso = datetime.now(timezone.utc).isoformat()
        expires = now + ttl_seconds

        with self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            # Delete expired leases
            conn.execute("DELETE FROM leases WHERE expires_at <= ?;", (now,))

            # Check if any active lease exists for a different client
            active = conn.execute(
                "SELECT * FROM leases WHERE client_id != ? AND expires_at > ?;",
                (client_id, now),
            ).fetchone()

            if active is not None:
                conn.execute("COMMIT;")
                return False

            # Update or insert lease for client_id
            conn.execute(
                """
                INSERT INTO leases (lease_id, client_id, acquired_at, last_heartbeat, expires_at)
                VALUES ('singleton_lease', ?, ?, ?, ?)
                ON CONFLICT(lease_id) DO UPDATE SET
                    client_id = excluded.client_id,
                    last_heartbeat = excluded.last_heartbeat,
                    expires_at = excluded.expires_at;
                """,
                (client_id, now_iso, now_iso, expires),
            )
            conn.execute("COMMIT;")
            return True

    def release_lease(self, client_id: str) -> None:
        """Release the lease if held by client_id."""
        with self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            conn.execute("DELETE FROM leases WHERE client_id = ?;", (client_id,))
            conn.execute("COMMIT;")

    # -------------------------------------------------------------------------
    # Sessions
    # -------------------------------------------------------------------------

    def register_session(
        self,
        session_id: str,
        user_agent: Optional[str] = None,
        mic_label: Optional[str] = None,
        mic_settings: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record or update a recording session."""
        now_iso = datetime.now(timezone.utc).isoformat()
        mic_json = json.dumps(mic_settings or {})
        with self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            conn.execute(
                """
                INSERT INTO sessions (session_id, started_at, user_agent, mic_label, mic_settings)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    user_agent = COALESCE(excluded.user_agent, sessions.user_agent),
                    mic_label = COALESCE(excluded.mic_label, sessions.mic_label),
                    mic_settings = COALESCE(excluded.mic_settings, sessions.mic_settings);
                """,
                (session_id, now_iso, user_agent, mic_label, mic_json),
            )
            conn.execute("COMMIT;")

    # -------------------------------------------------------------------------
    # Takes & Audio Persistence
    # -------------------------------------------------------------------------

    def save_raw_take(
        self,
        take_id: str,
        session_id: str,
        sentence_id: str,
        wav_bytes: bytes,
        mic_settings: Optional[Dict[str, Any]] = None,
        capture_time: Optional[str] = None,
    ) -> Tuple[TakeRecord, bool]:
        """Save raw WAV take and insert into SQLite with conversion_status='pending'.

        Returns (TakeRecord, is_new).
        Idempotent: Identical retry returns existing record.
        Conflict: Reused take_id with different audio or sentence raises ConflictError.
        """
        # Validate sentence_id against frozen catalog
        prompt_entry = self.catalog.get_prompt(sentence_id)
        if prompt_entry is None:
            raise StorageError(f"Sentence ID {sentence_id!r} is not in frozen prompt catalog.")

        # Validate WAV bytes
        audio_info = validate_wav_bytes(wav_bytes)

        now_iso = datetime.now(timezone.utc).isoformat()
        cap_time = capture_time or now_iso
        mic_json = json.dumps(mic_settings or {})

        # Path calculations
        raw_file = self.raw_dir / sentence_id / f"{take_id}.wav"
        wav16k_file = self.wav16k_dir / sentence_id / f"{take_id}.wav"

        raw_path_str = str(raw_file.resolve())
        wav16k_path_str = str(wav16k_file.resolve())

        with self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            existing = conn.execute(
                "SELECT * FROM takes WHERE take_id = ?;", (take_id,)
            ).fetchone()

            if existing is not None:
                # Check for conflict
                if (
                    existing["sentence_id"] != sentence_id
                    or existing["audio_sha256"] != audio_info.sha256
                ):
                    conn.execute("ROLLBACK;")
                    raise ConflictError(
                        f"Take ID {take_id!r} already exists with different sentence or audio content."
                    )
                # Idempotent retry
                rec = TakeRecord(**dict(existing))
                conn.execute("COMMIT;")
                return rec, False

            # Ensure session is registered
            conn.execute(
                """
                INSERT OR IGNORE INTO sessions (session_id, started_at)
                VALUES (?, ?);
                """,
                (session_id, now_iso),
            )

            # Atomic file write before DB commit
            atomic_write_bytes(raw_file, wav_bytes)

            conn.execute(
                """
                INSERT INTO takes (
                    take_id, session_id, sentence_id, speaker, prompt_text, prompt_sha256,
                    original_path, processed_path, original_sr, processed_sr, duration_s,
                    audio_sha256, mic_settings, capture_time, conversion_status,
                    conversion_error, keep_decision, review_status, created_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, 'pending',
                    NULL, 0, 'unreviewed', ?
                );
                """,
                (
                    take_id,
                    session_id,
                    sentence_id,
                    self.speaker,
                    prompt_entry.text,
                    self.catalog.source_sha256,
                    raw_path_str,
                    wav16k_path_str,
                    audio_info.sample_rate,
                    16000,
                    audio_info.duration_s,
                    audio_info.sha256,
                    mic_json,
                    cap_time,
                    now_iso,
                ),
            )
            row = conn.execute("SELECT * FROM takes WHERE take_id = ?;", (take_id,)).fetchone()
            conn.execute("COMMIT;")
            return TakeRecord(**dict(row)), True

    def mark_conversion_ready(
        self,
        take_id: str,
        duration_16k: float,
        processed_path: Path,
    ) -> TakeRecord:
        """Mark take as 16 kHz conversion ready and rebuild manifest if kept."""
        path_str = str(processed_path.resolve())
        with self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            conn.execute(
                """
                UPDATE takes SET
                    conversion_status = 'ready',
                    conversion_error = NULL,
                    processed_path = ?,
                    duration_s = ?
                WHERE take_id = ?;
                """,
                (path_str, duration_16k, take_id),
            )
            row = conn.execute("SELECT * FROM takes WHERE take_id = ?;", (take_id,)).fetchone()
            if row is None:
                conn.execute("ROLLBACK;")
                raise StorageError(f"Take {take_id} not found.")
            rec = TakeRecord(**dict(row))
            conn.execute("COMMIT;")

        if rec.keep_decision == 1:
            self.rebuild_manifest()
        return rec

    def mark_conversion_failed(self, take_id: str, error_message: str) -> None:
        """Mark take as failed during 16 kHz conversion."""
        with self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            conn.execute(
                """
                UPDATE takes SET
                    conversion_status = 'failed',
                    conversion_error = ?
                WHERE take_id = ?;
                """,
                (error_message, take_id),
            )
            conn.execute("COMMIT;")

    def set_kept_take(self, sentence_id: str, take_id: str) -> TakeRecord:
        """Select take_id as the kept take for sentence_id, superseding others."""
        with self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            take_row = conn.execute(
                "SELECT * FROM takes WHERE take_id = ? AND sentence_id = ?;",
                (take_id, sentence_id),
            ).fetchone()
            if take_row is None:
                conn.execute("ROLLBACK;")
                raise StorageError(f"Take {take_id} for sentence {sentence_id} does not exist.")

            # Supersede all takes for this sentence
            conn.execute(
                "UPDATE takes SET keep_decision = 0 WHERE sentence_id = ?;",
                (sentence_id,),
            )
            # Mark selected take as kept
            conn.execute(
                "UPDATE takes SET keep_decision = 1 WHERE take_id = ?;",
                (take_id,),
            )
            updated_row = conn.execute(
                "SELECT * FROM takes WHERE take_id = ?;", (take_id,)
            ).fetchone()
            rec = TakeRecord(**dict(updated_row))
            conn.execute("COMMIT;")

        self.rebuild_manifest()
        return rec

    def get_take(self, take_id: str) -> Optional[TakeRecord]:
        """Fetch take record by take_id."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM takes WHERE take_id = ?;", (take_id,)).fetchone()
            if row is None:
                return None
            return TakeRecord(**dict(row))

    def get_takes_for_sentence(self, sentence_id: str) -> List[TakeRecord]:
        """Fetch all takes for a sentence in chronological order."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM takes WHERE sentence_id = ? ORDER BY created_at ASC;",
                (sentence_id,),
            ).fetchall()
            return [TakeRecord(**dict(r)) for r in rows]

    def get_pending_conversions(self) -> List[TakeRecord]:
        """Fetch takes awaiting 16 kHz conversion."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM takes WHERE conversion_status = 'pending' ORDER BY created_at ASC;"
            ).fetchall()
            return [TakeRecord(**dict(r)) for r in rows]

    def get_stats(self) -> Dict[str, Any]:
        """Compute progress and storage stats."""
        with self._get_connection() as conn:
            total_takes = conn.execute("SELECT COUNT(*) FROM takes;").fetchone()[0]
            kept_sentences = conn.execute(
                "SELECT COUNT(DISTINCT sentence_id) FROM takes WHERE keep_decision = 1;"
            ).fetchone()[0]
            pending_conversions = conn.execute(
                "SELECT COUNT(*) FROM takes WHERE conversion_status = 'pending';"
            ).fetchone()[0]
            ready_takes = conn.execute(
                "SELECT COUNT(*) FROM takes WHERE conversion_status = 'ready';"
            ).fetchone()[0]

        total_prompts = len(self.catalog.prompts)
        return {
            "total_prompts": total_prompts,
            "kept_prompts": kept_sentences,
            "total_takes": total_takes,
            "ready_takes": ready_takes,
            "pending_conversions": pending_conversions,
            "review_status": "unreviewed",
        }

    # -------------------------------------------------------------------------
    # Reconciliation & Manifest Generation
    # -------------------------------------------------------------------------

    def reconcile_startup(self) -> None:
        """Reconcile orphaned temp files and verify integrity at startup."""
        # 1. Clean orphaned temporary files in raw and wav16k
        for folder in [self.raw_dir, self.wav16k_dir, self.output_dir]:
            if not folder.exists():
                continue
            for item in folder.rglob(".tmp_*"):
                try:
                    if item.is_file():
                        item.unlink()
                        logger.info("Cleaned stale temporary file: %s", item)
                except OSError as e:
                    logger.warning("Failed to remove temporary file %s: %s", item, e)

        # 2. Verify all stored takes have existing raw audio files
        with self._get_connection() as conn:
            rows = conn.execute("SELECT take_id, original_path FROM takes;").fetchall()
            for r in rows:
                p = Path(r["original_path"])
                if not p.is_file():
                    raise StorageError(
                        f"Authoritative raw audio file missing on disk for take {r['take_id']}: {p}. "
                        "Cannot silently report missing file as saved."
                    )

        # 3. Rebuild manifest to ensure snapshot is in sync
        self.rebuild_manifest()

    def _format_manifest_path(self, abs_path: Path) -> str:
        """Format audio path for manifest: relative if inside project root, absolute otherwise."""
        try:
            return str(abs_path.resolve().relative_to(self.project_root))
        except ValueError:
            return str(abs_path.resolve())

    def rebuild_manifest(self) -> None:
        """Atomically rebuild manifest.csv from selected, ready takes."""
        # Get all kept takes that have completed conversion
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM takes
                WHERE keep_decision = 1 AND conversion_status = 'ready'
                ORDER BY sentence_id ASC;
                """
            ).fetchall()

        kept_by_sentence = {r["sentence_id"]: TakeRecord(**dict(r)) for r in rows}

        manifest_rows: List[Dict[str, Any]] = []
        # Maintain exact prompt catalog order
        for p in self.catalog.prompts:
            rec = kept_by_sentence.get(p.sentence_id)
            if rec is None:
                continue

            path_formatted = self._format_manifest_path(Path(rec.processed_path))
            manifest_rows.append(
                {
                    "utt_id": f"{self.speaker}_{rec.sentence_id}",
                    "speaker": self.speaker,
                    "sentence_id": rec.sentence_id,
                    "path": path_formatted,
                    "text": rec.prompt_text,
                    "split": p.split,
                    "duration_s": rec.duration_s,
                    "take_id": rec.take_id,
                    "session_id": rec.session_id,
                    "review_status": rec.review_status,
                    "prompt_sha256": rec.prompt_sha256,
                }
            )

        tmp_manifest = self.manifest_path.parent / f".tmp_{self.manifest_path.name}_{os.getpid()}"
        fieldnames = [
            "utt_id",
            "speaker",
            "sentence_id",
            "path",
            "text",
            "split",
            "duration_s",
            "take_id",
            "session_id",
            "review_status",
            "prompt_sha256",
        ]

        try:
            with open(tmp_manifest, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in manifest_rows:
                    writer.writerow(row)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_manifest, self.manifest_path)
        finally:
            if tmp_manifest.exists():
                try:
                    tmp_manifest.unlink()
                except OSError:
                    pass
