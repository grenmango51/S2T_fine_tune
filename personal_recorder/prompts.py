"""Prompt parsing, split assignment, and snapshot management for Personal ARCTIC Recorder."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Dict, List, Optional, Set, Tuple


class PromptParseError(Exception):
    """Raised when a canonical prompt file is malformed."""


class SplitValidationError(Exception):
    """Raised when split assignments are invalid or incomplete."""


class IncompatibleDatasetError(Exception):
    """Raised when an existing dataset does not match current configuration."""


@dataclass(frozen=True)
class PromptEntry:
    sentence_id: str
    text: str
    order_index: int
    split: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PromptEntry":
        return cls(
            sentence_id=data["sentence_id"],
            text=data["text"],
            order_index=data["order_index"],
            split=data["split"],
        )


@dataclass(frozen=True)
class PromptCatalog:
    source_sha256: str
    split_provenance: str
    prompts: List[PromptEntry]

    @property
    def sentence_ids(self) -> List[str]:
        return [p.sentence_id for p in self.prompts]

    def get_prompt(self, sentence_id: str) -> Optional[PromptEntry]:
        for p in self.prompts:
            if p.sentence_id == sentence_id:
                return p
        return None

    def to_dict(self) -> dict:
        return {
            "source_sha256": self.source_sha256,
            "split_provenance": self.split_provenance,
            "prompts": [p.to_dict() for p in self.prompts],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PromptCatalog":
        return cls(
            source_sha256=data["source_sha256"],
            split_provenance=data["split_provenance"],
            prompts=[PromptEntry.from_dict(p) for p in data["prompts"]],
        )


# Strict pattern for Arctic prompt lines: ( arctic_a0001 "Prompt text here" )
PROMPT_LINE_REGEX = re.compile(r"^\(\s*([a-zA-Z0-9_]+)\s+\"([^\"]*)\"\s*\)$")


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def parse_canonical_prompts(path: Path) -> Tuple[List[Tuple[str, str]], str]:
    """Strictly parse canonical CMU Arctic prompts file.

    Never uses eval. Preserves exact sentence IDs, spelling, punctuation, and order.
    Rejects malformed lines, duplicate IDs, and empty text with 1-based line numbers.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Canonical prompt file not found: {path}")

    source_sha256 = compute_file_sha256(path)
    prompts: List[Tuple[str, str]] = []
    seen_ids: Set[str] = set()

    with open(path, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line:
                continue

            # Ensure non-empty line starts with ( and ends with )
            if not (line.startswith("(") and line.endswith(")")):
                raise PromptParseError(
                    f"Line {line_num}: malformed line, must start with '(' and end with ')'."
                )

            # Check quote count: must contain exactly 2 quotes
            quote_count = line.count('"')
            if quote_count != 2:
                raise PromptParseError(
                    f"Line {line_num}: malformed line, expected exactly 2 double quotes enclosing prompt text, found {quote_count}."
                )

            match = PROMPT_LINE_REGEX.match(line)
            if not match:
                raise PromptParseError(
                    f"Line {line_num}: malformed prompt syntax: {line!r}"
                )

            sentence_id, text = match.group(1), match.group(2)
            if not text.strip():
                raise PromptParseError(
                    f"Line {line_num}: empty prompt text for sentence ID {sentence_id!r}."
                )

            if sentence_id in seen_ids:
                raise PromptParseError(
                    f"Line {line_num}: duplicate sentence ID {sentence_id!r} found."
                )

            seen_ids.add(sentence_id)
            prompts.append((sentence_id, text))

    if not prompts:
        raise PromptParseError(f"Prompt file {path} contains no valid prompt entries.")

    return prompts, source_sha256


def validate_and_assign_splits(
    parsed_prompts: List[Tuple[str, str]],
    split_ids_path: Optional[Path],
) -> Tuple[List[PromptEntry], str]:
    """Validate split assignments for coverage and disjointness, returning frozen entries."""
    if split_ids_path is None:
        entries = [
            PromptEntry(
                sentence_id=sid,
                text=text,
                order_index=idx,
                split="unassigned",
            )
            for idx, (sid, text) in enumerate(parsed_prompts)
        ]
        return entries, "unassigned"

    if not split_ids_path.is_file():
        raise FileNotFoundError(f"Split file not found: {split_ids_path}")

    with open(split_ids_path, "r", encoding="utf-8") as f:
        try:
            split_data = json.load(f)
        except Exception as e:
            raise SplitValidationError(f"Invalid JSON in split file {split_ids_path}: {e}")

    # Extract split partitions (keys with list values)
    split_map: Dict[str, str] = {}
    partitions: Dict[str, List[str]] = {
        k: v for k, v in split_data.items() if isinstance(v, list)
    }

    if not partitions:
        raise SplitValidationError(f"No split partition lists found in {split_ids_path}")

    # Check disjointness
    for split_name, ids in partitions.items():
        for sid in ids:
            if not isinstance(sid, str):
                raise SplitValidationError(
                    f"Non-string sentence ID found in split {split_name}: {sid!r}"
                )
            if sid in split_map:
                prev_split = split_map[sid]
                raise SplitValidationError(
                    f"Split overlap: sentence ID {sid!r} belongs to both '{prev_split}' and '{split_name}'."
                )
            split_map[sid] = split_name

    # Check coverage
    catalog_ids = {sid for sid, _ in parsed_prompts}
    missing_from_splits = catalog_ids - set(split_map.keys())
    if missing_from_splits:
        sample_missing = sorted(list(missing_from_splits))[:5]
        raise SplitValidationError(
            f"Split file does not cover all prompts. Missing {len(missing_from_splits)} IDs, "
            f"e.g., {sample_missing}"
        )

    entries = [
        PromptEntry(
            sentence_id=sid,
            text=text,
            order_index=idx,
            split=split_map[sid],
        )
        for idx, (sid, text) in enumerate(parsed_prompts)
    ]
    return entries, str(split_ids_path.resolve())


def create_catalog_snapshot(
    prompts_path: Path,
    split_ids_path: Optional[Path],
    snapshot_json_path: Path,
) -> PromptCatalog:
    """Create and write an immutable snapshot of prompts and split assignments."""
    parsed_prompts, source_sha256 = parse_canonical_prompts(prompts_path)
    entries, split_provenance = validate_and_assign_splits(parsed_prompts, split_ids_path)

    catalog = PromptCatalog(
        source_sha256=source_sha256,
        split_provenance=split_provenance,
        prompts=entries,
    )

    snapshot_json_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = snapshot_json_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(catalog.to_dict(), f, indent=2, ensure_ascii=False)
    tmp_path.replace(snapshot_json_path)

    return catalog


def load_catalog_snapshot(snapshot_json_path: Path) -> PromptCatalog:
    """Load existing immutable prompt catalog snapshot."""
    if not snapshot_json_path.is_file():
        raise FileNotFoundError(f"Prompt catalog snapshot not found: {snapshot_json_path}")
    with open(snapshot_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return PromptCatalog.from_dict(data)


def verify_dataset_compatibility(
    existing_catalog: PromptCatalog,
    prompts_path: Path,
    split_ids_path: Optional[Path],
) -> None:
    """Verify that current prompt file and split options match the existing dataset snapshot."""
    curr_sha256 = compute_file_sha256(prompts_path)
    if existing_catalog.source_sha256 != curr_sha256:
        raise IncompatibleDatasetError(
            f"Prompt source checksum mismatch! Existing dataset was created with SHA-256 "
            f"{existing_catalog.source_sha256}, but provided prompt file {prompts_path} "
            f"has SHA-256 {curr_sha256}. Resuming requires identical prompt source."
        )

    expected_provenance = "unassigned" if split_ids_path is None else str(split_ids_path.resolve())
    if existing_catalog.split_provenance != expected_provenance:
        raise IncompatibleDatasetError(
            f"Split assignment mismatch! Existing dataset was created with split provenance "
            f"'{existing_catalog.split_provenance}', but launch specifies '{expected_provenance}'. "
            f"Split assignment must be frozen per dataset."
        )
