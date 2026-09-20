"""Tests for prompt parsing, split validation, and catalog snapshot integrity."""

import json
from pathlib import Path
import tempfile
import pytest

from personal_recorder.prompts import (
    IncompatibleDatasetError,
    PromptCatalog,
    PromptEntry,
    PromptParseError,
    SplitValidationError,
    compute_file_sha256,
    create_catalog_snapshot,
    load_catalog_snapshot,
    parse_canonical_prompts,
    validate_and_assign_splits,
    verify_dataset_compatibility,
)


def test_parse_valid_prompts_with_punctuation(tmp_path: Path):
    content = """( arctic_a0001 "Author of the danger trail, Philip Steels, etc." )
( arctic_a0002 "Not at this particular case, Tom, apologized Whittemore." )
( arctic_a0004 "Lord, but I'm glad to see you again, Phil." )
( arctic_a0006 "God bless 'em, I hope I'll go on seeing them forever." )
"""
    p = tmp_path / "prompts.data"
    p.write_text(content, encoding="utf-8")

    prompts, sha = parse_canonical_prompts(p)
    assert len(prompts) == 4
    assert prompts[0] == ("arctic_a0001", "Author of the danger trail, Philip Steels, etc.")
    assert prompts[2] == ("arctic_a0004", "Lord, but I'm glad to see you again, Phil.")
    assert prompts[3] == ("arctic_a0006", "God bless 'em, I hope I'll go on seeing them forever.")
    assert sha == compute_file_sha256(p)


def test_reject_missing_parentheses(tmp_path: Path):
    content = """( arctic_a0001 "Valid prompt" )
arctic_a0002 "Missing opening parenthesis" )
"""
    p = tmp_path / "prompts.data"
    p.write_text(content, encoding="utf-8")

    with pytest.raises(PromptParseError) as exc:
        parse_canonical_prompts(p)
    assert "Line 2" in str(exc.value)


def test_reject_quote_errors(tmp_path: Path):
    content = """( arctic_a0001 "Valid prompt" )
( arctic_a0002 "Unclosed quote )
"""
    p = tmp_path / "prompts.data"
    p.write_text(content, encoding="utf-8")

    with pytest.raises(PromptParseError) as exc:
        parse_canonical_prompts(p)
    assert "Line 2" in str(exc.value)


def test_reject_empty_prompt_text(tmp_path: Path):
    content = """( arctic_a0001 "Valid prompt" )
( arctic_a0002 "" )
"""
    p = tmp_path / "prompts.data"
    p.write_text(content, encoding="utf-8")

    with pytest.raises(PromptParseError) as exc:
        parse_canonical_prompts(p)
    assert "Line 2" in str(exc.value)
    assert "empty prompt text" in str(exc.value).lower()


def test_reject_duplicate_sentence_id(tmp_path: Path):
    content = """( arctic_a0001 "First prompt" )
( arctic_a0001 "Duplicate ID prompt" )
"""
    p = tmp_path / "prompts.data"
    p.write_text(content, encoding="utf-8")

    with pytest.raises(PromptParseError) as exc:
        parse_canonical_prompts(p)
    assert "Line 2" in str(exc.value)
    assert "duplicate sentence ID" in str(exc.value)


def test_split_unassigned_fallback(tmp_path: Path):
    prompts = [
        ("arctic_a0001", "Text one"),
        ("arctic_a0002", "Text two"),
    ]
    entries, prov = validate_and_assign_splits(prompts, None)
    assert prov == "unassigned"
    assert len(entries) == 2
    assert entries[0].split == "unassigned"
    assert entries[1].split == "unassigned"


def test_split_validation_coverage_and_disjointness(tmp_path: Path):
    prompts = [
        ("arctic_a0001", "Text one"),
        ("arctic_a0002", "Text two"),
        ("arctic_a0003", "Text three"),
    ]

    # Overlapping splits (disjointness failure)
    split_file = tmp_path / "split_overlap.json"
    split_file.write_text(
        json.dumps({"train": ["arctic_a0001", "arctic_a0002"], "val": ["arctic_a0002", "arctic_a0003"]}),
        encoding="utf-8",
    )
    with pytest.raises(SplitValidationError) as exc:
        validate_and_assign_splits(prompts, split_file)
    assert "Split overlap" in str(exc.value)

    # Incomplete coverage
    split_file_missing = tmp_path / "split_missing.json"
    split_file_missing.write_text(
        json.dumps({"train": ["arctic_a0001"], "val": ["arctic_a0002"]}),
        encoding="utf-8",
    )
    with pytest.raises(SplitValidationError) as exc:
        validate_and_assign_splits(prompts, split_file_missing)
    assert "does not cover all prompts" in str(exc.value)

    # Valid disjoint and complete coverage
    split_file_valid = tmp_path / "split_valid.json"
    split_file_valid.write_text(
        json.dumps({"train": ["arctic_a0001", "arctic_a0002"], "test": ["arctic_a0003"]}),
        encoding="utf-8",
    )
    entries, prov = validate_and_assign_splits(prompts, split_file_valid)
    assert len(entries) == 3
    assert entries[0].split == "train"
    assert entries[1].split == "train"
    assert entries[2].split == "test"


def test_snapshot_roundtrip_and_compatibility(tmp_path: Path):
    prompts_file = tmp_path / "prompts.data"
    prompts_file.write_text('( arctic_a0001 "Prompt one" )\n( arctic_a0002 "Prompt two" )\n', encoding="utf-8")

    snapshot_json = tmp_path / "prompts.json"
    catalog = create_catalog_snapshot(prompts_file, None, snapshot_json)
    assert snapshot_json.is_file()

    loaded = load_catalog_snapshot(snapshot_json)
    assert loaded.source_sha256 == catalog.source_sha256
    assert len(loaded.prompts) == 2
    assert loaded.prompts[0].sentence_id == "arctic_a0001"

    # Compatibility check passes
    verify_dataset_compatibility(loaded, prompts_file, None)

    # Incompatible prompt file
    altered_prompts = tmp_path / "altered.data"
    altered_prompts.write_text('( arctic_a0001 "Altered text" )\n', encoding="utf-8")
    with pytest.raises(IncompatibleDatasetError) as exc:
        verify_dataset_compatibility(loaded, altered_prompts, None)
    assert "checksum mismatch" in str(exc.value)


def test_cmu_arctic_canonical_source():
    repo_prompts = Path("docs/cmu_arctic_sources/cmuarctic.data")
    assert repo_prompts.is_file(), "Canonical prompt file must exist in repo"
    prompts, sha = parse_canonical_prompts(repo_prompts)
    assert len(prompts) == 1132
    assert prompts[0][0] == "arctic_a0001"
    assert prompts[-1][0] == "arctic_b0539"
