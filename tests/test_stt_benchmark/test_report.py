import json
import os
import tempfile
import pytest

from stt_benchmark.config import BenchmarkConfig
from stt_benchmark.metrics import compute_utterance_result, aggregate_model_results
from stt_benchmark.report import generate_benchmark_reports


def test_generate_benchmark_reports():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create synthetic model results
        utts1 = [
            compute_utterance_result("utt1", "hello world", "hello world", 0.3, "local_mod", speaker="SPK1"),
            compute_utterance_result("utt2", "good morning", "good evening", 0.4, "local_mod", speaker="SPK2"),
        ]
        res1 = aggregate_model_results("local_mod", "local", utts1)

        utts2 = [
            compute_utterance_result("utt1", "hello world", "hello world", 0.9, "cloud_mod", speaker="SPK1", cost_usd=0.001),
            compute_utterance_result("utt2", "good morning", "good morning", 1.1, "cloud_mod", speaker="SPK2", cost_usd=0.001),
        ]
        res2 = aggregate_model_results("cloud_mod", "cloud", utts2)

        dummy_manifest = os.path.join(tmpdir, "test_manifest.csv")
        with open(dummy_manifest, "w", encoding="utf-8") as f:
            f.write("path,text\n")

        config = BenchmarkConfig(
            manifest_path=dummy_manifest,
            local_model="local_mod",
            openrouter_models=["cloud_mod"],
            output_dir=tmpdir,
            dry_run=True,
        )

        json_path, md_path = generate_benchmark_reports([res1, res2], config)

        assert os.path.exists(json_path)
        assert os.path.exists(md_path)

        # Verify JSON content
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert "results" in data
            assert len(data["results"]) == 2
            assert data["results"][0]["model_name"] == "local_mod"
            assert data["results"][1]["model_name"] == "cloud_mod"

        # Verify Markdown content
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()
            assert "# Speech-to-Text Model Benchmark Report" in md_content
            assert "## Table 1: Model Accuracy & Performance Summary" in md_content
            assert "`local_mod`" in md_content
            assert "`cloud_mod`" in md_content
            assert "## Table 2: Per-Speaker Accuracy Breakdown" in md_content
            assert "SPK1" in md_content
            assert "SPK2" in md_content
            assert "Top 5 Most Accurate Transcriptions" in md_content
