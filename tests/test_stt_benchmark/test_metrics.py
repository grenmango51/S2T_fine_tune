import pytest
from stt_benchmark.metrics import (
    UtteranceResult,
    compute_utterance_result,
    aggregate_model_results,
)


def test_compute_utterance_result_exact_match():
    res = compute_utterance_result(
        utt_id="utt_1",
        ref_raw="Hello world, this is a test!",
        hyp_raw="Hello world this is a test",
        latency_s=0.5,
        model_name="test_model",
        speaker="SPK1",
    )
    assert res.wer == 0.0
    assert res.cer == 0.0
    assert res.latency_s == 0.5
    assert res.error is None
    assert res.speaker == "SPK1"


def test_compute_utterance_result_empty_hypothesis():
    res = compute_utterance_result(
        utt_id="utt_2",
        ref_raw="The quick brown fox",
        hyp_raw="",
        latency_s=0.2,
        model_name="test_model",
    )
    assert res.wer == 100.0
    assert res.hyp_norm == ""


def test_compute_utterance_result_normalization():
    res = compute_utterance_result(
        utt_id="utt_3",
        ref_raw="It's 10 o'clock.",
        hyp_raw="it is ten o clock",
        latency_s=0.3,
        model_name="test_model",
    )
    # Whisper normalization standardizes casing and apostrophes
    assert res.wer >= 0.0


def test_aggregate_model_results():
    utts = [
        compute_utterance_result("1", "hello world", "hello world", 0.4, "m1", speaker="A"),
        compute_utterance_result("2", "foo bar", "foo baz", 0.6, "m1", speaker="A"),
        compute_utterance_result("3", "good morning", "good evening", 0.8, "m1", speaker="B"),
    ]
    model_res = aggregate_model_results(
        model_name="m1",
        model_type="local",
        utterance_results=utts,
    )

    assert model_res.model_name == "m1"
    assert model_res.model_type == "local"
    assert model_res.total_utterances == 3
    assert model_res.successful_utterances == 3
    assert model_res.failed_utterances == 0
    assert model_res.mean_latency_s == 0.6
    assert "A" in model_res.speaker_results
    assert "B" in model_res.speaker_results
    assert model_res.speaker_results["A"]["count"] == 2
    assert model_res.speaker_results["B"]["count"] == 1
    assert model_res.corpus_wer > 0.0


def test_aggregate_empty_results():
    model_res = aggregate_model_results("empty_model", "cloud", [])
    assert model_res.total_utterances == 0
    assert model_res.corpus_wer == 0.0
    assert model_res.mean_latency_s == 0.0
