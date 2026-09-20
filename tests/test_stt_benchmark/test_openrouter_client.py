import base64
import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch

from stt_benchmark.models.openrouter import OpenRouterSTTModel


@pytest.fixture
def temp_audio_file():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(b"RIFF dummy wav audio content for testing 1234567890")
        tmp_path = f.name
    yield tmp_path
    if os.path.exists(tmp_path):
        os.remove(tmp_path)


def test_prepare_payload(temp_audio_file):
    model = OpenRouterSTTModel(
        model_slug="openai/whisper-large-v3",
        api_key="test-key",
    )
    payload = model._prepare_payload(temp_audio_file)
    assert payload["model"] == "openai/whisper-large-v3"
    assert "input_audio" in payload
    assert payload["input_audio"]["format"] == "wav"
    assert "data" in payload["input_audio"]

    # Verify base64 decoded matches
    decoded = base64.b64decode(payload["input_audio"]["data"])
    assert b"RIFF dummy wav audio content" in decoded


@patch("requests.post")
def test_transcribe_success(mock_post, temp_audio_file):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "text": "Hello this is transcribed audio",
        "usage": {"cost": 0.0025},
    }
    mock_post.return_value = mock_resp

    model = OpenRouterSTTModel(
        model_slug="openai/whisper-large-v3",
        api_key="test-key",
    )
    res = model.transcribe(temp_audio_file)

    assert res.text == "Hello this is transcribed audio"
    assert res.cost_usd == 0.0025
    assert res.error is None
    assert res.latency_s > 0 or res.latency_s == 0.0


@patch("time.sleep")
@patch("requests.post")
def test_transcribe_retry_on_429(mock_post, mock_sleep, temp_audio_file):
    # First call returns 429, second returns 200
    resp_429 = MagicMock()
    resp_429.status_code = 429
    resp_429.text = "Rate limit exceeded"

    resp_200 = MagicMock()
    resp_200.status_code = 200
    resp_200.json.return_value = {
        "text": "Success after retry",
        "usage": {"cost": 0.001},
    }

    mock_post.side_effect = [resp_429, resp_200]

    model = OpenRouterSTTModel(
        model_slug="openai/whisper-large-v3",
        api_key="test-key",
        max_retries=3,
    )
    res = model.transcribe(temp_audio_file)

    assert res.text == "Success after retry"
    assert res.error is None
    assert mock_post.call_count == 2
    mock_sleep.assert_called_once()


@patch("requests.post")
def test_transcribe_batch_concurrency(mock_post, temp_audio_file):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"text": "Transcribed sample"}
    mock_post.return_value = mock_resp

    model = OpenRouterSTTModel(
        model_slug="openai/whisper-large-v3",
        api_key="test-key",
        concurrency=3,
    )
    results = model.transcribe_batch([temp_audio_file, temp_audio_file])

    assert len(results) == 2
    assert results[0].text == "Transcribed sample"
    assert results[1].text == "Transcribed sample"
