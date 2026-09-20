"""End-to-end HTTP and API tests for RecorderServer."""

import asyncio
import io
from pathlib import Path
import tempfile
from unittest import IsolatedAsyncioTestCase
from urllib.parse import urlparse

from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer
import numpy as np
import soundfile as sf

from personal_recorder.prompts import PromptCatalog, PromptEntry
from personal_recorder.server import RecorderServer
from personal_recorder.storage import DatasetStorage


def _make_wav_bytes(duration_s: float = 0.5, sr: int = 16000, freq: float = 440.0) -> bytes:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    sig = (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, sig, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class TestRecorderServer(IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.tmp_dir.name)

        self.catalog = PromptCatalog(
            source_sha256="mock_server_sha",
            split_provenance="unassigned",
            prompts=[
                PromptEntry("arctic_a0001", "First prompt text", 0, "unassigned"),
                PromptEntry("arctic_a0002", "Second prompt text", 1, "unassigned"),
            ],
        )

        self.storage = DatasetStorage(
            self.output_dir, self.catalog, speaker="TEST_SPK", project_root=self.output_dir
        )
        self.recorder_server = RecorderServer(self.storage, self.catalog)
        self.test_server = TestServer(self.recorder_server.app)
        self.client = TestClient(self.test_server)
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        self.tmp_dir.cleanup()

    async def test_get_dataset_and_catalog(self):
        resp = await self.client.get("/api/dataset")
        assert resp.status == 200
        data = await resp.json()
        assert data["metadata"]["speaker"] == "TEST_SPK"
        assert data["stats"]["total_prompts"] == 2

        cat_resp = await self.client.get("/api/catalog")
        assert cat_resp.status == 200
        cat_data = await cat_resp.json()
        assert len(cat_data["prompts"]) == 2
        assert cat_data["prompts"][0]["sentence_id"] == "arctic_a0001"

    async def test_writer_lease_endpoints(self):
        # Acquire lease
        resp = await self.client.post(
            "/api/lease/acquire",
            json={"client_id": "tab_1", "ttl_seconds": 10},
        )
        assert resp.status == 200

        # Conflicting acquisition by tab 2
        resp2 = await self.client.post(
            "/api/lease/acquire",
            json={"client_id": "tab_2", "ttl_seconds": 10},
        )
        assert resp2.status == 409

        # Heartbeat tab 1
        resp_hb = await self.client.post(
            "/api/lease/heartbeat",
            json={"client_id": "tab_1", "ttl_seconds": 10},
        )
        assert resp_hb.status == 200

        # Release tab 1
        resp_rel = await self.client.post(
            "/api/lease/release",
            json={"client_id": "tab_1"},
        )
        assert resp_rel.status == 200

        # Tab 2 can acquire now
        resp_tab2 = await self.client.post(
            "/api/lease/acquire",
            json={"client_id": "tab_2", "ttl_seconds": 10},
        )
        assert resp_tab2.status == 200

    async def test_upload_take_and_background_conversion(self):
        wav_bytes = _make_wav_bytes(duration_s=0.5, sr=44100)

        # Upload take
        data = FormData()
        data.add_field("audio", wav_bytes, filename="take_1.wav", content_type="audio/wav")
        data.add_field("take_id", "take_1")
        data.add_field("session_id", "sess_1")
        data.add_field("sentence_id", "arctic_a0001")

        resp = await self.client.post("/api/takes", data=data)
        assert resp.status == 201
        res_data = await resp.json()
        assert res_data["status"] == "saved"
        assert res_data["take_id"] == "take_1"

        # Wait briefly for background conversion queue to finish
        for _ in range(50):
            await asyncio.sleep(0.05)
            status_resp = await self.client.get("/api/takes/take_1/status")
            status_data = await status_resp.json()
            if status_data.get("conversion_status") == "ready":
                break

        assert status_data["conversion_status"] == "ready"

        # Test audio streaming
        raw_audio_resp = await self.client.get("/api/takes/take_1/audio?type=raw")
        assert raw_audio_resp.status == 200
        assert raw_audio_resp.headers["Content-Type"] == "audio/wav"

        proc_audio_resp = await self.client.get("/api/takes/take_1/audio?type=16k")
        assert proc_audio_resp.status == 200

        # Test keep decision
        keep_resp = await self.client.post(
            "/api/takes/take_1/keep",
            json={"sentence_id": "arctic_a0001"},
        )
        assert keep_resp.status == 200
        keep_data = await keep_resp.json()
        assert keep_data["status"] == "kept"

        # Check manifest
        man_resp = await self.client.get("/api/manifest")
        assert man_resp.status == 200
        manifest_text = await man_resp.text()
        assert "TEST_SPK_arctic_a0001" in manifest_text
        assert "unreviewed" in manifest_text

    async def test_cross_origin_rejected(self):
        resp = await self.client.post(
            "/api/lease/acquire",
            headers={"Origin": "https://malicious-site.com"},
            json={"client_id": "evil_client"},
        )
        assert resp.status == 403
