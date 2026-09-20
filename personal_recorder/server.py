"""Aiohttp backend server for Personal ARCTIC Recorder."""

import asyncio
from dataclasses import asdict
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from aiohttp import web

from personal_recorder.audio import (
    AudioValidationError,
    resample_and_save_16k,
    validate_wav_bytes,
)
from personal_recorder.prompts import PromptCatalog
from personal_recorder.storage import (
    ConflictError,
    DatasetStorage,
    LeaseConflictError,
    StorageError,
)

logger = logging.getLogger(__name__)


def validate_same_origin(request: web.Request) -> None:
    """Validate request Origin or Referer against Host to protect against cross-site requests."""
    host = request.headers.get("Host", "").strip()
    origin = request.headers.get("Origin")

    if origin:
        parsed = urlparse(origin)
        origin_host = parsed.netloc
        if origin_host and host and origin_host.lower() != host.lower():
            raise web.HTTPForbidden(text="Cross-origin requests are forbidden.")
        return

    referer = request.headers.get("Referer")
    if referer:
        parsed = urlparse(referer)
        ref_host = parsed.netloc
        if ref_host and host and ref_host.lower() != host.lower():
            raise web.HTTPForbidden(text="Cross-origin requests are forbidden.")


class RecorderServer:
    """Aiohttp-based server managing routes, static assets, and background audio conversion."""

    def __init__(
        self,
        storage: DatasetStorage,
        catalog: PromptCatalog,
        static_dir: Optional[Path] = None,
    ):
        self.storage = storage
        self.catalog = catalog
        self.static_dir = (
            static_dir.resolve()
            if static_dir is not None
            else Path(__file__).resolve().parent / "static"
        )
        self.conversion_queue: asyncio.Queue[str] = asyncio.Queue()
        self.worker_task: Optional[asyncio.Task] = None

        self.app = web.Application(client_max_size=50 * 1024 * 1024)  # 50 MB max body
        self._setup_routes()
        self.app.on_startup.append(self._on_startup)
        self.app.on_cleanup.append(self._on_cleanup)

    def _setup_routes(self) -> None:
        self.app.router.add_get("/", self.handle_index)
        self.app.router.add_static("/static", str(self.static_dir), name="static")

        # API routes
        self.app.router.add_get("/api/dataset", self.handle_get_dataset)
        self.app.router.add_get("/api/catalog", self.handle_get_catalog)
        self.app.router.add_get("/api/prompts/{sentence_id}", self.handle_get_prompt)
        self.app.router.add_post("/api/sessions", self.handle_post_session)
        self.app.router.add_post("/api/lease/acquire", self.handle_lease_acquire)
        self.app.router.add_post("/api/lease/heartbeat", self.handle_lease_heartbeat)
        self.app.router.add_post("/api/lease/release", self.handle_lease_release)
        self.app.router.add_post("/api/takes", self.handle_post_take)
        self.app.router.add_post("/api/takes/{take_id}/keep", self.handle_post_keep)
        self.app.router.add_get("/api/takes/{take_id}/audio", self.handle_get_take_audio)
        self.app.router.add_get("/api/takes/{take_id}/status", self.handle_get_take_status)
        self.app.router.add_get("/api/stats", self.handle_get_stats)
        self.app.router.add_get("/api/manifest", self.handle_get_manifest)

    async def _on_startup(self, app: web.Application) -> None:
        # Enqueue pending conversions from storage
        pending = self.storage.get_pending_conversions()
        logger.info("Found %d pending 16 kHz conversions at startup", len(pending))
        for rec in pending:
            await self.conversion_queue.put(rec.take_id)

        self.worker_task = asyncio.create_task(self._conversion_worker())

    async def _on_cleanup(self, app: web.Application) -> None:
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass

    async def _conversion_worker(self) -> None:
        """Background worker consuming pending takes and generating 16 kHz WAVs."""
        logger.info("Audio conversion worker started.")
        while True:
            take_id = await self.conversion_queue.get()
            try:
                take = self.storage.get_take(take_id)
                if take is None:
                    continue

                if take.conversion_status == "ready":
                    continue

                raw_path = Path(take.original_path)
                wav16k_path = Path(take.processed_path)

                if not raw_path.is_file():
                    self.storage.mark_conversion_failed(
                        take_id, f"Raw audio file missing: {raw_path}"
                    )
                    continue

                # Run heavy audio decoding and resampling in default executor
                def do_resample():
                    with open(raw_path, "rb") as f:
                        raw_bytes = f.read()
                    info = validate_wav_bytes(raw_bytes)
                    info_16k = resample_and_save_16k(info.samples, info.sample_rate, wav16k_path)
                    return info_16k.duration_s

                loop = asyncio.get_running_loop()
                dur_16k = await loop.run_in_executor(None, do_resample)
                self.storage.mark_conversion_ready(take_id, dur_16k, wav16k_path)
                logger.info("Take %s converted to 16 kHz (%.2fs)", take_id, dur_16k)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Conversion failed for take %s: %s", take_id, e)
                self.storage.mark_conversion_failed(take_id, str(e))
            finally:
                self.conversion_queue.task_done()

    # -------------------------------------------------------------------------
    # Route Handlers
    # -------------------------------------------------------------------------

    async def handle_index(self, request: web.Request) -> web.Response:
        index_file = self.static_dir / "index.html"
        if not index_file.is_file():
            return web.Response(text="Frontend index.html not found.", status=404)
        return web.FileResponse(index_file)

    async def handle_get_dataset(self, request: web.Request) -> web.Response:
        meta = self.storage.get_dataset_metadata()
        stats = self.storage.get_stats()
        return web.json_response({"metadata": meta, "stats": stats})

    async def handle_get_catalog(self, request: web.Request) -> web.Response:
        with self.storage._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT sentence_id,
                       COUNT(take_id) as take_count,
                       MAX(CASE WHEN keep_decision = 1 THEN take_id ELSE NULL END) as kept_take_id
                FROM takes
                GROUP BY sentence_id;
                """
            ).fetchall()

        takes_info = {r["sentence_id"]: (r["take_count"], r["kept_take_id"]) for r in rows}

        catalog_list = []
        for p in self.catalog.prompts:
            cnt, kept_id = takes_info.get(p.sentence_id, (0, None))
            catalog_list.append(
                {
                    "sentence_id": p.sentence_id,
                    "text": p.text,
                    "order_index": p.order_index,
                    "split": p.split,
                    "take_count": cnt,
                    "is_kept": kept_id is not None,
                    "kept_take_id": kept_id,
                }
            )

        return web.json_response({"prompts": catalog_list})

    async def handle_get_prompt(self, request: web.Request) -> web.Response:
        sentence_id = request.match_info["sentence_id"]
        prompt = self.catalog.get_prompt(sentence_id)
        if prompt is None:
            return web.json_response({"error": "Prompt not found."}, status=404)

        takes = self.storage.get_takes_for_sentence(sentence_id)

        # Previous and next sentences
        idx = prompt.order_index
        prev_id = self.catalog.prompts[idx - 1].sentence_id if idx > 0 else None
        next_id = (
            self.catalog.prompts[idx + 1].sentence_id
            if idx + 1 < len(self.catalog.prompts)
            else None
        )

        return web.json_response(
            {
                "sentence_id": prompt.sentence_id,
                "text": prompt.text,
                "order_index": prompt.order_index,
                "split": prompt.split,
                "prev_sentence_id": prev_id,
                "next_sentence_id": next_id,
                "takes": [t.to_dict() for t in takes],
            }
        )

    async def handle_post_session(self, request: web.Request) -> web.Response:
        validate_same_origin(request)
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body."}, status=400)

        session_id = body.get("session_id")
        if not session_id or not isinstance(session_id, str):
            return web.json_response({"error": "Missing or invalid session_id."}, status=400)

        self.storage.register_session(
            session_id=session_id,
            user_agent=body.get("user_agent"),
            mic_label=body.get("mic_label"),
            mic_settings=body.get("mic_settings"),
        )
        return web.json_response({"status": "registered", "session_id": session_id})

    async def handle_lease_acquire(self, request: web.Request) -> web.Response:
        validate_same_origin(request)
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body."}, status=400)

        client_id = body.get("client_id")
        if not client_id:
            return web.json_response({"error": "Missing client_id."}, status=400)

        ttl = float(body.get("ttl_seconds", 15.0))
        acquired = self.storage.acquire_or_renew_lease(client_id, ttl)
        if not acquired:
            return web.json_response(
                {"error": "Another browser tab is currently controlling recording."},
                status=409,
            )
        return web.json_response({"status": "acquired", "client_id": client_id})

    async def handle_lease_heartbeat(self, request: web.Request) -> web.Response:
        validate_same_origin(request)
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body."}, status=400)

        client_id = body.get("client_id")
        if not client_id:
            return web.json_response({"error": "Missing client_id."}, status=400)

        ttl = float(body.get("ttl_seconds", 15.0))
        renewed = self.storage.acquire_or_renew_lease(client_id, ttl)
        if not renewed:
            return web.json_response(
                {"error": "Lease lost: another tab has taken control."},
                status=409,
            )
        return web.json_response({"status": "renewed", "client_id": client_id})

    async def handle_lease_release(self, request: web.Request) -> web.Response:
        validate_same_origin(request)
        try:
            body = await request.json()
        except Exception:
            body = {}
        client_id = body.get("client_id")
        if client_id:
            self.storage.release_lease(client_id)
        return web.json_response({"status": "released"})

    async def handle_post_take(self, request: web.Request) -> web.Response:
        validate_same_origin(request)
        if not request.content_type.startswith("multipart/"):
            return web.json_response(
                {"error": "Multipart form-data required for take upload."}, status=400
            )

        reader = await request.multipart()
        fields: Dict[str, Any] = {}
        audio_bytes: Optional[bytes] = None

        while True:
            part = await reader.next()
            if part is None:
                break
            if part.name == "audio":
                audio_bytes = await part.read()
            else:
                fields[part.name] = (await part.text()).strip()

        if audio_bytes is None:
            return web.json_response({"error": "Missing audio file part."}, status=400)

        take_id = fields.get("take_id")
        session_id = fields.get("session_id")
        sentence_id = fields.get("sentence_id")

        if not take_id or not session_id or not sentence_id:
            return web.json_response(
                {"error": "take_id, session_id, and sentence_id are required fields."},
                status=400,
            )

        mic_settings = None
        if "mic_settings" in fields:
            try:
                mic_settings = json.loads(fields["mic_settings"])
            except Exception:
                pass

        try:
            rec, is_new = self.storage.save_raw_take(
                take_id=take_id,
                session_id=session_id,
                sentence_id=sentence_id,
                wav_bytes=audio_bytes,
                mic_settings=mic_settings,
                capture_time=fields.get("capture_time"),
            )
        except AudioValidationError as e:
            return web.json_response({"error": f"Audio validation error: {e}"}, status=400)
        except ConflictError as e:
            return web.json_response({"error": str(e)}, status=409)
        except StorageError as e:
            return web.json_response({"error": str(e)}, status=400)

        if is_new:
            await self.conversion_queue.put(take_id)

        return web.json_response(
            {
                "status": "saved",
                "take_id": rec.take_id,
                "sentence_id": rec.sentence_id,
                "duration_s": rec.duration_s,
                "is_new": is_new,
                "conversion_status": rec.conversion_status,
            },
            status=201 if is_new else 200,
        )

    async def handle_post_keep(self, request: web.Request) -> web.Response:
        validate_same_origin(request)
        take_id = request.match_info["take_id"]
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body."}, status=400)

        sentence_id = body.get("sentence_id")
        if not sentence_id:
            return web.json_response({"error": "Missing sentence_id."}, status=400)

        try:
            rec = self.storage.set_kept_take(sentence_id, take_id)
        except StorageError as e:
            return web.json_response({"error": str(e)}, status=400)

        return web.json_response(
            {
                "status": "kept",
                "take_id": rec.take_id,
                "sentence_id": rec.sentence_id,
                "keep_decision": rec.keep_decision,
            }
        )

    async def handle_get_take_audio(self, request: web.Request) -> web.Response:
        take_id = request.match_info["take_id"]
        take = self.storage.get_take(take_id)
        if take is None:
            return web.json_response({"error": "Take not found."}, status=404)

        audio_type = request.query.get("type", "raw")
        if audio_type == "16k":
            if take.conversion_status != "ready":
                return web.json_response(
                    {"error": f"16 kHz audio not ready (status: {take.conversion_status})."},
                    status=404,
                )
            target_path = Path(take.processed_path)
        else:
            target_path = Path(take.original_path)

        if not target_path.is_file():
            return web.json_response({"error": "Audio file missing from disk."}, status=404)

        return web.FileResponse(target_path, headers={"Content-Type": "audio/wav"})

    async def handle_get_take_status(self, request: web.Request) -> web.Response:
        take_id = request.match_info["take_id"]
        take = self.storage.get_take(take_id)
        if take is None:
            return web.json_response({"error": "Take not found."}, status=404)

        return web.json_response(
            {
                "take_id": take.take_id,
                "conversion_status": take.conversion_status,
                "duration_s": take.duration_s,
                "keep_decision": take.keep_decision,
                "error": take.conversion_error,
            }
        )

    async def handle_get_stats(self, request: web.Request) -> web.Response:
        return web.json_response(self.storage.get_stats())

    async def handle_get_manifest(self, request: web.Request) -> web.Response:
        if not self.storage.manifest_path.is_file():
            return web.Response(text="Manifest not yet generated.", status=404)
        return web.FileResponse(
            self.storage.manifest_path,
            headers={"Content-Type": "text/csv; charset=utf-8"},
        )
