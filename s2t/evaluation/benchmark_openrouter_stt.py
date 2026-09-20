#!/usr/bin/env python3
"""Benchmark OpenRouter transcription models on a held-out speech manifest."""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import jiwer
import pandas as pd

from s2t.common import PROJECT_ROOT, corpus_cer, corpus_wer, normalize_text


OPENROUTER_URL = "https://openrouter.ai/api/v1/audio/transcriptions"
OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/auth/key"
DEFAULT_MODELS = ["microsoft/mai-transcribe-2"]


def load_openrouter_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    env_path = PROJECT_ROOT / ".env"
    if env_path.is_file():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() == "OPENROUTER_API_KEY":
                return value.strip().strip('"').strip("'")
    raise RuntimeError("OPENROUTER_API_KEY was not found in the environment or .env")


def resolve_audio(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def validate_openrouter_key(api_key: str) -> None:
    """Fail once before fan-out if the OpenRouter credential is invalid."""
    request = Request(
        OPENROUTER_KEY_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(
            f"OpenRouter credential validation failed (HTTP {exc.code}): {detail}"
        ) from exc
    except (URLError, TimeoutError) as exc:
        # This metadata endpoint can be unavailable even while inference works.
        # The transcription call will still provide authoritative auth errors.
        print(f"Warning: OpenRouter key metadata check unavailable ({exc}); continuing.", flush=True)
        return
    if not isinstance(payload, dict) or "data" not in payload:
        raise RuntimeError("OpenRouter credential validation returned an unexpected response")


def model_slug(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


def transcribe_once(api_key: str, model: str, row: dict[str, Any]) -> dict[str, Any]:
    audio_path = resolve_audio(str(row["path"]))
    audio_b64 = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    payload = {
        "model": model,
        "input_audio": {"data": audio_b64, "format": "wav"},
        "language": "en",
    }
    request = Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost/personal-whisper-benchmark",
            "X-Title": "Personal Whisper Fine-Tuning Benchmark",
        },
        method="POST",
    )

    started = time.monotonic()
    with urlopen(request, timeout=120) as response:
        result = json.loads(response.read().decode("utf-8"))
    elapsed = time.monotonic() - started
    text = result.get("text")
    if not isinstance(text, str):
        raise RuntimeError(f"OpenRouter response for {row['utt_id']} has no text field")
    return {
        "utt_id": row["utt_id"],
        "sentence_id": row["sentence_id"],
        "speaker": row["speaker"],
        "duration_s": float(row["duration_s"]),
        "reference": row["text"],
        "hypothesis": text,
        "latency_s": round(elapsed, 4),
        "usage": result.get("usage", {}),
        "generation_id": result.get("id"),
    }


def transcribe_with_retry(
    api_key: str,
    model: str,
    row: dict[str, Any],
    retries: int,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return transcribe_once(api_key, model, row)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            last_error = RuntimeError(f"HTTP {exc.code}: {detail}")
            retryable = exc.code in (408, 409, 429) or exc.code >= 500
            if not retryable:
                break
        except (URLError, TimeoutError, OSError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = exc
        if attempt < retries:
            time.sleep(min(20, 2 ** (attempt + 1)))
    raise RuntimeError(f"{row['utt_id']}: {last_error}")


def load_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    cached: dict[str, dict[str, Any]] = {}
    if not cache_path.is_file():
        return cached
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        cached[item["utt_id"]] = item
    return cached


def usage_cost(usage: Any) -> float:
    if not isinstance(usage, dict):
        return 0.0
    try:
        return float(usage.get("cost") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def evaluate_model(
    api_key: str,
    model: str,
    rows: list[dict[str, Any]],
    cache_dir: Path,
    workers: int,
    retries: int,
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{model_slug(model)}.jsonl"
    cached = load_cache(cache_path)
    pending = [row for row in rows if row["utt_id"] not in cached]
    print(f"\n{model}: {len(cached)} cached, {len(pending)} requests remaining", flush=True)

    failures: list[str] = []
    if pending:
        with cache_path.open("a", encoding="utf-8") as cache_handle:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(transcribe_with_retry, api_key, model, row, retries): row
                    for row in pending
                }
                completed = 0
                for future in as_completed(futures):
                    row = futures[future]
                    try:
                        item = future.result()
                        cached[item["utt_id"]] = item
                        cache_handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                        cache_handle.flush()
                    except Exception as exc:
                        failures.append(str(exc))
                    completed += 1
                    print(
                        f"{model}: {completed}/{len(pending)} new requests complete "
                        f"({len(failures)} failed)",
                        flush=True,
                    )

    missing = [row["utt_id"] for row in rows if row["utt_id"] not in cached]
    if missing:
        raise RuntimeError(
            f"{model} is incomplete: {len(missing)} missing transcripts. "
            f"First failures: {failures[:3]}"
        )

    ordered = [cached[row["utt_id"]] for row in rows]
    refs = [item["reference"] for item in ordered]
    hyps = [item["hypothesis"] for item in ordered]
    for item in ordered:
        ref_norm = normalize_text(item["reference"])
        hyp_norm = normalize_text(item["hypothesis"]) or "empty"
        item["ref_norm"] = ref_norm
        item["hyp_norm"] = hyp_norm
        item["sample_wer"] = float(jiwer.wer(ref_norm or "empty", hyp_norm))

    return {
        "model": model,
        "n_utts": len(ordered),
        "audio_duration_s": round(sum(item["duration_s"] for item in ordered), 4),
        "wer": round(corpus_wer(refs, hyps) * 100, 4),
        "cer": round(corpus_cer(refs, hyps) * 100, 4),
        "total_latency_s": round(sum(item["latency_s"] for item in ordered), 4),
        "mean_latency_s": round(sum(item["latency_s"] for item in ordered) / len(ordered), 4),
        "reported_cost_usd": round(sum(usage_cost(item.get("usage")) for item in ordered), 8),
        "samples": ordered,
    }


def load_local_result(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    test = report.get("splits", {}).get("test")
    if not test:
        return None
    return {
        "model": report.get("model", "local fine-tuned model"),
        "tag": report.get("tag"),
        "n_utts": test.get("n_utts"),
        "wer": test.get("wer"),
        "cer": test.get("cer"),
        "total_latency_s": test.get("decode_time_s"),
        "reported_cost_usd": 0.0,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Personal Speech-to-Text Benchmark",
        "",
        f"Generated: {report['created_at']}",
        "",
        f"Held-out set: {report['test_utterances']} utterances / {report['test_audio_duration_s'] / 60:.2f} minutes",
        "",
        "| Model | WER | CER | Decode/API time | Reported API cost |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    local = report.get("local_model")
    if local:
        lines.append(
            f"| Local fine-tuned Whisper medium.en | **{local['wer']:.2f}%** | "
            f"{local['cer']:.2f}% | {local['total_latency_s']}s | $0.00 |"
        )
    for result in report["openrouter_models"]:
        lines.append(
            f"| `{result['model']}` | **{result['wer']:.2f}%** | {result['cer']:.2f}% | "
            f"{result['total_latency_s']}s | ${result['reported_cost_usd']:.6f} |"
        )

    lines.extend(["", "## Worst examples", ""])
    for result in report["openrouter_models"]:
        lines.extend([f"### {result['model']}", ""])
        for sample in sorted(result["samples"], key=lambda x: x["sample_wer"], reverse=True)[:5]:
            lines.append(
                f"- `{sample['sentence_id']}` ({sample['sample_wer'] * 100:.1f}%): "
                f"ref “{sample['reference']}” → hyp “{sample['hypothesis']}”"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark OpenRouter STT models")
    parser.add_argument("--manifest", default="data/personal/manifest.csv")
    parser.add_argument("--split", default="test")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--local-report", default="reports/finetuned_medium_en_personal.json")
    parser.add_argument("--output", default="reports/openrouter_personal_stt_benchmark.json")
    args = parser.parse_args()

    manifest_path = (PROJECT_ROOT / args.manifest).resolve()
    frame = pd.read_csv(manifest_path)
    frame = frame[frame["split"] == args.split].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No {args.split!r} rows in {manifest_path}")
    rows = frame.to_dict(orient="records")
    api_key = load_openrouter_key()
    validate_openrouter_key(api_key)
    models = [model.strip() for model in args.models.split(",") if model.strip()]
    cache_dir = PROJECT_ROOT / "reports" / "openrouter_cache" / args.split

    results = [
        evaluate_model(api_key, model, rows, cache_dir, args.workers, args.retries)
        for model in models
    ]
    local_path = (PROJECT_ROOT / args.local_report).resolve() if args.local_report else None
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "split": args.split,
        "test_utterances": len(rows),
        "test_audio_duration_s": round(sum(float(row["duration_s"]) for row in rows), 4),
        "local_model": load_local_result(local_path),
        "openrouter_models": results,
    }
    output_path = (PROJECT_ROOT / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path = output_path.with_suffix(".md")
    write_markdown(markdown_path, report)
    print(f"Saved {output_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved {markdown_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
