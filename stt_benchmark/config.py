import argparse
import datetime
import os
import pathlib
from dataclasses import dataclass, field
from typing import List, Optional

import dotenv
import torch


@dataclass
class BenchmarkConfig:
    manifest_path: str
    local_model: Optional[str] = "openai/whisper-medium.en"
    openrouter_models: List[str] = field(default_factory=list)
    openrouter_api_key: Optional[str] = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    output_dir: str = ""
    batch_size: int = 16
    language: str = "en"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_audio_duration_s: float = 300.0
    text_normalizer: str = "whisper"
    split: Optional[str] = None
    speaker: Optional[str] = None
    max_samples: Optional[int] = None
    concurrency: int = 5
    dry_run: bool = False
    resume: bool = False

    def __post_init__(self):
        # Load environment variables (.env in workspace or parent)
        dotenv.load_dotenv()

        if not self.openrouter_api_key:
            self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

        if not self.output_dir:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join("reports", f"benchmark_{timestamp}")

        self.validate()

    def validate(self) -> None:
        if not self.manifest_path:
            raise ValueError("manifest_path must be specified.")

        p = pathlib.Path(self.manifest_path)
        if not p.is_absolute():
            p = pathlib.Path(__file__).resolve().parent.parent / p
        if not p.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")

        if len(self.openrouter_models) > 3:
            raise ValueError(
                f"Maximum of 3 OpenRouter models allowed, got {len(self.openrouter_models)}: {self.openrouter_models}"
            )

        if not self.local_model and not self.openrouter_models:
            raise ValueError("Must specify at least one local or cloud model to benchmark.")

        if self.openrouter_models and not self.openrouter_api_key and not self.dry_run:
            raise ValueError(
                "OPENROUTER_API_KEY must be provided via environment or --api-key when testing cloud models."
            )


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Speech-to-Text Multi-Model Benchmark (Local Whisper + OpenRouter Cloud Models)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/manifest.csv",
        help="Path to manifest CSV containing 'path', 'text', and optional 'speaker', 'split', 'utt_id'",
    )
    parser.add_argument(
        "--local-model",
        type=str,
        default="openai/whisper-medium.en",
        help="Local Whisper Hugging Face ID or directory (e.g. 'models/whisper-medium-en-vi-accent'). Set to 'none' to skip.",
    )
    parser.add_argument(
        "--cloud-models",
        nargs="*",
        default=[],
        help="Up to 3 OpenRouter model slugs (e.g. 'openai/whisper-large-v3' 'openai/gpt-transcribe')",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenRouter API key (defaults to OPENROUTER_API_KEY env var)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://openrouter.ai/api/v1",
        help="OpenRouter API base URL",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory to save benchmark reports (defaults to reports/benchmark_<timestamp>)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for local model inference",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language hint code for transcription",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device for local model ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--max-audio-duration",
        type=float,
        default=300.0,
        help="Maximum audio duration in seconds before skipping or truncating",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Filter manifest to a specific split (e.g. 'test', 'val', 'train')",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default=None,
        help="Filter manifest to a specific speaker ID",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap evaluation to first N samples (useful for rapid testing)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Maximum concurrent cloud API requests",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config, check files and estimate costs without executing inference",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume previous benchmark, skipping models that have already written results JSON",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Query OpenRouter and list available STT transcription models",
    )

    return parser.parse_args(args)


def config_from_args(ns: argparse.Namespace) -> BenchmarkConfig:
    local_mod = None if ns.local_model and ns.local_model.lower() in ("none", "null", "") else ns.local_model
    return BenchmarkConfig(
        manifest_path=ns.manifest,
        local_model=local_mod,
        openrouter_models=ns.cloud_models or [],
        openrouter_api_key=ns.api_key,
        openrouter_base_url=ns.base_url,
        output_dir=ns.output_dir,
        batch_size=ns.batch_size,
        language=ns.language,
        device=ns.device,
        max_audio_duration_s=ns.max_audio_duration,
        split=ns.split,
        speaker=ns.speaker,
        max_samples=ns.max_samples,
        concurrency=ns.concurrency,
        dry_run=ns.dry_run,
        resume=ns.resume,
    )
