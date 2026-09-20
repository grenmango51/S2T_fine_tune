"""CLI entry point for Personal ARCTIC Recorder."""

import argparse
import logging
from pathlib import Path
import sys

from aiohttp import web

from personal_recorder.prompts import (
    IncompatibleDatasetError,
    PromptParseError,
    SplitValidationError,
    create_catalog_snapshot,
    load_catalog_snapshot,
    verify_dataset_compatibility,
)
from personal_recorder.server import RecorderServer
from personal_recorder.storage import ConflictError, DatasetStorage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("personal_recorder")

from s2t.paths import PROJECT_ROOT


def resolve_path(p: str, base_dir: Path = PROJECT_ROOT) -> Path:
    """Resolve path relative to project root if not absolute."""
    path = Path(p)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Personal ARCTIC Speech Recorder backend and UI server."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind (default: 127.0.0.1 for loopback-only security).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to listen on (default: 8765).",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default="PERSONAL",
        help="Speaker ID (default: PERSONAL).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="recordings/personal_arctic",
        help="Output directory for audio and dataset files (default: recordings/personal_arctic).",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="docs/cmu_arctic_sources/cmuarctic.data",
        help="Path to canonical CMU Arctic prompt file (default: docs/cmu_arctic_sources/cmuarctic.data).",
    )
    parser.add_argument(
        "--split-ids",
        type=str,
        default=None,
        help="Optional path to frozen split JSON file (e.g. data/split_ids.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = resolve_path(args.output)
    prompts_path = resolve_path(args.prompts)
    split_ids_path = resolve_path(args.split_ids) if args.split_ids else None
    prompts_snapshot_path = output_dir / "prompts.json"

    if not prompts_path.is_file():
        logger.error("Canonical prompt file not found: %s", prompts_path)
        sys.exit(1)

    if split_ids_path and not split_ids_path.is_file():
        logger.error("Split file not found: %s", split_ids_path)
        sys.exit(1)

    # Dataset creation or resumption
    try:
        if prompts_snapshot_path.is_file():
            logger.info("Resuming existing dataset at %s", output_dir)
            catalog = load_catalog_snapshot(prompts_snapshot_path)
            verify_dataset_compatibility(catalog, prompts_path, split_ids_path)
        else:
            logger.info("Creating new dataset at %s", output_dir)
            catalog = create_catalog_snapshot(
                prompts_path=prompts_path,
                split_ids_path=split_ids_path,
                snapshot_json_path=prompts_snapshot_path,
            )

        storage = DatasetStorage(
            output_dir=output_dir,
            catalog=catalog,
            speaker=args.speaker,
            project_root=PROJECT_ROOT,
        )
    except (PromptParseError, SplitValidationError) as e:
        logger.error("Prompt/Split configuration error: %s", e)
        sys.exit(1)
    except (IncompatibleDatasetError, ConflictError) as e:
        logger.error("Incompatible dataset configuration: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("Failed to initialize dataset storage: %s", e)
        sys.exit(1)

    server = RecorderServer(
        storage=storage,
        catalog=catalog,
    )

    banner = f"""
================================================================================
 Personal ARCTIC Recorder Server Ready!
--------------------------------------------------------------------------------
 Web UI URL:         http://{args.host}:{args.port}
 Storage directory:  {output_dir}
 Speaker ID:         {args.speaker}
 Total prompts:      {len(catalog.prompts)}
 Split provenance:   {catalog.split_provenance}
--------------------------------------------------------------------------------
 For remote access from a computer with a microphone:
   ssh -N -L {args.port}:127.0.0.1:{args.port} <workspace-host>
 Then navigate to http://localhost:{args.port} in your local browser.
================================================================================
"""
    print(banner)

    web.run_app(
        server.app,
        host=args.host,
        port=args.port,
        print=None,  # We print our own banner
    )


if __name__ == "__main__":
    main()
