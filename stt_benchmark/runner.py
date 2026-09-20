import json
import os
import pathlib
from typing import List, Optional
import pandas as pd
from tqdm import tqdm

import s2t.common as common
from .config import BenchmarkConfig
from .metrics import ModelResults, aggregate_model_results, compute_utterance_result
from .models.base import STTModel
from .models.local_whisper import LocalWhisperModel
from .models.openrouter import OpenRouterSTTModel


def load_and_filter_manifest(config: BenchmarkConfig) -> pd.DataFrame:
    """Load manifest CSV and apply split, speaker, and sample limits."""
    manifest_path = pathlib.Path(config.manifest_path)
    if not manifest_path.is_absolute():
        manifest_path = common.PROJECT_ROOT / manifest_path

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)

    required_cols = ["path", "text"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Manifest {manifest_path} must contain column '{col}'")

    if "utt_id" not in df.columns:
        df["utt_id"] = [
            os.path.splitext(os.path.basename(str(p)))[0] or f"utt_{i}"
            for i, p in enumerate(df["path"])
        ]

    if "speaker" not in df.columns:
        df["speaker"] = None

    if config.split and "split" in df.columns:
        df = df[df["split"].astype(str).str.lower() == config.split.lower()]

    if config.speaker and "speaker" in df.columns:
        df = df[df["speaker"].astype(str).str.upper() == config.speaker.upper()]

    if config.max_samples and config.max_samples > 0:
        df = df.iloc[: config.max_samples]

    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No records found matching filters in manifest: {manifest_path}")

    # Resolve audio paths
    resolved_paths = []
    for p in df["path"]:
        p_path = pathlib.Path(p)
        if not p_path.is_absolute():
            # Check relative to manifest directory or PROJECT_ROOT
            if (manifest_path.parent / p_path).exists():
                resolved_paths.append(str(manifest_path.parent / p_path))
            elif (common.PROJECT_ROOT / p_path).exists():
                resolved_paths.append(str(common.PROJECT_ROOT / p_path))
            else:
                resolved_paths.append(str(common.PROJECT_ROOT / p_path))
        else:
            resolved_paths.append(str(p_path))

    df["resolved_path"] = resolved_paths
    return df


def safe_tag(name: str) -> str:
    """Make model name safe for filenames."""
    return name.replace("/", "_").replace("\\", "_").replace(":", "_").replace(" ", "_")


def run_benchmark(config: BenchmarkConfig) -> List[ModelResults]:
    """Execute benchmarking across local and cloud models."""
    os.makedirs(config.output_dir, exist_ok=True)
    df = load_and_filter_manifest(config)

    print(f"\n=======================================================")
    print(f"   STARTING STT BENCHMARK ({len(df)} utterances)")
    print(f"   Manifest: {config.manifest_path}")
    print(f"   Output:   {config.output_dir}")
    print(f"=======================================================\n")

    models: List[STTModel] = []

    # 1. Local Whisper Model
    if config.local_model:
        models.append(
            LocalWhisperModel(
                model_name_or_path=config.local_model,
                device=config.device,
                language=config.language,
            )
        )

    # 2. Cloud OpenRouter Models
    for slug in config.openrouter_models:
        models.append(
            OpenRouterSTTModel(
                model_slug=slug,
                api_key=config.openrouter_api_key or "",
                base_url=config.openrouter_base_url,
                language=config.language,
                concurrency=config.concurrency,
            )
        )

    all_results: List[ModelResults] = []

    for model in models:
        tag = safe_tag(model.name)
        cached_result_file = os.path.join(config.output_dir, f"{tag}_results.json")

        if config.resume and os.path.exists(cached_result_file):
            print(f"Skipping already completed model '{model.name}' (loaded from {cached_result_file})")
            try:
                with open(cached_result_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                all_results.append(ModelResults.from_dict(cached_data))
                continue
            except Exception as e:
                print(f"Warning: Could not read cached file {cached_result_file}: {e}. Re-running.")

        print(f"\n--- Benchmarking Model: {model.name} ({model.model_type.upper()}) ---")

        utterance_results = []
        batch_size = config.batch_size if model.model_type == "local" else config.concurrency

        pbar = tqdm(total=len(df), desc=f"{model.name[:25]:<25}")

        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i : i + batch_size]
            batch_paths = batch_df["resolved_path"].tolist()

            batch_transcriptions = model.transcribe_batch(batch_paths)

            for (_, row), trans_res in zip(batch_df.iterrows(), batch_transcriptions):
                utt_res = compute_utterance_result(
                    utt_id=str(row["utt_id"]),
                    ref_raw=str(row["text"]),
                    hyp_raw=trans_res.text,
                    latency_s=trans_res.latency_s,
                    model_name=model.name,
                    speaker=str(row["speaker"]) if pd.notna(row["speaker"]) else None,
                    cost_usd=trans_res.cost_usd,
                    error=trans_res.error,
                )
                utterance_results.append(utt_res)
                pbar.update(1)

        pbar.close()

        model_res = aggregate_model_results(
            model_name=model.name,
            model_type=model.model_type,
            utterance_results=utterance_results,
        )

        print(
            f"Result for {model.name}: Corpus WER = {model_res.corpus_wer:.2f}%, "
            f"CER = {model_res.corpus_cer:.2f}%, Avg Latency = {model_res.mean_latency_s:.2f}s"
        )

        # Save intermediate checkpoint
        with open(cached_result_file, "w", encoding="utf-8") as f:
            json.dump(model_res.to_dict(), f, indent=2)

        all_results.append(model_res)

    return all_results
