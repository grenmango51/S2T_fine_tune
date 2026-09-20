import sys
import os
import pandas as pd

from .config import config_from_args, parse_args
from .models.openrouter import list_available_openrouter_stt_models
from .report import generate_benchmark_reports
from .runner import load_and_filter_manifest, run_benchmark


def main(args=None):
    parsed = parse_args(args)

    # 1. Check for --list-models
    if parsed.list_models:
        print("\nFetching available Speech-to-Text / Audio models from OpenRouter...")
        models = list_available_openrouter_stt_models(
            api_key=parsed.api_key or os.getenv("OPENROUTER_API_KEY"),
            base_url=parsed.base_url,
        )
        if not models:
            print("No transcription models found or could not reach OpenRouter API.")
            return 0

        print(f"\nFound {len(models)} speech/transcription capable model(s):")
        print("-" * 80)
        print(f"{'Model ID':<45} {'Context':<10} {'Name'}")
        print("-" * 80)
        for m in sorted(models, key=lambda x: x.get("id", "")):
            mid = m.get("id", "")
            ctx = str(m.get("context_length", "N/A"))
            name = m.get("name", "")
            print(f"{mid:<45} {ctx:<10} {name}")
        print("-" * 80)
        return 0

    try:
        config = config_from_args(parsed)
    except Exception as e:
        print(f"Configuration Error: {e}", file=sys.stderr)
        return 1

    # 2. Check for --dry-run
    if config.dry_run:
        print("\n=== DRY RUN VALIDATION ===")
        try:
            df = load_and_filter_manifest(config)
            print(f"Manifest:          {config.manifest_path} (VALID)")
            print(f"Selected Samples:  {len(df)} utterances")
            if config.split:
                print(f"Split Filter:      {config.split}")
            if config.speaker:
                print(f"Speaker Filter:    {config.speaker}")

            # Check duration column if present
            if "duration_s" in df.columns:
                total_dur_s = df["duration_s"].sum()
                print(f"Total Duration:    {total_dur_s:.1f}s ({total_dur_s / 60.0:.2f} mins)")

            print("\nModels to benchmark:")
            if config.local_model:
                print(f"  [Local]  {config.local_model} on {config.device}")
            else:
                print("  [Local]  None")

            for m in config.openrouter_models:
                print(f"  [Cloud]  OpenRouter: {m}")

            print(f"\nOutput Directory:  {config.output_dir}")
            print("Dry run completed successfully. All configurations and inputs are valid.")
            return 0
        except Exception as e:
            print(f"Dry run failed: {e}", file=sys.stderr)
            return 1

    # 3. Run Benchmark
    try:
        results = run_benchmark(config)
        json_path, md_path = generate_benchmark_reports(results, config)
        print("\n=======================================================")
        print("   BENCHMARK COMPLETED SUCCESSFULLY!")
        print(f"   Markdown Report: {md_path}")
        print(f"   JSON Summary:    {json_path}")
        print("=======================================================\n")
        return 0
    except Exception as e:
        print(f"\nBenchmark Execution Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
