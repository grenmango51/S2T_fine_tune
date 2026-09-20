"""Run pipeline stages consistently, stopping immediately on failure."""
import shlex
import subprocess
import time

from s2t.paths import PROJECT_ROOT


def run_cmd(command: list[str], description: str) -> float:
    print(f"\n{description}\nCommand: {shlex.join(command)}", flush=True)
    started = time.monotonic()
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    elapsed = time.monotonic() - started
    if result.returncode:
        raise RuntimeError(f"Step '{description}' failed with exit code {result.returncode}")
    print(f"Completed in {elapsed / 60:.1f} minutes", flush=True)
    return elapsed
