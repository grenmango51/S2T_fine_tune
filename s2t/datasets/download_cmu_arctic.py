#!/usr/bin/env python3
"""Download all official archives, verify readability, and materialize transcripts."""
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from s2t.datasets.cmu_arctic import CMU_SPEAKERS, ROOT, SOURCE_URL, speaker_dir


def parse_prompts(path):
    prompts = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r'\s*\(\s*(\S+)\s+"(.*)"\s*\)\s*', line)
        if not match:
            raise ValueError(f"Invalid prompt in {path}: {line!r}")
        sid, text = match.groups()
        if sid in prompts:
            raise ValueError(f"Duplicate prompt {sid} in {path}")
        prompts[sid] = text
    return prompts


def download(speaker):
    archive_dir = ROOT / "corpus" / "cmu_arctic" / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive = archive_dir / f"cmu_us_{speaker.lower()}_arctic.tar.bz2"
    url = SOURCE_URL + "packed/" + archive.name
    if not archive.exists():
        previous = Path("/tmp/cmu_arctic_download") / archive.name
        if previous.exists():
            shutil.copy2(previous, archive)
        else:
            partial = archive.with_suffix(".bz2.part")
            subprocess.run(["curl", "--fail", "--location", "--retry", "5",
                            "--retry-delay", "5", "--connect-timeout", "30",
                            "--max-time", "3600", "--continue-at", "-",
                            "--silent", "--show-error", "--output", str(partial), url], check=True)
            partial.replace(archive)
    # A complete bzip2 scan checks integrity, including the end-of-stream CRC.
    subprocess.run(["bzip2", "--test", str(archive)], check=True)
    dest = speaker_dir(speaker)
    marker = dest / ".verified.json"
    if not marker.exists():
        with tarfile.open(archive, "r:bz2") as tar:
            members = []
            for member in tar.getmembers():
                parts = Path(member.name).parts
                if not parts or parts[0] != dest.name or ".." in parts:
                    raise ValueError(f"Unsafe archive path: {member.name}")
                if member.isfile() and (len(parts) > 1 and parts[1] in {"wav", "etc"}
                                        or Path(member.name).name.lower().startswith(("readme", "copying", "license"))):
                    members.append(member)
            tar.extractall(dest.parent, members=members, filter="data")
        prompts = parse_prompts(dest / "etc" / "txt.done.data")
        canonical = parse_prompts(ROOT / "docs" / "cmu_arctic_sources" / "cmuarctic.data")
        wavs = sorted((dest / "wav").glob("*.wav"))
        if len(wavs) < 400:
            raise ValueError(f"Unexpectedly incomplete speaker {speaker}: {len(wavs)} WAVs")
        transcript_dir = dest / "transcript"
        transcript_dir.mkdir(exist_ok=True)
        filled = []
        for wav in wavs:
            text = prompts.get(wav.stem)
            if text is None:
                text = canonical[wav.stem]
                filled.append(wav.stem)
            (transcript_dir / f"{wav.stem}.txt").write_text(text + "\n", encoding="utf-8")
        with archive.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        marker.write_text(json.dumps({"speaker": speaker, "url": url,
                         "sha256": digest, "utterances": len(wavs),
                         "canonical_prompt_fallbacks": filled}, indent=2) + "\n")
    result = json.loads(marker.read_text())
    print(f"Verified {speaker}: {result['utterances']} utterances", flush=True)
    return result


def main():
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(download, CMU_SPEAKERS))
    path = ROOT / "reports" / "cmu_arctic_downloads.json"
    path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Verified all {len(results)} official archives", flush=True)


if __name__ == "__main__":
    main()
