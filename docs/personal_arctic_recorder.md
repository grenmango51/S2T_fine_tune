# Personal ARCTIC recorder

## 1. Overview & Scope

The **Personal ARCTIC Recorder** is a local, lightweight recording system designed for capturing speech in the user's voice against canonical CMU ARCTIC sentences. It optimizes for:
- **Uninterrupted workflow**: Single-screen UI, responsive keyboard shortcuts, and zero-distraction flow.
- **Lossless audio capture**: Native sample rate float32 PCM captured via Web Audio `AudioWorklet`, encoded to mono PCM16 WAV without lossy browser compression.
- **Immediate persistence**: Every take is durably staged in browser IndexedDB before clearing memory and automatically uploaded to the backend with retry backoff.
- **Authoritative backend storage**: Atomically written WAV files, SQLite database for session and take metadata, background 16 kHz resampling via `soxr`, and continuous manifest generation.

> [!IMPORTANT]
> **Capture Inventory vs. Verified Training Data**:
> All recorded takes carry `review_status = "unreviewed"`. Pressing **Enter** to keep a take indicates that the user selected that take for the prompt; it does **not** certify transcript or label verification. The exported `manifest.csv` is a capture inventory. Existing training pipelines are not automatically invoked.

---

## 2. Directory Layout & Storage

The recorder writes all session files into an isolated directory (default `recordings/personal_arctic/` relative to project root), which is gitignored:

```text
recordings/personal_arctic/
├── prompts.json                   # Frozen snapshot of prompt catalog and checksum
├── session.sqlite                 # SQLite database with dataset metadata, sessions, takes, and leases
├── raw/                           # Lossless raw audio uploads
│   └── <sentence_id>/
│       └── <take_id>.wav
├── wav16k/                        # Resampled 16 kHz mono PCM16 WAV files
│   └── <sentence_id>/
│       └── <take_id>.wav
└── manifest.csv                   # Atomically replaced snapshot of selected 16 kHz takes
```

---

## 3. Launching the Backend Server

From the project root (`S2T fine tune/`):

```bash
python -m personal_recorder \
  --host 127.0.0.1 \
  --port 8765 \
  --speaker PERSONAL \
  --output recordings/personal_arctic
```

### CLI Arguments

| Flag | Default | Description |
|---|---|---|
| `--host` | `127.0.0.1` | Host address to bind (loopback only by default for local security). |
| `--port` | `8765` | TCP port for HTTP server and UI. |
| `--speaker` | `PERSONAL` | Configurable speaker identifier for utterance IDs and manifest entries. |
| `--output` | `recordings/personal_arctic` | Output directory (resolved relative to project root). |
| `--prompts` | `docs/cmu_arctic_sources/cmuarctic.data` | Path to canonical CMU ARCTIC prompt text file. |
| `--split-ids` | `None` (optional) | Path to JSON split assignment file (e.g. `data/split_ids.json`). If omitted, all prompts are recorded with `split = "unassigned"`. |

### Dataset Creation & Resumption
- **First Launch**: Automatically parses canonical prompts, calculates source SHA-256 checksum, freezes split assignments, writes `prompts.json`, and creates `session.sqlite`.
- **Subsequent Launches**: Reopens the existing dataset. If conflicting speaker IDs, altered prompt files, or changed split assignments are supplied, the server refuses to launch and prints a diagnostic error.

---

## 4. Remote Access via SSH Port Forwarding

When running the backend on a remote host (e.g., a cloud workstation or server), browser microphone access requires a **Secure Context** (`https://` or `http://localhost`). Browsers strictly disallow microphone recording over unencrypted `http://<remote-ip>:8765`.

To access the recorder from your laptop or local desktop with a microphone:

1. Start the server on the remote machine:
   ```bash
   python -m personal_recorder --host 127.0.0.1 --port 8765
   ```
2. On your local computer, create an SSH tunnel with local port forwarding:
   ```bash
   ssh -N -L 8765:127.0.0.1:8765 <username>@<remote-host>
   ```
3. Open your local browser to:
   ```
   http://localhost:8765
   ```
4. Click **Enable Microphone** to grant microphone permission.

---

## 5. Single-Screen Recording Workflow

### Keyboard Controls & On-Screen Equivalents

| Key | Button | Description |
|---|---|---|
| **Space** | **Record / Stop** | Toggles recording. Starts capture with visual lead-in; stops and stages take to IndexedDB and backend. Suppresses page scroll. |
| **Enter** | **Keep & Advance** | Marks the stopped take as selected for the prompt, queues the keep decision, and advances to the next uncompleted prompt. |
| **R** | **Retake** | Starts a new take for the same sentence without deleting or overwriting previous takes. |
| **P** | **Replay** | Plays back the active take (does not feed audio into microphone input). |
| **Esc** | **Pause Session** | Finalizes any in-progress take, preserves it, and releases microphone hardware tracks. |

### UI Status Indicators

- **Ready**: Standby, microphone stream open, level meter responsive.
- **Recording**: Pulsing red indicator, elapsed timer active, Float32 PCM streaming into worklet.
- **Saved Locally / Uploading**: Audio encoded to WAV and written to browser IndexedDB; background upload underway.
- **Saved on Server**: Backend has confirmed write and SQLite persistence.
- **16k Ready**: Background worker has finished resampling with `soxr` and updated `manifest.csv`.
