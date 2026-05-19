# TuneTalk Model Backend — Local Setup Guide

## Prerequisites

- **Python 3.10+**
- **uv** (Python package manager) — [install guide](https://docs.astral.sh/uv/#installation)
- **HuggingFace account** with access token (for model downloads)

## Step 1 — Clone the Repository

```bash
git clone https://github.com/<your-org>/be_model-tunetalk.git
cd be_model-tunetalk
```

> Replace `<your-org>` with the actual GitHub org/username.

## Step 2 — Configure Environment

Copy the example env file and fill in your HuggingFace token:

```bash
cp .env.example .env
```

Edit `.env` and set your token:

```
HF_TOKEN=hf_your_actual_token_here
```

Create your token at: <https://huggingface.co/settings/tokens>

## Step 3 — Install Dependencies

```bash
uv sync
```

This reads `pyproject.toml` + `uv.lock` and installs all packages into a local `.venv`.

## Step 4 — Verify Dataset File

The app expects the Spotify lyrics CSV at:

```
nlp preprocessing/dataset-cleaning/cleaned_spotify_lyrics.csv
```

Ensure this file exists. It is required for playlist generation.

## Step 5 — Run the Server

```bash
uv run python run.py
```

The Flask dev server starts at <http://localhost:5001> with debug mode on.

## Step 6 — Test the API

Health check:

```bash
curl http://localhost:5001/api/health
```

Chat endpoint (emotion analysis + playlist):

```bash
curl -X POST http://localhost:5001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user123", "message": "I feel really down today, everything seems hopeless."}'
```

Playlist by emotion:

```bash
curl -X POST http://localhost:5001/api/playlist \
  -H "Content-Type: application/json" \
  -d '{"emotion_label": "sadness", "n": 3}'
```

Swagger UI is available at: <http://localhost:5001/apidocs>

## Endpoints

| Method | Path            | Description                      |
| ------ | --------------- | -------------------------------- |
| GET    | `/api/health`   | Health check                     |
| POST   | `/api/chat`     | Analyze emotion, return playlist |
| POST   | `/api/playlist` | Get playlist by emotion label    |

## Notes

- On first run, the app downloads two HuggingFace models (~500 MB each). Subsequent starts are fast.
- The models are cached under the HuggingFace hub cache directory (typically `~/.cache/huggingface/`).
- Port 5001 must be free. Change it in `run.py` if needed.
