# vllm-cli-tui — design

A terminal UI companion for [vLLM](https://github.com/vllm-project/vllm), built with
Python and [Textual](https://textual.textualize.io/). Sibling of
[ollama-cli-tui](https://github.com/elmisi/ollama-cli-tui), which it borrows its
structure and interaction style from.

## Why

vLLM is an engine, not a manager. One process serves one model, chosen at startup;
the OpenAI-compatible API it exposes has no endpoint to search, download or install
anything, and no official UI exists. The workflow around a vLLM box is therefore
split across tools: hunting models on the HuggingFace website, downloading them by
hand, and reading raw Prometheus text to know what the server is doing.

This tool folds that workflow into one keyboard-first TUI, meant to run **on the
machine that serves vLLM** (typically over SSH), exactly like ollama-cli-tui runs
next to Ollama.

## What it deliberately does NOT do

**Process lifecycle.** Starting, stopping and supervising vLLM servers belongs to
whatever the host already uses — systemd, container runtimes, site-specific
tooling. Owning that would duplicate infrastructure the box already has; showing
what is *running* is the Monitor tab's job, controlling it is not. This is the
main scope difference from tools like `llmserve`, which is a launcher.

The one step toward control that is already in: the **serve plan** (step 1 of
the lifecycle roadmap, below). It is read-only — it shows the exact command and
what would go wrong, then stops there.

## The three tabs

### 1. Search — find and download models

- Query the HuggingFace Hub REST API (`/api/models`) — real JSON, no scraping:
  search text, `pipeline_tag=text-generation`, sorted by downloads.
- Filter by quantization, inferred from repo id and tags: AWQ, GPTQ, FP8, INT4/8,
  unquantized. GGUF repos are flagged since vLLM's GGUF support is limited.
- Model detail before committing: file list with sizes (`?blobs=true`), total
  download size, gated flag.
- Download into the standard HuggingFace cache via `huggingface_hub`'s
  `snapshot_download`, in a background worker with a progress screen (bytes on
  disk vs expected total). The cache is where `vllm serve <repo-id>` looks first,
  so a completed download is immediately serveable with no extra step. Interrupted
  downloads resume — that is why `huggingface_hub` is a dependency rather than a
  hand-rolled urllib downloader: resume and gated-repo auth are exactly the wheels
  not worth reinventing. Token comes from the standard HF locations (env var or
  `huggingface-cli login`); the tool never stores it.

### 2. Models — what is on disk

- The HuggingFace cache, listed via `huggingface_hub.scan_cache_dir()`: repo id,
  size on disk, last accessed.
- Plus any extra directories listed in the config (for hosts that keep models in
  a custom folder): one entry per subdirectory, with recursive size.
- Delete with confirmation. Cache entries are deleted through the scan result's
  revision API; extra-dir entries with `shutil.rmtree`.
- `s` opens the **serve plan** for the selected row (`lifecycle.py`): the exact
  `vllm serve <repo-id-or-path> --port N [--max-model-len M]` command, plus
  pre-flight checks — serveability (reused from the scan), size on disk, vllm
  binary on PATH (fallback `python -m vllm`), port availability (bind test),
  VRAM headroom (weights ×1.25 vs `nvidia-smi` free). Read-only by design.

### Lifecycle roadmap (behind the `lifecycle` config, default `off`)

1. **Serve plan preview** — done, read-only (above).
2. **`systemd` mode** — the TUI generates and drives user units / `systemd-run`
   transient scopes: the host supervisor stays the owner of the process.
3. **`direct` mode** — the TUI spawns `vllm serve` itself (setsid, state file,
   SIGTERM-with-drain via `/pause`, adoption of foreign servers), for hosts without
   systemd.

### 3. Monitor — what the servers are doing

- Polls each configured endpoint (default `http://localhost:8000`) every 5s:
  - `GET /v1/models` → served model id and max context length
  - `GET /metrics` → requests running / waiting, KV-cache usage, prompt and
    generation token totals; token/s derived by diffing successive samples
- One row per endpoint; unreachable endpoints show as down rather than erroring.
- The Prometheus text parser is a small pure function of our own (the format is
  four regexes worth of simple), so the tab needs no extra dependency.

## Configuration

`~/.config/vllm-tui/config.json`, created on first run:

```json
{
  "endpoints": ["http://localhost:8000"],
  "extra_model_dirs": []
}
```

Nothing machine-specific is baked in: defaults are localhost and the standard HF
cache, everything else is config.

## Architecture

Cloned from ollama-cli-tui: `run.py` dev entry point → `src/vllm_tui/app.py`
(Textual app, tabs 1/2/3) → `widgets/` (one view per tab) → `screens/` (modal
dialogs: confirm, model detail, download progress, serve plan). All network and disk work in
`hf_client.py`, `local_models.py`, `metrics_client.py`, `config.py`, `lifecycle.py` — pure
"data in / data out" modules with no Textual imports, so they are unit-testable.
Workers keep the event loop free; UI updates via messages.

Dependencies: `textual`, `huggingface_hub`. Python 3.10+.

## Testing

Unlike its sibling, this project has a pytest suite from day one, covering the
pure layer: HF JSON parsing and quantization inference, the Prometheus parser and
rate derivation, config load/save/defaults, local-dir scanning and size
formatting. The TUI itself is exercised by hand (Textual pilot + a real terminal),
as is the live HF API.
