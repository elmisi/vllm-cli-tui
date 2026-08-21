# vLLM CLI TUI

A terminal companion for [vLLM](https://github.com/vllm-project/vllm): find and
download models, see what is on disk, watch your servers. Built with Python and
[Textual](https://textual.textualize.io/). Sibling of
[ollama-cli-tui](https://github.com/elmisi/ollama-cli-tui).

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

## Why This Tool?

vLLM is an engine, not a manager: one process serves one model, and its API has
no way to search, download or list anything. The workflow around a vLLM box ends
up scattered — hunting models on the HuggingFace website, downloading by hand,
reading raw Prometheus text to see what the server is doing. This TUI folds that
into one keyboard-first terminal app, meant to run on the machine that serves
vLLM (typically over SSH).

What it deliberately does **not** do: start or stop vLLM servers. Process
lifecycle belongs to whatever your host already uses (systemd, containers,
site tooling); this tool observes and provisions, it does not supervise.
It does show you *exactly what it would take* to serve a local model —
see the serve plan below. Process control is a planned feature behind the
`lifecycle` config option, default `off`.

## The three tabs

**Monitor [1]** — one row per configured endpoint, refreshed every 5s: served
model, context length, requests running/waiting, KV-cache usage, generation and
prompt tokens/s (derived by diffing the Prometheus counters). Down servers show
as down instead of erroring.

**Models [2]** — everything on disk: the HuggingFace cache (via the library's
own scanner) plus any extra directories you configure, with sizes and last-used
dates. Delete with confirmation. `s` shows the **serve plan** for the selected
row: the exact `vllm serve` command plus pre-flight checks — serveability,
size on disk, vllm binary, port availability, VRAM headroom (via nvidia-smi).
The preview is read-only: it tells you what would go wrong before you run
anything yourself.

**Search [3]** — the HuggingFace Hub, filtered for text-generation: real-time
search, quantization inferred from the repo (AWQ, GPTQ, FP8, INT4/8, GGUF
flagged since vLLM's GGUF support is limited), download counts, gated flag.
Enter shows the file list with sizes *before* any bytes move; `d` downloads into
the standard HF cache — which is exactly where `vllm serve <repo-id>` looks, so
a finished download is immediately serveable. Interrupted downloads resume.

## Installation

```bash
pipx install git+https://github.com/elmisi/vllm-cli-tui.git
vllm-tui
```

Or from source:

```bash
git clone https://github.com/elmisi/vllm-cli-tui.git
cd vllm-cli-tui
./run.py
```

## Configuration

`~/.config/vllm-tui/config.json`:

```json
{
  "endpoints": ["http://localhost:8000"],
  "extra_model_dirs": [],
  "lifecycle": "off"
}
```

- `endpoints` — vLLM servers to monitor
- `extra_model_dirs` — folders (besides the HF cache) where you keep models;
each subdirectory is listed as one model
- `lifecycle` — planned process-control mode: `off` (default, observe-only),
  `systemd` or `direct`. The serve-plan preview does not use it; start/stop
  arrives in a later release.

- `endpoints` — vLLM servers to monitor
- `extra_model_dirs` — folders (besides the HF cache) where you keep models;
  each subdirectory is listed as one model

Gated models (Llama, FLUX, …) need a HuggingFace login: `huggingface-cli login`
or the `HF_TOKEN` environment variable. The tool never stores your token.

## Requirements

- Python 3.10+
- Network access to your vLLM endpoints and to huggingface.co

## License

MIT
