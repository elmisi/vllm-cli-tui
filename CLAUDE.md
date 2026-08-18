# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

vllm-cli-tui is a terminal companion for vLLM: HuggingFace model search and
download, local disk inventory, live server monitoring. Python + Textual,
structure cloned from its sibling ollama-cli-tui. Dependencies: `textual`,
`huggingface_hub`. Python 3.10+. Public repository: everything in English, no
machine-specific defaults (localhost and the standard HF cache only).

**Scope guard:** the tool never starts or stops vLLM processes — lifecycle
belongs to the host's own tooling. Search/download/observe only.

## Development Commands

```bash
./run.py                                   # run from source
PYTHONPATH=src .venv/bin/python -m pytest tests/ -v   # test suite
PYTHONPATH=src .venv/bin/python -m pytest tests/test_hf_client.py -k quant -v   # one file / one test
./run.py --version
```

## Architecture

`run.py` → `src/vllm_tui/app.py:main()` → `VllmTUI` (Textual App, three tabs)

- **app.py** — tabs, tab switching (which must move focus into the new pane:
  keys otherwise fall through to App bindings, and the first letter of a
  search that starts with "q" quits the app)
- **widgets/** — one view per tab (MonitorView, ModelsView, SearchView)
- **screens/** — modal dialogs (ConfirmDialog, ModelDetailScreen,
  DownloadProgressScreen), all returning results via `dismiss()`
- **hf_client.py / metrics_client.py / local_models.py / config.py / downloads.py**
  — the pure layer: data in / data out, no Textual imports, unit-tested

### Conventions

- Fetching is separated from parsing so parsers are testable offline
- Network and disk work run in thread workers; UI updates via `call_from_thread`
- Never name a widget method `_render` — it collides with Textual's
  `Widget._render()` and blows up the screen (bitten twice across projects)
- Downloads go through `snapshot_download` (resume, auth); progress is read
  from bytes on disk vs the expected total from the API, and tqdm is disabled
  because it scribbles on the Textual screen
- vLLM metric names differ between engine versions: prefer
  `vllm:kv_cache_usage_perc`, fall back to `vllm:gpu_cache_usage_perc`

### Serveability (can vLLM load this?)

Three checks, one principle: **certain noes come from the repo's format;
architecture support is only ever "✓ known" or "? unknown", never "no"** —
`SUPPORTED_ARCHITECTURES` in `hf_client.py` is a deliberate snapshot of vLLM's
registry (unqueryable live), so an absent entry means "?", and the list needs
occasional additions as vLLM releases.

- Search rows: `hit_verdict()` — architectures arrive in the search response
  itself via the Hub's `expand[]` params (zero extra round trips), but the API
  rejects `expand` combined with `sort`, so results are re-sorted client-side.
  MLX repos carry a valid transformers config yet Apple-format weights: the
  library check must run before the architecture check.
- Search detail: `detail_flags()` — GGUF-only / adapter-only / config-less
  flags from the file list, shown before any bytes move.
- Local disk: `probe_dir_serveability()` in `local_models.py` — same idea from
  files on disk; Models tab hides non-serveable entries by default.

### Testing

pytest covers the pure layer (parsers, config, disk scanning, rate math). The
TUI is verified by hand under a real terminal (tmux). When driving the TUI
with automation against real data: never send a confirmation key in the same
keystroke chain as the destructive action — capture and read the dialog first —
and isolate ALL state the tool touches (`XDG_CONFIG_HOME` *and* `HF_HOME`).
