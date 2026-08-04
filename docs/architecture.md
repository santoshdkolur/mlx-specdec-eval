# `mlx-specdec-eval` Architecture & System Design

This document details the software architecture, modular decomposition, data flow, and design patterns of `mlx-specdec-eval`.

---

## Architectural Overview Diagram

```text
                               +----------------------------------+
                               |        CLI Interface             |
                               |        (spec_bench/cli.py)       |
                               +----------------+-----------------+
                                                |
        +-----------------------+---------------+-----------------------+
        |                       |               |                       |
        v                       v               v                       v
+---------------+       +---------------+ +---------------+     +---------------+
| Environment   |       | Pre-Flight    | | Local Model   |     | Benchmark     |
| Diagnostic    |       | Validator     | | Cache Manager |     | Runner        |
| (validator.py)|       | (validator.py)| | (models/)     |     | (engine_mlx)  |
+---------------+       +---------------+ +---------------+     +-------+-------+
                                                                        |
                                                                        v
                                                                +---------------+
                                                                | Metrics &     |
                                                                | Task Breakdown|
                                                                | (metrics.py)  |
                                                                +-------+-------+
                                                                        |
                                                                        v
                                                                +---------------+
                                                                | TUI & Report  |
                                                                | Exporters     |
                                                                | (reporter.py) |
                                                                +---------------+
```

---

## Component Deep Dive

### 1. `spec_bench/cli.py` (Control & Entry Point)
Powered by `click` and `rich`, providing structured subcommands:
- `spec-bench check-env`: System diagnostics (Python version, `.venv`, Darwin OS, Metal acceleration).
- `spec-bench validate`: Pre-flight compatibility & RAM footprint checks.
- `spec-bench run`: Benchmark generation loops with multi-$K$ parameter sweeps.
- `spec-bench prompts`: Inspection of evaluation prompts suite.
- `spec-bench report`: Post-processing saved JSON logs into HTML or Markdown reports.
- `spec-bench clean`: Artifacts and optional Hugging Face model cache cleanup (`--all`).
- **Hugging Face Cache Management:** Uses default system Hugging Face cache (`~/.cache/huggingface/hub/`) to load previously downloaded weights instantly without re-downloading.


### 2. `spec_bench/validator.py` (Fail-Fast Verification Engine)
- `validate_system_environment()`: Verifies Python 3.10+, virtual environment activation, macOS Darwin platform, and Apple Silicon Metal backend.
- `validate_tokenizer_compatibility()`: Downloads lightweight tokenizer configs (~17MB), verifies vocabulary length identity, special token IDs (`bos`, `eos`, `pad`), and runs sample text identity encoding sweeps.
- `validate_memory_footprint()`: Queries Hugging Face Hub metadata API (`HfApi`) or local disk sizes to calculate total model + KV cache memory requirements, alerting users if available RAM (`psutil`) is exceeded.

### 3. `spec_bench/engine_mlx.py` (Native MLX Execution Engine)
- Loads target and draft models into Apple Silicon Metal Unified Memory via `mlx_lm.load`.
- Configures sampler functions (`make_sampler(temp)`) to support greedy and stochastic sampling without deprecation errors.
- Executes baseline (target-only) and speculative (target + draft) generation loops.
- **Speculative Tracking State Machine**: Operates a state machine over `stream_generate` to count accepted tokens (where `chunk.from_draft` is `True`) and proposed draft tokens ($\min(\text{remaining}, K)$ added once per round).
- Tracks peak Metal memory allocation (`mx.metal.get_peak_memory()`) and elapsed wall-clock latency.

### 4. `spec_bench/metrics.py` (Quantitative Analysis & Domain Breakdown)
- Computes Acceptance Rate $\alpha = \frac{N_{\text{accepted}}}{N_{\text{proposed}}}$ directly from state machine outputs.
- Computes Tokens Per Second ($\text{TPS}$) for baseline and speculative modes.
- Computes Speedup Ratio $S = \frac{\text{TPS}_{\text{speculative}}}{\text{TPS}_{\text{baseline}}}$.
- **Output Match Verification:** Evaluates string equality (`exact_match_rate`) between baseline target and speculative outputs.
- **Task Domain Breakdown:** Groups benchmark runs dynamically using the explicit prompt category metadata to report category-level TPS, acceptance rate $\alpha$, and speedup $S$.

### 5. `spec_bench/reporter.py` (Visualization & Exporters)
- **Rich TUI Tables:** Renders formatted terminal tables for overall parameter sweeps and per-task domain breakdowns.
- **JSON Exporter (`export_json`):** Saves full raw execution traces, prompt texts, and per-token latencies.
- **Markdown Exporter (`export_markdown`):** Generates clean GitHub-style Markdown documents.
- **Visual HTML Exporter (`export_html`):** Renders standalone interactive dashboards featuring Chart.js bar graphs.
