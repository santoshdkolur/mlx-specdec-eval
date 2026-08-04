# `mlx-specdec-eval`

A lightweight, local CLI benchmarking tool to profile, measure, and analyze **Speculative Decoding** performance across target and draft LLM pairs on Apple Silicon Macs using **MLX**.

---

## Overview & Problem Statement

Local LLM inference on Apple Silicon is heavily bottlenecked by **Memory Bandwidth**. Auto-regressively generating text requires transferring gigabytes of model weights from Unified Memory to GPU compute cores for *every single token*.

**Speculative Decoding** attempts to solve this by pairing a small, fast **Draft Model** ($M_D$) with a larger **Target Model** ($M_T$). The draft model proposes candidate tokens, and the target model verifies them in a single parallel forward pass.

> [!CAUTION]
> **Speculative decoding does not always speed up inference:**
> - If the token acceptance rate ($\alpha$) is low ($\le 50\%$), candidate tokens are rejected, causing **performance degradation** ($S < 1.0\times$).
> - If the target model is already small & fast (e.g. 3B parameters running at 35 TPS), draft model overhead can exceed parallel verification gains.

**`mlx-specdec-eval` provides a scientific framework to answer:**
- Is speculative decoding actually faster on my specific Mac setup?
- What is the exact token acceptance rate $\alpha$ across Code vs Chat tasks?
- What is the optimal candidate draft length $K$?

---

## Key Features

- **Fail-Fast Pre-Flight Check (`mlx-specdec-eval validate`):** Verifies tokenizer vocabulary identity, special tokens (`bos`, `eos`, `pad`), and estimates Apple Silicon RAM/VRAM footprints before downloading heavy model weights.
- **System Diagnostic Check (`mlx-specdec-eval check-env`):** Verifies Python 3.10+, active virtual environment (`.venv`), macOS Darwin, and Apple Silicon Metal GPU availability.
- **Native MLX Profiling Engine (`mlx-specdec-eval run`):** Profiles acceptance rates ($\alpha$), Tokens Per Second (TPS), Speedup factor ($S$), peak Metal memory, and exact output correctness.
- **Task Domain Breakdown:** Analyzes performance separately across **Code**, **Reasoning**, **Chat**, and **Prose** prompt categories.
- **Multi-$K$ Draft Token Sweeps:** Evaluates $K \in \{3, 5, 7\}$ in a single run to identify peak speedup settings.
- **Visual HTML Dashboard & Exports:** Formats Rich terminal tables and exports interactive Chart.js HTML dashboards, Markdown summaries, and JSON logs.
- **Untracked Session-Local Storage (`./models/`):** Automatically routes Hugging Face model downloads into an untracked local `./models/` directory, preventing Git repository pollution.

---

## Metrics Tracked

| Metric | Definition | Significance |
| :--- | :--- | :--- |
| **Acceptance Rate ($\alpha$)** | $\alpha = \frac{\text{Accepted Draft Tokens}}{\text{Total Proposed Draft Tokens}}$ | Measures prediction alignment between draft & target models ($0\% - 100\%$). |
| **Tokens Per Second (TPS)** | $\text{TPS} = \frac{N_{\text{total\_tokens}}}{T_{\text{wall\_clock}}}$ | Real-time generation throughput. |
| **Speedup Factor ($S$)** | $S = \frac{\text{TPS}_{\text{speculative}}}{\text{TPS}_{\text{baseline}}}$ | Ratio comparing speculative decoding throughput to baseline target generation ($S > 1.0\times$ = Accelerated). |
| **Output Match Rate** | $\text{Exact Match } (EM = 100\%)$ | Verifies token-for-token output identity against baseline target generation. |
| **Peak Memory Footprint** | Measured via `mlx.core.metal.get_peak_memory()` | Peak Metal Unified VRAM allocation in MB. |

---

## Prerequisites & Setup Guide

### System Requirements
- **Hardware:** Apple Silicon Mac (M1, M1 Pro/Max/Ultra, M2, M3, or M4 series).
- **OS:** macOS 13.5+ (Ventura, Sonoma, Sequoia, or later).
- **Python:** Python 3.10, 3.11, or 3.12.

### Installation Steps

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/santoshdkolur/mlx-specdec-eval.git
   cd mlx-specdec-eval
   ```

2. **Create and Activate a Virtual Environment (`.venv`):**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Package Dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -e ".[dev]"
   ```

---

## Quickstart Command Guide

### 1. Verify System Environment
Ensure Apple Silicon Metal GPU and dependencies are configured:
```bash
mlx-specdec-eval check-env
```

### 2. Pre-Flight Model Compatibility Check
Check tokenizer alignment and system RAM footprints without downloading heavy weights:
```bash
mlx-specdec-eval validate \
  --target mlx-community/Llama-3.2-3B-Instruct-4bit \
  --draft mlx-community/Llama-3.2-1B-Instruct-4bit
```

### 3. Run Benchmark Sweeps & Export Visual Dashboard
Execute speculative decoding benchmarks across draft lengths $K \in \{3, 5\}$ and export interactive reports (using `mlx-specdec-eval` or `spec-bench`):
```bash
spec-bench run \
  --target mlx-community/Qwen2.5-7B-Instruct-4bit \
  --draft mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --num-draft-tokens 3,5 \
  --export-html report.html
```

#### Evaluate a Single Prompt ID or Custom Prompt String
```bash
# Evaluate a specific prompt by ID (e.g. code_quicksort, reason_math_proof)
spec-bench run \
  --target mlx-community/Qwen2.5-7B-Instruct-4bit \
  --draft mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --prompt-id code_quicksort \
  --num-draft-tokens 3

# Evaluate a custom prompt string directly
spec-bench run \
  --target mlx-community/Qwen2.5-7B-Instruct-4bit \
  --draft mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --prompt "Write a Python implementation of the Quicksort algorithm" \
  --num-draft-tokens 3
```

### 4. Open Interactive Dashboard
```bash
open report.html
```


### 5. Inspect Evaluation Prompts
List the 20 built-in prompts across Code, Reasoning, Chat, and Prose categories:
```bash
mlx-specdec-eval prompts
```

### 6. Clean Artifacts & Downloaded Models
```bash
# Clean local report files
mlx-specdec-eval clean

# Clean local report files AND untracked downloaded models in ./models/
mlx-specdec-eval clean --all
```

---

## Repository & Output Structure

```text
mlx-specdec-eval/
├── pyproject.toml              # Dependencies & CLI entry point
├── LICENSE                     # MIT License
├── README.md                   # Setup & usage guide
├── .gitignore                  # Ignores .venv, models/, caches, and output reports
├── configs/
│   └── default_prompts.json    # 20 evaluation prompts (Code, Reasoning, Chat, Prose)
├── docs/                       # Architectural & mathematical documentation suite
│   ├── index.md
│   ├── architecture.md
│   ├── environment_setup.md
│   ├── speculative_decoding_math.md
│   ├── preflight_validation.md
│   ├── mlx_profiling_guide.md
│   ├── cli_reference.md
│   └── user_questionnaire_backlog.md
├── models/                     # Untracked local directory for downloaded HF models
├── spec_bench/
│   ├── __init__.py
│   ├── cli.py                  # Click CLI interface
│   ├── validator.py            # Pre-flight environment, tokenizer & memory checks
│   ├── engine_mlx.py           # Native MLX speculative runner
│   ├── metrics.py              # Math engine for alpha, TPS, speedup, domain breakdown
│   └── reporter.py             # Rich TUI formatting & HTML/Markdown/JSON exports
└── tests/                      # 13 automated unit & integration tests (pytest)
    ├── test_validator.py
    ├── test_metrics.py
    └── test_cli.py
```

---

## Documentation Index

Explore our technical documentation suite under `docs/`:

- **[Architecture & System Design](docs/architecture.md):** Modular breakdown, control flow, and data pipelines.
- **[Environment Setup Guide](docs/environment_setup.md):** Python `.venv` management and Metal prerequisites.
- **[Speculative Decoding Math](docs/speculative_decoding_math.md):** Rejection sampling equations, probability bounds, and memory bandwidth tradeoffs.
- **[Pre-Flight Validation Engine](docs/preflight_validation.md):** Tokenizer identity sweeps and memory footprint algorithms.
- **[MLX Profiling Guide](docs/mlx_profiling_guide.md):** Native MLX step loop integration and peak memory tracking.
- **[CLI Reference Manual](docs/cli_reference.md):** Full subcommand and option documentation.
- **[User Questionnaire Backlog](docs/user_questionnaire_backlog.md):** Log of open research topics reserved for future deep dives.

---

## Feedback & Open Source Contributions

I would love your feedback and contributions to make `mlx-specdec-eval` even better!

If you run into issues, have ideas for new features (e.g., support for new quantization schemes, dynamic $K$ scheduling, or additional task domain prompts), or want to share benchmark results from your Mac:

- **Star this repository** if you find `mlx-specdec-eval` helpful.
- **[Open an Issue](https://github.com/santoshdkolur/mlx-specdec-eval/issues)** for bug reports or feature requests.
- **Submit a Pull Request** with improvements, extra prompts, or documentation enhancements.

---

## License
Released under the [MIT License](LICENSE).
