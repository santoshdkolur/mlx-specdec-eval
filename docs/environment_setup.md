# Environment Setup Guide

This guide covers setting up an environment for running `spec-bench` on Apple Silicon Macs.

---

## 1. Hardware & Operating System Requirements

- **Processor:** Apple Silicon M-series chip (M1, M1 Pro, M1 Max, M1 Ultra, M2, M3, M4).
- **RAM:** Minimum 8GB (16GB+ recommended for 7B+ models; 36GB+ for 14B+ target models).
- **Operating System:** macOS Ventura (13.5) or newer.

---

## 2. Python & Virtual Environment Setup

`spec-bench` requires **Python 3.10** or higher.

### Creating the Virtual Environment (`.venv`)

Always isolate project dependencies inside a virtual environment:

```bash
# Navigate to the repository
cd /path/to/EvalSpecDec

# Create a virtual environment named .venv
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate
```

When activated, your terminal prompt will display `(.venv)`.

---

## 3. Package Installation

Install `spec-bench` and MLX in editable mode:

```bash
pip install --upgrade pip
pip install -e ".[dev]"
```

### Key Dependencies Installed

- `mlx`: Framework for machine learning on Apple Silicon.
- `mlx-lm`: LLM utilities and speculative decoding primitives for MLX.
- `transformers`: Hugging Face tokenizers and model metadata utilities.
- `psutil`: System memory and process profiling.
- `rich`: Terminal UI rendering.
- `click`: Command-line interface framework.
- `jinja2`: HTML report template rendering.

---

## 4. Diagnostics & Verification

Run the built-in environment diagnostic command to verify your environment setup:

```bash
spec-bench check-env
```

### Expected Output Checklist

- `[PASS]` Python Version: 3.10+ detected.
- `[PASS]` Virtual Environment: Active (`.venv`).
- `[PASS]` OS Platform: macOS (`darwin`).
- `[PASS]` Apple Silicon Metal: Available (`mlx.core.metal.is_available() == True`).
- `[PASS]` Dependencies: `mlx`, `mlx-lm`, `transformers`, `rich`, `click` imported successfully.
