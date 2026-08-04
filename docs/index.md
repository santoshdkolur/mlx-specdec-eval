# `mlx-specdec-eval` Documentation Index

Welcome to the documentation suite for `mlx-specdec-eval`, an evaluation tool for Speculative Decoding on Apple Silicon Macs using MLX.

## Documentation Structure

1. **[Architecture](architecture.md)**: Detailed system design, component breakdown, module interactions, and software patterns.
2. **[Environment Setup](environment_setup.md)**: Guide to setting up Python 3.10+, virtual environments (`.venv`), Apple Silicon Metal prerequisites, and MLX installation.
3. **[Speculative Decoding Math](speculative_decoding_math.md)**: Theoretical foundation, acceptance rate ($\alpha$), Tokens Per Second (TPS), Speedup factor ($S$), and memory bandwidth tradeoffs.
4. **[Pre-Flight Validation Engine](preflight_validation.md)**: Tokenizer compatibility checks, vocabulary identity verification, and Apple Silicon RAM/VRAM footprint estimation algorithms.
5. **[MLX Profiling Guide](mlx_profiling_guide.md)**: How `mlx-specdec-eval` hooks into `mlx-lm` step loops to collect per-token metrics, latency breakdowns, and peak Metal memory.
6. **[CLI Reference](cli_reference.md)**: Complete guide to command-line interface subcommands (`run`, `validate`, `check-env`, `prompts`, `report`) and parameter sweeps.
7. **[User Questionnaire Backlog](user_questionnaire_backlog.md)**: Log of open topics and concepts reserved for future deep dives.


---

## Overview

Running local LLMs on Apple Silicon requires balancing compute throughput with Unified Memory Bandwidth. Speculative Decoding pairs a small, low-latency **Draft Model** ($M_D$) with a larger **Target Model** ($M_T$). 

`mlx-specdec-eval` provides an automated, scientific framework to answer:
- *Is speculative decoding actually faster on my specific Mac configuration?*
- *What is the token acceptance rate $\alpha$ for my prompt domain?*
- *What is the optimal draft token count $K$?*
- *Does loading two models into Unified Memory cause VRAM thrashing or memory bandwidth saturation?*
