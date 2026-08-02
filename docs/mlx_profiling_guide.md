# MLX Integration & Profiling Guide

This document details how `spec-bench` integrates with Apple's **MLX** framework (`mlx-lm`) to record step-by-step speculative metrics.

---

## 1. Native MLX Speculative Decoding Integration

`mlx-lm` provides native support for speculative generation through internal step utilities (`speculative_generate_step`).

### Model Loading Protocol

Both target and draft models are loaded into Metal unified memory using `mlx_lm.load`:

```python
from mlx_lm import load

target_model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
draft_model, _ = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
```

---

## 2. Profiling Step Loop

To record exact acceptance rates ($\alpha$), time-to-first-token, and latency per token step, `spec-bench` iterates through MLX step generators:

### Metrics Captured Per Generation Run

1. **Prefill Latency ($T_{\text{prefill}}$):** Time taken to process input prompt tokens and generate the initial KV cache.
2. **Decode Latency ($T_{\text{decode}}$):** Time taken to run generation steps.
3. **Draft Acceptance Trace:** Array of accepted tokens per draft verification step.
4. **Peak Metal Memory:** Queried using `mlx.core.metal.get_peak_memory()` in bytes.

---

## 3. Comparative Baseline Step

For every benchmark prompt, `spec-bench` runs:
1. **Baseline Run:** Target model $M_T$ running standard auto-regressive generation without draft model.
2. **Speculative Run:** Target model $M_T$ + Draft model $M_D$ running speculative generation with $K$ draft tokens.

Comparing both runs under identical prompt conditions isolates the exact speedup multiplier $S = \text{TPS}_{\text{speculative}} / \text{TPS}_{\text{baseline}}$.
