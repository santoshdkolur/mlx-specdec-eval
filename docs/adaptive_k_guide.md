# Adaptive $K$ (Dynamic Speculative Decoding) Guide

## 1. The Core Motivation: Why Static $K$ Fails

In standard Speculative Decoding, the number of draft candidate tokens proposed per iteration ($K$) is held fixed (e.g., $K=3$ or $K=5$). 

However, the token acceptance rate $\alpha$ is **non-stationary** across different tasks and within single responses:
- **High-Predictability Domains (Code & Formal Math):** Target models routinely accept 4 to 6 tokens in sequence ($\alpha \ge 75\%$). A low static $K$ (e.g., $K=2$) under-speculates and caps throughput.
- **Low-Predictability Domains (Open Prose & Creative Dialogue):** High semantic entropy causes target models to reject early candidate tokens ($\alpha \le 35\%$). A high static $K$ (e.g., $K=5$) over-speculates, forcing the draft model to generate tokens that will be discarded, creating net negative speedup ($S < 1.0\times$).

**Adaptive $K$ dynamically modulates the speculation depth $K_t \in [K_{\min}, K_{\max}]$ in response to live empirical predictability.**

---

## 2. Why Naive Token-Level Heuristics Fail on Apple Silicon

Many theoretical papers propose token-level confidence thresholding (e.g., stopping the draft phase if draft token logprob $P(x_i) < \tau$ or Shannon entropy $H(x_i) > \eta$). 

On Apple Silicon (Metal) and PyTorch MPS/MLX backends, **naive token-level early stopping severely degrades performance** due to three hardware bottlenecks:

1. **Host-Device Synchronization Barriers (`mx.eval()` CPU Halts):**
   Inspecting token probabilities in Python requires transferring tensor values from Metal unified buffers to CPU memory. This triggers an explicit pipeline synchronization barrier (`mx.eval()`), destroying GPU concurrency and costing 0.5ms–2.0ms per token.
2. **Dynamic Metal Shader JIT Re-compilation:**
   Varying tensor batch lengths step-by-step ($K = 3 \to 1 \to 5$) causes the Metal compiler to repeatedly JIT compile specialized sequence length kernels (costing 50ms–200ms latency spikes).
3. **Draft-Target Calibration Divergence:**
   Small draft models (0.5B) are often overconfident on incorrect tokens; draft entropy does not reliably correlate with target model acceptance.

---

## 3. The `mlx-specdec-eval` Solution: Round-Boundary EMA Scheduler

`mlx-specdec-eval` uses a **Round-Boundary Exponential Moving Average (EMA) Scheduler** that operates strictly between verification passes without interrupting the GPU generation stream:

```
                          ┌────────────────────────┐
                          │   Current Depth K_t    │
                          └───────────┬────────────┘
                                      │
               ┌──────────────────────┴──────────────────────┐
               ▼                                             ▼
     [ Draft Phase (K_t) ]                         [ Target Verify (K_t + 1) ]
  • GPU executes K_t passes                     • 1 single parallel forward pass
  • Asynchronous execution                      • Exact matching in memory
               │                                             │
               └──────────────────────┬──────────────────────┘
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │    Round Acceptance α_round   │
                      │  α_round = n_accepted / K_t   │
                      └───────────────┬───────────────┘
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │     EMA Acceptance Update     │
                      │ α_EMA = β·α_EMA + (1-β)·α_rnd │
                      └───────────────┬───────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
  [ High Acceptance ]          [ Low Acceptance ]           [ Moderate ]
  (n == K_t or α ≥ 0.75)       (n == 0 or α ≤ 0.33)         (0.33 < α < 0.75)
         │                            │                            │
         ▼                            ▼                            ▼
  K_{t+1} = min(K_t+1, K_max)   K_{t+1} = max(K_t-1, K_min)   K_{t+1} = K_t
```

### Mathematical Formulation

1. **Round Acceptance Rate:**
   $$\alpha_{\text{round}} = \frac{n_{\text{accepted}}}{K_t}$$
2. **Exponential Moving Average:**
   $$\alpha_{\text{EMA}}^{(t)} = \beta \cdot \alpha_{\text{EMA}}^{(t-1)} + (1 - \beta) \cdot \alpha_{\text{round}}$$
3. **Transition Rules:**
   $$K_{t+1} = \begin{cases} \min(K_t + 1, K_{\max}) & \text{if } n_{\text{accepted}} = K_t \text{ or } \alpha_{\text{round}} \ge 0.75 \\ \max(K_t - 1, K_{\min}) & \text{if } n_{\text{accepted}} = 0 \text{ or } \alpha_{\text{round}} \le 0.33 \\ K_t & \text{otherwise} \end{cases}$$

---

## 4. Key Metrics for Adaptive Speculative Decoding

### 1. Mean Effective $K$ ($\bar{K}$)
$$\bar{K} = \frac{1}{R} \sum_{r=1}^{R} K_r$$
Measures the average draft depth chosen by the scheduler. Differentiates true dynamic adaptation from static values.

### 2. Wasted Draft Ratio ($\text{WDR}$)
$$\text{WDR} = \frac{N_{\text{proposed}} - N_{\text{accepted}}}{N_{\text{generated}}}$$
Quantifies the percentage of draft compute that was rejected and thrown away. Adaptive $K$ significantly lowers $\text{WDR}$ on low-predictability prompts.

### 3. Strict Output Determinism ($EM = 100\%$)
At $T=0.0$, the generated completion across Baseline, Fixed $K$, and Adaptive $K$ must be 100% token-for-token identical.

---

## 5. Usage in `spec-bench` CLI

Run adaptive speculative evaluation alongside baseline and fixed sweeps:

```bash
spec-bench run \
  --target mlx-community/Qwen2.5-7B-Instruct-4bit \
  --draft mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --num-draft-tokens 3,5 \
  --adaptive \
  --min-k 1 \
  --max-k 5 \
  --initial-k 3
```

### CLI Flags Reference

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--adaptive / --no-adaptive` | Flag | `True` | Enable or disable Adaptive $K$ in benchmark. |
| `--min-k` | Integer | `1` | Lower bound for draft depth $K$. |
| `--max-k` | Integer | `5` | Upper bound for draft depth $K$. |
| `--initial-k` | Integer | `3` | Starting draft depth $K$ for first round. |

---

## 6. Interpreting Head-to-Head Results

```text
====================================================================================================
                        HEAD-TO-HEAD SPECULATIVE DECODING EVALUATION
====================================================================================================
Mode           Decode TPS   Speedup (S)   Acceptance (α)   Avg K (K̄)   Wasted Draft %   Output Match
----------------------------------------------------------------------------------------------------
Baseline       24.8 TPS     1.00x         N/A              0.0         0.0%             100%
Fixed K=3      30.5 TPS     1.23x         61.2%            3.0         31.5%            100%
Fixed K=5      26.4 TPS     1.06x         44.1%            5.0         50.8%            100%
----------------------------------------------------------------------------------------------------
Adaptive K     33.8 TPS     1.36x (BEST)  69.8%            2.8         18.2% (LOWEST)   100%
====================================================================================================
```

- **Speedup ($S > 1.0\times$):** Higher than fixed $K$ sweeps because it expands depth during repetitive tokens and contracts during difficult transitions.
- **Wasted Draft Ratio (Lowest):** Avoids generating wasted tokens in prose/chat sections.
