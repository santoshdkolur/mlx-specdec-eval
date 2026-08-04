# Pre-Flight Validation Engine

Before allocating GPU memory or running generation steps, `mlx-specdec-eval` executes fail-fast pre-flight checks to prevent runtime errors, vocabulary mismatches, or Unified Memory thrashing.

---

## Pre-Flight Check Pipeline

```text
[Input Target & Draft Models]
           │
           ▼
┌──────────────────────────────┐
│ 1. System Environment Check  │  --> Verifies Python >= 3.10, venv, macOS, Metal GPU
└──────────┬───────────────────┘
           │ Pass
           ▼
┌──────────────────────────────┐
│ 2. Tokenizer Identity Check  │  --> Vocabulary size identity
└──────────┬───────────────────┘  --> Special token alignment (bos, eos, pad)
           │ Pass                 --> Encoding sweep on sample benchmark string
           ▼
┌──────────────────────────────┐
│ 3. Unified Memory Footprint  │  --> Estimates model weights + KV cache size
└──────────┬───────────────────┘  --> Compares against available system RAM (psutil)
           │ Pass
           ▼
[ Proceed to Benchmark Engine ]
```

---

## 1. Tokenizer Compatibility Validation

For speculative decoding to work correctly, both target and draft models MUST share identical tokenization rules. A mismatch causes invalid token indices or corrupted generation outputs.

The validation checks include:
1. **Vocabulary Size Identity:**
   $$\text{len}(\text{vocab}_T) == \text{len}(\text{vocab}_D)$$

2. **Special Tokens Alignment:**
   - `bos_token_id` match
   - `eos_token_id` match
   - `pad_token_id` match

3. **Sample Encoding Identity Sweep:**
   A test string (e.g. `"def benchmark_test(): return 'speculative_decoding_test_123'"`) is encoded by both tokenizers. The generated token ID arrays must be identical element-for-element:
   $$\text{encode}_T(\text{sample}) == \text{encode}_D(\text{sample})$$

---

## 2. Unified Memory Footprint Validation

Apple Silicon uses Unified Memory shared between CPU and GPU. If total model weight footprint plus KV cache memory exceeds available RAM, macOS initiates swapfile thrashing to SSD, causing severe performance degradation.

### Footprint Formula

$$\text{Memory}_{\text{required}} = \text{Size}(M_T) + \text{Size}(M_D) + \text{KV}_{\text{Cache}}(M_T) + \text{KV}_{\text{Cache}}(M_D) + \text{SafetyBuffer (2GB)}$$

- **Local Directory Size:** Calculated by scanning `.safetensors`, `.bin`, `.npz` file sizes on disk.
- **Hugging Face Hub Model Size:** Fetched from HF Hub API metadata (`siblings` file size summation) when models are specified via repository IDs (e.g. `mlx-community/Llama-3.2-3B-Instruct-4bit`).
- **Available System RAM:** Queried via `psutil.virtual_memory().available`.

If $\text{Memory}_{\text{required}} > \text{RAM}_{\text{available}}$, `mlx-specdec-eval` aborts with a `MemoryError` diagnostic unless `--force` or `--skip-preflight` is passed.
