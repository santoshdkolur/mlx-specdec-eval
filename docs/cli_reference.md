# `mlx-specdec-eval` CLI Reference

Complete command-line interface specification for `mlx-specdec-eval`.

---

## Global Entry Point

```bash
mlx-specdec-eval [SUBCOMMAND] [OPTIONS]
```

---

## Subcommands

### 1. `mlx-specdec-eval check-env`

Runs environment diagnostics to ensure Python, virtualenv, Apple Silicon OS, and Metal acceleration are configured correctly.

#### Options
* `--verbose`: Print detailed package versions and system paths.

#### Example
```bash
mlx-specdec-eval check-env --verbose
```

---

### 2. `mlx-specdec-eval validate`

Runs pre-flight compatibility and memory checks for a target/draft model pair.

#### Options
* `-t, --target TEXT`: Path or HF Hub repo ID of the target model. **[Required]**
* `-d, --draft TEXT`: Path or HF Hub repo ID of the draft model. **[Required]**
* `-b, --buffer-gb FLOAT`: Safety memory buffer in GB (default: `2.0`).
* `--force`: Continue even if validation checks fail.

#### Example
```bash
mlx-specdec-eval validate \
  --target mlx-community/Llama-3.2-3B-Instruct-4bit \
  --draft mlx-community/Llama-3.2-1B-Instruct-4bit
```

* Note: Commands can be executed using either `spec-bench` or `mlx-specdec-eval`.

---

### 3. `mlx-specdec-eval run` (or `spec-bench run`)

Executes speculative decoding benchmarks and baseline comparisons across target and draft models.

#### Options
* `-t, --target TEXT`: Path or HF Hub repo ID of the target model. **[Required]**
* `-d, --draft TEXT`: Path or HF Hub repo ID of the draft model. **[Required]**
* `-k, --num-draft-tokens TEXT`: Comma-separated list of draft token counts to evaluate (e.g. `3,5,7`, default: `5`).
* `--prompt-id TEXT`: Filter evaluation to a single prompt ID (e.g. `code_quicksort`, `reason_math_proof`).
* `--prompt TEXT`: Run evaluation on a single custom prompt string.
* `--max-tokens INTEGER`: Maximum tokens to generate per prompt (default: `128`).
* `--temp FLOAT`: Generation temperature (default: `0.0`).
* `--warmup INTEGER`: Number of warmup runs prior to timing (default: `1`). Pre-compiles Metal GPU sequence shaders and warms unified memory cache.
* `--repeats INTEGER`: Number of benchmark repeats per prompt (default: `2`). Runs each prompt $N$ times and averages TPS, speedup, and acceptance rates across runs.

* `--category TEXT`: Prompt category to evaluate (`code`, `reasoning`, `chat`, `prose`, or `all`; default: `all`).
* `--prompts-file PATH`: Path to a custom JSON prompts file.
* `--export-json PATH`: Output JSON log filepath.
* `--export-markdown PATH`: Output Markdown summary report filepath.
* `--export-html PATH`: Output interactive HTML report filepath.
* `--skip-preflight`: Skip pre-flight validator checks.
* `--force`: Force execution despite pre-flight check warnings.


#### Example
```bash
mlx-specdec-eval run \
  --target mlx-community/Llama-3.2-3B-Instruct-4bit \
  --draft mlx-community/Llama-3.2-1B-Instruct-4bit \
  --num-draft-tokens 3,5,7 \
  --max-tokens 256 \
  --export-html report.html \
  --export-markdown summary.md \
  --export-json results.json
```

---

### 4. `mlx-specdec-eval prompts`

Inspects built-in evaluation prompt suites.

#### Options
* `-c, --category TEXT`: Filter by prompt category (`code`, `reasoning`, `chat`, `prose`).

#### Example
```bash
mlx-specdec-eval prompts --category code
```

---

### 5. `mlx-specdec-eval report`

Generates Markdown or interactive HTML reports from a previously saved benchmark JSON results file.

#### Options
* `RESULTS_JSON`: Path to saved benchmark JSON file. **[Required]**
* `-o, --output PATH`: Target HTML or Markdown output path. **[Required]**
* `-f, --format [html|markdown]`: Report format to generate (default: `html`).

#### Example
```bash
mlx-specdec-eval report results.json --output report.html --format html
```

---

### 6. `mlx-specdec-eval clean`

Cleans generated benchmark reports (`results.json`, `summary.md`, `report.html`) and optional Hugging Face model cache.

#### Options
* `--all`: Clean Hugging Face local model cache (`~/.cache/huggingface/hub/`) and all generated artifacts.

#### Example
```bash
# Clean local report files
mlx-specdec-eval clean

# Clean local reports AND Hugging Face cached model weights
mlx-specdec-eval clean --all
```

