# `spec-bench` CLI Reference

Complete command-line interface specification for `spec-bench`.

---

## Global Entry Point

```bash
spec-bench [SUBCOMMAND] [OPTIONS]
```

---

## Subcommands

### 1. `spec-bench check-env`

Runs environment diagnostics to ensure Python, virtualenv, Apple Silicon OS, and Metal acceleration are configured correctly.

#### Options
* `--verbose`: Print detailed package versions and system paths.

#### Example
```bash
spec-bench check-env --verbose
```

---

### 2. `spec-bench validate`

Runs pre-flight compatibility and memory checks for a target/draft model pair.

#### Options
* `-t, --target TEXT`: Path or HF Hub repo ID of the target model. **[Required]**
* `-d, --draft TEXT`: Path or HF Hub repo ID of the draft model. **[Required]**
* `-b, --buffer-gb FLOAT`: Safety memory buffer in GB (default: `2.0`).
* `--force`: Continue even if validation checks fail.

#### Example
```bash
spec-bench validate \
  --target mlx-community/Llama-3.2-3B-Instruct-4bit \
  --draft mlx-community/Llama-3.2-1B-Instruct-4bit
```

---

### 3. `spec-bench run`

Executes speculative decoding benchmarks and baseline comparisons across target and draft models.

#### Options
* `-t, --target TEXT`: Path or HF Hub repo ID of the target model. **[Required]**
* `-d, --draft TEXT`: Path or HF Hub repo ID of the draft model. **[Required]**
* `-k, --num-draft-tokens TEXT`: Comma-separated list of draft token counts to evaluate (e.g. `3,5,7`, default: `5`).
* `--max-tokens INTEGER`: Maximum tokens to generate per prompt (default: `128`).
* `--temp FLOAT`: Generation temperature (default: `0.0`).
* `--warmup INTEGER`: Number of warmup runs prior to timing (default: `1`).
* `--repeats INTEGER`: Number of benchmark repeats per prompt (default: `2`).
* `--category TEXT`: Prompt category to evaluate (`code`, `reasoning`, `chat`, `prose`, or `all`; default: `all`).
* `--prompts-file PATH`: Path to a custom JSON prompts file.
* `--export-json PATH`: Output JSON log filepath.
* `--export-markdown PATH`: Output Markdown summary report filepath.
* `--export-html PATH`: Output interactive HTML report filepath.
* `--skip-preflight`: Skip pre-flight validator checks.
* `--force`: Force execution despite pre-flight check warnings.

#### Example
```bash
spec-bench run \
  --target mlx-community/Llama-3.2-3B-Instruct-4bit \
  --draft mlx-community/Llama-3.2-1B-Instruct-4bit \
  --num-draft-tokens 3,5,7 \
  --max-tokens 256 \
  --export-html report.html \
  --export-markdown summary.md \
  --export-json results.json
```

---

### 4. `spec-bench prompts`

Inspects built-in evaluation prompt suites.

#### Options
* `-c, --category TEXT`: Filter by prompt category (`code`, `reasoning`, `chat`, `prose`).

#### Example
```bash
spec-bench prompts --category code
```

---

### 5. `spec-bench report`

Generates Markdown or interactive HTML reports from a previously saved benchmark JSON results file.

#### Options
* `RESULTS_JSON`: Path to saved benchmark JSON file. **[Required]**
* `-o, --output PATH`: Target HTML or Markdown output path. **[Required]**
* `-f, --format [html|markdown]`: Report format to generate (default: `html`).

#### Example
```bash
spec-bench report results.json --output report.html --format html
```

---

### 6. `spec-bench clean`

Cleans generated benchmark reports (`results.json`, `summary.md`, `report.html`) and optional Hugging Face model cache.

#### Options
* `--all`: Clean Hugging Face local model cache (`~/.cache/huggingface/hub/`) and all generated artifacts.

#### Example
```bash
# Clean local report files
spec-bench clean

# Clean local reports AND Hugging Face cached model weights
spec-bench clean --all
```

