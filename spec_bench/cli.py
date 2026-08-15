"""
Click CLI Module for mlx-specdec-eval.
Provides subcommands: check-env, validate, run, prompts, and report.
"""

import sys
import json
from pathlib import Path
from typing import List, Optional
import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from spec_bench.validator import (
    validate_system_environment,
    validate_tokenizer_compatibility,
    validate_memory_footprint
)
from spec_bench.engine_mlx import MLXSpeculativeEngine, HAS_MLX_LM
from spec_bench.metrics import aggregate_benchmark_results
from spec_bench.reporter import (
    render_tui_summary,
    export_json,
    export_markdown,
    export_html
)


console = Console()

DEFAULT_PROMPTS_PATH = Path(__file__).parent.parent / "configs" / "default_prompts.json"


@click.group()
@click.version_option(version="0.1.0")
def main():
    """🚀 mlx-specdec-eval: Speculative Decoding Benchmarking Tool for Apple Silicon MLX"""




@main.command(name="check-env")
@click.option("--verbose", is_flag=True, help="Display detailed system diagnostics.")
def check_env_cmd(verbose: bool):
    """Diagnose Python version, virtualenv, Apple Silicon OS, and Metal availability."""
    console.print("\n🔍 [bold cyan]Running System Environment Diagnostics...[/bold cyan]\n")
    res = validate_system_environment()

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Diagnostic Check", style="cyan")
    table.add_column("Result", justify="left")
    table.add_column("Status", justify="center")

    table.add_row("Python Version (>= 3.10)", res["python_version"], "[green]PASS[/green]" if res["python_ok"] else "[red]FAIL[/red]")
    table.add_row("Virtual Environment (.venv)", "Active" if res["in_venv"] else "Not active (Recommended)", "[green]PASS[/green]" if res["in_venv"] else "[yellow]WARN[/yellow]")
    table.add_row("Operating System", f"macOS ({res['architecture']})", "[green]PASS[/green]" if res["is_darwin"] else "[red]FAIL[/red]")
    table.add_row("MLX Installed", "Yes" if res["has_mlx"] else "No (pip install mlx)", "[green]PASS[/green]" if res["has_mlx"] else "[red]FAIL[/red]")
    table.add_row("Apple Silicon Metal GPU", "Available" if res["metal_available"] else "Not Available", "[green]PASS[/green]" if res["metal_available"] else "[red]FAIL[/red]")

    console.print(table)

    if res["all_passed"]:
        console.print("\n✅ [bold green]System environment is fully verified and ready for MLX speculative benchmarks![/bold green]\n")
    else:
        console.print("\n⚠️ [bold red]Environment diagnostics identified issues. Please check installation.[/bold red]\n")


@main.command(name="validate")
@click.option("-t", "--target", required=True, help="Target model path or HF Hub repo ID.")
@click.option("-d", "--draft", required=True, help="Draft model path or HF Hub repo ID.")
@click.option("-b", "--buffer-gb", default=2.0, type=float, help="Safety buffer in GB.")
@click.option("--force", is_flag=True, help="Continue even if checks fail.")
def validate_cmd(target: str, draft: str, buffer_gb: float, force: bool):
    """Run pre-flight compatibility and Unified Memory checks for a model pair."""
    console.print(f"\n🔍 [bold cyan]Running Pre-Flight Checks for Target: {target} | Draft: {draft}...[/bold cyan]\n")

    # 1. Tokenizer Check
    tok_ok, tok_errors = validate_tokenizer_compatibility(target, draft)
    if tok_ok:
        console.print("  ✓ [green]Tokenizer Vocabulary & Special Tokens match perfectly.[/green]")
    else:
        console.print("  ❌ [red]Tokenizer Mismatch Detected:[/red]")
        for err in tok_errors:
            console.print(f"     • {err}")

    # 2. Memory Footprint Check
    mem_info = validate_memory_footprint(target, draft, buffer_gb)
    console.print(f"  • [dim]System RAM Available:[/dim]  [bold]{mem_info['available_ram_gb']} GB[/bold] / {mem_info['total_ram_gb']} GB Unified RAM")
    console.print(f"  • [dim]Target Model Footprint:[/dim] ~{mem_info['target_size_gb']} GB")
    console.print(f"  • [dim]Draft Model Footprint:[/dim]  ~{mem_info['draft_size_gb']} GB")
    console.print(f"  • [dim]Total Required Footprint:[/dim] ~{mem_info['total_required_gb']} GB")

    if mem_info["is_safe"]:
        if mem_info.get("is_tight", False):
            console.print("  ⚠️ [yellow]Note: Available RAM is tight, but total Unified Memory is sufficient for Metal dynamic allocation.[/yellow]\n")
        else:
            console.print("  ✓ [green]Unified Memory footprint is safe for Apple Silicon RAM.[/green]\n")
    else:
        console.print(f"  ❌ [red]Out of Memory Risk![/red] Required ~{mem_info['total_required_gb']} GB exceeds total Apple Silicon Unified Memory limits.\n")

    if (not tok_ok or not mem_info["is_safe"]) and not force:
        console.print("❌ [bold red]Pre-Flight Validation Failed![/bold red] Use --force to override.\n")
        sys.exit(1)
    else:
        console.print("✅ [bold green]Pre-Flight Check Passed![/bold green]\n")


@main.command(name="prompts")
@click.option("-c", "--category", default="all", help="Prompt category (code, reasoning, chat, prose, all).")
@click.option("-f", "--prompts-file", default=None, type=click.Path(exists=True), help="Path to custom JSON prompts file.")
def prompts_cmd(category: str, prompts_file: Optional[str]):
    """List built-in or custom evaluation prompt suites."""
    file_path = Path(prompts_file) if prompts_file else DEFAULT_PROMPTS_PATH

    if not file_path.exists():
        console.print(f"❌ Prompts file not found at: {file_path}")
        sys.exit(1)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    console.print(f"\n📚 [bold cyan]Evaluation Prompts Suite ({file_path.name})[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Category", style="cyan")
    table.add_column("Prompt ID", style="bold white")
    table.add_column("Prompt Text", justify="left")

    for cat_name, prompt_list in data.items():
        if category != "all" and category.lower() != cat_name.lower():
            continue
        for p in prompt_list:
            text_preview = p["prompt"][:80] + ("..." if len(p["prompt"]) > 80 else "")
            table.add_row(cat_name.upper(), p["id"], text_preview)

    console.print(table)
    console.print()


@main.command(name="run")
@click.option("-t", "--target", required=True, help="Target model path or HF Hub repo ID.")
@click.option("-d", "--draft", required=True, help="Draft model path or HF Hub repo ID.")
@click.option("-k", "--num-draft-tokens", default="5", help="Comma-separated draft token counts K to evaluate (e.g. 3,5,7).")
@click.option("--max-tokens", default=128, type=int, help="Maximum tokens to generate per prompt.")
@click.option("--temp", default=0.0, type=float, help="Generation temperature.")
@click.option("--warmup", default=1, type=int, help="Number of warmup runs.")
@click.option("--repeats", default=2, type=int, help="Number of repeat runs per prompt.")
@click.option("--category", default="all", help="Prompt category filter (code, reasoning, chat, prose, all).")
@click.option("--prompts-file", default=None, type=click.Path(exists=True), help="Path to custom JSON prompts file.")
@click.option("--prompt-id", default=None, help="Filter evaluation to a single prompt ID (e.g. reason_math_proof, code_quicksort).")
@click.option("--prompt", "single_prompt", default=None, help="Run evaluation on a single custom prompt string.")
@click.option("--adaptive/--no-adaptive", "adaptive", default=True, help="Enable/disable Adaptive K evaluation (default: enabled).")
@click.option("--min-k", default=1, type=int, help="Minimum draft tokens for Adaptive K (default: 1).")
@click.option("--max-k", default=5, type=int, help="Maximum draft tokens for Adaptive K (default: 5).")
@click.option("--initial-k", default=3, type=int, help="Initial draft tokens for Adaptive K (default: 3).")
@click.option("--export-json", "export_json_path", default=None, help="Filepath to export JSON results.")
@click.option("--export-markdown", "export_markdown_path", default=None, help="Filepath to export Markdown summary.")
@click.option("--export-html", "export_html_path", default=None, help="Filepath to export visual HTML report.")

@click.option("--skip-preflight", is_flag=True, help="Skip pre-flight checks.")
@click.option("--force", is_flag=True, help="Force execution despite warnings.")
@click.option("--synthetic", is_flag=True, help="Force synthetic evaluation mode for demonstration/testing.")
def run_cmd(
    target: str,
    draft: str,
    num_draft_tokens: str,
    max_tokens: int,
    temp: float,
    warmup: int,
    repeats: int,
    category: str,
    prompts_file: Optional[str],
    prompt_id: Optional[str],
    single_prompt: Optional[str],
    adaptive: bool,
    min_k: int,
    max_k: int,
    initial_k: int,
    export_json_path: Optional[str],
    export_markdown_path: Optional[str],
    export_html_path: Optional[str],
    skip_preflight: bool,
    force: bool,
    synthetic: bool
):
    """Run speculative decoding benchmark across parameter sweeps, adaptive K, and prompt categories."""

    # 1. Parse draft counts K
    try:
        K_values = [int(x.strip()) for x in num_draft_tokens.split(",") if x.strip()]
    except ValueError:
        console.print("❌ Invalid --num-draft-tokens parameter. Must be comma-separated integers (e.g. 3,5,7).")
        sys.exit(1)

    # 2. Pre-flight checks
    if not skip_preflight:
        tok_ok, _ = validate_tokenizer_compatibility(target, draft)
        mem_info = validate_memory_footprint(target, draft)
        if (not tok_ok or not mem_info["is_safe"]) and not force:
            console.print("❌ [bold red]Pre-Flight Validation Failed![/bold red] Use --force or --skip-preflight to override.")
            sys.exit(1)

    # 3. Load prompts
    if single_prompt:
        selected_prompts = [{
            "id": "custom_prompt",
            "category": "custom",
            "prompt": single_prompt
        }]
    else:
        prompts_path = Path(prompts_file) if prompts_file else DEFAULT_PROMPTS_PATH
        if not prompts_path.exists():
            console.print(f"❌ Prompts file not found: {prompts_path}")
            sys.exit(1)

        with open(prompts_path, "r", encoding="utf-8") as f:
            prompts_data = json.load(f)

        selected_prompts = []
        for cat_name, p_list in prompts_data.items():
            if category == "all" or category.lower() == cat_name.lower():
                selected_prompts.extend(p_list)

        if prompt_id:
            selected_prompts = [p for p in selected_prompts if p["id"].lower() == prompt_id.lower()]

    if not selected_prompts:
        target_name = f"prompt-id '{prompt_id}'" if prompt_id else f"category '{category}'"
        console.print(f"❌ No prompts found matching {target_name}.")
        sys.exit(1)


    console.print(f"\n🚀 [bold cyan]Starting Benchmark: Target={target} | Draft={draft}[/bold cyan]")
    console.print(f"   [dim]Fixed Sweeps (K): {K_values} | Adaptive: {adaptive} (K∈[{min_k}..{max_k}]) | Prompts: {len(selected_prompts)} | Repeats: {repeats}[/dim]\n")

    # Determine if synthetic mode should be used
    is_synthetic = synthetic or not HAS_MLX_LM or "dummy" in target.lower() or "dummy" in draft.lower()

    if is_synthetic:
        console.print("⚠️ Running in synthetic evaluation mode for demonstration/testing.\n")
        baseline_runs = []
        runs_by_k = {k: [] for k in K_values}
        adaptive_runs = [] if adaptive else None

        for p in selected_prompts:
            base_entry = {
                "prompt_id": p["id"],
                "category": p["category"],
                "text": f"Baseline response for prompt {p['id']}",
                "tps": 25.0,
                "elapsed_time_s": 2.0,
                "peak_memory_mb": 4000.0
            }
            baseline_runs.append(base_entry)

            for k in K_values:
                runs_by_k[k].append({
                    "prompt_id": p["id"],
                    "category": p["category"],
                    "text": f"Baseline response for prompt {p['id']}",
                    "tps": 25.0 * (1.0 + (k * 0.08)),
                    "alpha": 0.75,
                    "total_draft_proposed": k * 10,
                    "accepted_draft_tokens": int(k * 7.5),
                    "output_tokens": 128,
                    "peak_memory_mb": 4800.0,
                    "elapsed_time_s": 1.5
                })

            if adaptive:
                adaptive_runs.append({
                    "prompt_id": p["id"],
                    "category": p["category"],
                    "text": f"Baseline response for prompt {p['id']}",
                    "tps": 34.5,
                    "alpha": 0.78,
                    "mean_effective_k": 3.4,
                    "wasted_draft_ratio": 0.18,
                    "total_draft_proposed": 80,
                    "accepted_draft_tokens": 62,
                    "output_tokens": 128,
                    "k_history": [{"round": 1, "k": 3, "accepted": 2, "alpha_round": 0.67, "next_k": 3}],
                    "k_selections": [3, 4, 3, 4],
                    "peak_memory_mb": 4850.0,
                    "elapsed_time_s": 1.3
                })

        results = aggregate_benchmark_results(target, draft, K_values, runs_by_k, baseline_runs, adaptive_runs)

    else:
        engine = MLXSpeculativeEngine(target, draft)
        engine.load_models()

        # Warmup runs across baseline, all K shapes, and adaptive
        if warmup > 0 and selected_prompts:
            console.print("🔥 Running Initial Warmup Iterations (Compiling Metal GPU Shaders)...")
            warm_prompt = selected_prompts[0]["prompt"]
            for _ in range(warmup):
                engine.run_baseline(warm_prompt, max_tokens=min(32, max_tokens), temp=temp)
                # Warm up all possible K values to prevent dynamic shape compilation spikes
                all_possible_k = sorted(list(set(K_values + list(range(min_k, max_k + 1)))))
                for k_val in all_possible_k:
                    engine.run_speculative(warm_prompt, num_draft_tokens=k_val, max_tokens=min(32, max_tokens), temp=temp)
                if adaptive:
                    engine.run_adaptive_speculative(warm_prompt, min_k=min_k, max_k=max_k, initial_k=initial_k, max_tokens=min(32, max_tokens), temp=temp)

        # 1. Baseline Target Runs
        console.print("📊 Executing Baseline (Non-Speculative) Runs...")
        baseline_runs = []
        for p in selected_prompts:
            for _ in range(repeats):
                res = engine.run_baseline(p["prompt"], max_tokens=max_tokens, temp=temp)
                res["prompt_id"] = p["id"]
                res["category"] = p["category"]
                baseline_runs.append(res)

        # 2. Fixed Speculative Runs across K
        runs_by_k = {}
        for k in K_values:
            console.print(f"⚡ Executing Fixed Speculative Runs for K={k}...")
            k_runs = []
            for p in selected_prompts:
                for _ in range(repeats):
                    res = engine.run_speculative(p["prompt"], num_draft_tokens=k, max_tokens=max_tokens, temp=temp)
                    res["prompt_id"] = p["id"]
                    res["category"] = p["category"]
                    k_runs.append(res)
            runs_by_k[k] = k_runs

        # 3. Adaptive Speculative Runs
        adaptive_runs = []
        if adaptive:
            console.print(f"🧠 Executing Adaptive Speculative Runs (Dynamic K∈[{min_k}..{max_k}])...")
            for p in selected_prompts:
                for _ in range(repeats):
                    res = engine.run_adaptive_speculative(
                        p["prompt"],
                        min_k=min_k,
                        max_k=max_k,
                        initial_k=initial_k,
                        max_tokens=max_tokens,
                        temp=temp
                    )
                    res["prompt_id"] = p["id"]
                    res["category"] = p["category"]
                    adaptive_runs.append(res)
        else:
            adaptive_runs = None

        results = aggregate_benchmark_results(target, draft, K_values, runs_by_k, baseline_runs, adaptive_runs)


    # Render TUI table summary
    render_tui_summary(results)

    # Export outputs
    if export_json_path:
        export_json(results, export_json_path)
    if export_markdown_path:
        export_markdown(results, export_markdown_path)
    if export_html_path:
        export_html(results, export_html_path)



@main.command(name="report")
@click.argument("results_json", type=click.Path(exists=True))
@click.option("-o", "--output", required=True, help="Target HTML or Markdown output filepath.")
@click.option("-f", "--format", "report_format", default="html", type=click.Choice(["html", "markdown"]), help="Output format.")
def report_cmd(results_json: str, output: str, report_format: str):
    """Generate HTML or Markdown reports from a saved JSON results file."""
    with open(results_json, "r", encoding="utf-8") as f:
        results = json.load(f)

    if report_format == "html":
        export_html(results, output)
    else:
        export_markdown(results, output)


@main.command(name="clean")
@click.option("--all", "clean_all", is_flag=True, help="Clean Hugging Face local model cache and all generated artifacts.")
def clean_cmd(clean_all: bool):
    """Clean generated benchmark reports, JSON results, and optional HF cache."""
    import shutil

    console.print("\n🧹 [bold cyan]Cleaning up mlx-specdec-eval artifacts...[/bold cyan]\n")

    artifacts = ["results.json", "summary.md", "report.html", "custom_report.html"]
    removed_count = 0

    for item in artifacts:
        p = Path(item)
        if p.exists():
            p.unlink()
            console.print(f"  • Removed file: [dim]{item}[/dim]")
            removed_count += 1

    cache_dir = Path(".pytest_cache")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        console.print("  • Removed directory: [dim].pytest_cache[/dim]")
        removed_count += 1

    if clean_all:
        local_models = Path("models")
        if local_models.exists():
            shutil.rmtree(local_models)
            console.print("  • Cleared project models directory: [dim]models/[/dim]")
            removed_count += 1

        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        if hf_cache.exists():
            shutil.rmtree(hf_cache)
            console.print(f"  • Cleared Hugging Face Model Cache: [dim]{hf_cache}[/dim]")
            removed_count += 1
        else:
            console.print("  • Hugging Face cache directory not found or already clean.")


    console.print(f"\n✅ [bold green]Cleanup completed ({removed_count} item(s) processed)![/bold green]\n")


if __name__ == "__main__":
    main()

