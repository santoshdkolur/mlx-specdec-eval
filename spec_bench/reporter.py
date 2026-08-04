"""
Reporter & Exporters Module for mlx-specdec-eval.
Renders Rich TUI terminal tables and exports results to JSON, Markdown, and interactive HTML.
"""

import json
from pathlib import Path
from typing import Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
import jinja2

console = Console()


def render_tui_summary(results: Dict[str, Any]):
    """Renders formatted Rich terminal table summary of benchmark results."""
    target_model = results.get("target_model", "Unknown Target")
    draft_model = results.get("draft_model", "Unknown Draft")
    optimal_k = results.get("optimal_k", 5)
    best_speedup = results.get("best_speedup", 1.0)

    header_text = Text()
    header_text.append("🚀 mlx-specdec-eval Speculative Decoding Report\n", style="bold cyan")
    header_text.append(f"Target Model: {target_model}\n", style="bold white")
    header_text.append(f"Draft Model:  {draft_model}\n", style="dim white")

    console.print(Panel(header_text, expand=False, border_style="cyan"))

    table = Table(title="Parameter Sweep Results (Draft Tokens K)", show_header=True, header_style="bold magenta")
    table.add_column("K", justify="center", style="cyan")
    table.add_column("Baseline TPS", justify="right")
    table.add_column("Speculative TPS", justify="right")
    table.add_column("Acceptance Rate (α)", justify="right")
    table.add_column("Output Match", justify="center")
    table.add_column("Speedup (S)", justify="right")
    table.add_column("Status", justify="center")

    k_sweeps = results.get("k_sweeps", {})
    for k_str, data in k_sweeps.items():
        k_val = data["num_draft_tokens"]
        base_tps = f"{data['avg_baseline_tps']:.1f}"
        spec_tps = f"{data['avg_speculative_tps']:.1f}"
        alpha = f"{data['avg_acceptance_rate_alpha'] * 100:.1f}%"
        match_rate = f"{data.get('output_exact_match_rate', 1.0) * 100:.0f}%"
        match_styled = f"[green]{match_rate} 100% Exact[/green]" if data.get('output_exact_match_rate', 1.0) == 1.0 else f"[yellow]{match_rate}[/yellow]"
        speedup_val = data['speedup_factor']
        speedup_str = f"{speedup_val:.2f}x"

        if speedup_val > 1.05:
            status = "[green]Accelerated[/green]"
            speedup_styled = f"[bold green]{speedup_str}[/bold green]"
        elif speedup_val >= 0.95:
            status = "[yellow]Neutral[/yellow]"
            speedup_styled = f"[yellow]{speedup_str}[/yellow]"
        else:
            status = "[red]Degraded[/red]"
            speedup_styled = f"[bold red]{speedup_str}[/bold red]"

        if k_val == optimal_k:
            k_display = f"[bold underline cyan]{k_val} (Best)[/bold underline cyan]"
        else:
            k_display = str(k_val)

        table.add_row(k_display, base_tps, spec_tps, alpha, match_styled, speedup_styled, status)


    console.print(table)

    # Category Domain Breakdown Table
    cat_breakdown = results.get("category_breakdown", {})
    if cat_breakdown:
        cat_table = Table(title=f"Task Domain Breakdown (Optimal K={optimal_k})", show_header=True, header_style="bold cyan")
        cat_table.add_column("Category", style="bold white")
        cat_table.add_column("Baseline TPS", justify="right")
        cat_table.add_column("Speculative TPS", justify="right")
        cat_table.add_column("Acceptance Rate (α)", justify="right")
        cat_table.add_column("Output Match", justify="center")
        cat_table.add_column("Speedup (S)", justify="right")

        for cat_name, cdata in cat_breakdown.items():
            c_base = f"{cdata['baseline_tps']:.1f}"
            c_spec = f"{cdata['speculative_tps']:.1f}"
            c_alpha = f"{cdata['alpha'] * 100:.1f}%"
            c_match = f"{cdata['exact_match_rate'] * 100:.0f}%"
            c_s = cdata['speedup_factor']
            c_s_styled = f"[bold green]{c_s:.2f}x[/bold green]" if c_s > 1.05 else (f"[yellow]{c_s:.2f}x[/yellow]" if c_s >= 0.95 else f"[bold red]{c_s:.2f}x[/bold red]")
            cat_table.add_row(cat_name.upper(), c_base, c_spec, c_alpha, c_match, c_s_styled)

        console.print(cat_table)

    opt_style = "bold green" if best_speedup > 1.0 else "bold yellow"
    console.print(f"\n💡 [bold]Recommendation:[/bold] Optimal draft length is [cyan]K={optimal_k}[/cyan] with a speedup of [{opt_style}]{best_speedup:.2f}x[/{opt_style}].\n")



def export_json(results: Dict[str, Any], filepath: str):
    """Exports benchmark results to a JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    console.print(f"📄 JSON report saved to: [underline cyan]{filepath}[/underline cyan]")


def export_markdown(results: Dict[str, Any], filepath: str):
    """Exports benchmark results to a Markdown file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    target_model = results.get("target_model", "Unknown Target")
    draft_model = results.get("draft_model", "Unknown Draft")
    optimal_k = results.get("optimal_k", 5)
    best_speedup = results.get("best_speedup", 1.0)
    k_sweeps = results.get("k_sweeps", {})

    md_lines = [
        f"# `mlx-specdec-eval` Speculative Decoding Report",
        "",
        f"- **Target Model:** `{target_model}`",
        f"- **Draft Model:** `{draft_model}`",
        f"- **Optimal Draft Length (K):** `{optimal_k}`",
        f"- **Best Speedup Factor (S):** `{best_speedup:.2f}x`",
        "",
        "## Performance Summary Table",
        "",
        "| Draft Count (K) | Baseline TPS | Speculative TPS | Acceptance Rate (α) | Speedup (S) | Status |",
        "| :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for k_str, data in k_sweeps.items():
        k_val = data["num_draft_tokens"]
        base_tps = data['avg_baseline_tps']
        spec_tps = data['avg_speculative_tps']
        alpha = data['avg_acceptance_rate_alpha'] * 100
        s_val = data['speedup_factor']
        status = "Accelerated 🚀" if s_val > 1.0 else "Degraded ⚠️"

        md_lines.append(
            f"| **K={k_val}** | {base_tps:.1f} | {spec_tps:.1f} | {alpha:.1f}% | `{s_val:.2f}x` | {status} |"
        )

    md_lines.extend([
        "",
        "## Key Takeaways",
        "",
        f"> [!TIP]",
        f"> Using draft length **K={optimal_k}** yields the maximum speedup of **{best_speedup:.2f}x** over baseline target model generation.",
        ""
    ])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    console.print(f"📝 Markdown report saved to: [underline cyan]{filepath}[/underline cyan]")


def export_html(results: Dict[str, Any], filepath: str):
    """Renders and exports an interactive HTML visual report using Chart.js."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>mlx-specdec-eval Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #0f172a; color: #f8fafc; margin: 0; padding: 2rem; }
        .container { max-width: 1000px; margin: 0 auto; }
        .header { background: linear-gradient(135deg, #1e293b, #0f172a); border: 1px solid #334155; padding: 2rem; border-radius: 12px; margin-bottom: 2rem; }
        h1 { margin: 0 0 0.5rem 0; color: #38bdf8; }
        .subtitle { color: #94a3b8; font-size: 1.1rem; }
        .card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 1.5rem; margin-bottom: 2rem; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem; }
        .metric-card { background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 1.25rem; text-align: center; }
        .metric-value { font-size: 2rem; font-weight: bold; color: #38bdf8; margin-top: 0.5rem; }
        .metric-label { color: #94a3b8; font-size: 0.9rem; text-transform: uppercase; }
        table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
        th, td { padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid #334155; }
        th { background: #0f172a; color: #94a3b8; font-weight: 600; }
        .chart-container { position: relative; height: 350px; width: 100%; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 mlx-specdec-eval Speculative Decoding Report</h1>
            <div class="subtitle">Target: {{ results.target_model }} | Draft: {{ results.draft_model }}</div>
        </div>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Optimal Draft Count (K)</div>
                <div class="metric-value">K = {{ results.optimal_k }}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Best Speedup Factor (S)</div>
                <div class="metric-value" style="color: #4ade80;">{{ "%.2f"|format(results.best_speedup) }}x</div>
            </div>
        </div>

        <div class="card">
            <h2>Performance Metrics Chart</h2>
            <div class="chart-container">
                <canvas id="speedupChart"></canvas>
            </div>
        </div>

        <div class="card">
            <h2>Parameter Sweep Breakdown</h2>
            <table>
                <thead>
                    <tr>
                        <th>Draft Tokens (K)</th>
                        <th>Baseline TPS</th>
                        <th>Speculative TPS</th>
                        <th>Acceptance Rate (α)</th>
                        <th>Speedup Factor (S)</th>
                    </tr>
                </thead>
                <tbody>
                    {% for k, data in results.k_sweeps.items() %}
                    <tr>
                        <td><strong>K = {{ k }}</strong></td>
                        <td>{{ "%.1f"|format(data.avg_baseline_tps) }}</td>
                        <td>{{ "%.1f"|format(data.avg_speculative_tps) }}</td>
                        <td>{{ "%.1f"|format(data.avg_acceptance_rate_alpha * 100) }}%</td>
                        <td><strong style="color: {{ '#4ade80' if data.speedup_factor > 1.0 else '#f87171' }}">{{ "%.2f"|format(data.speedup_factor) }}x</strong></td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>

    <script>
        const kLabels = [{{ k_labels_str }}];
        const speedupData = [{% for k, data in results.k_sweeps.items() %}{{ data.speedup_factor }}{{ "," if not loop.last }}{% endfor %}];
        const specTpsData = [{% for k, data in results.k_sweeps.items() %}{{ data.avg_speculative_tps }}{{ "," if not loop.last }}{% endfor %}];

        const ctx = document.getElementById('speedupChart').getContext('2d');
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: kLabels.map(k => `K = ${k}`),
                datasets: [
                    {
                        label: 'Speedup Factor (S)',
                        data: speedupData,
                        backgroundColor: '#38bdf8',
                        borderRadius: 6,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Speculative TPS',
                        data: specTpsData,
                        backgroundColor: '#4ade80',
                        borderRadius: 6,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { type: 'linear', position: 'left', title: { display: true, text: 'Speedup Ratio (x)', color: '#94a3b8' }, grid: { color: '#334155' } },
                    y1: { type: 'linear', position: 'right', title: { display: true, text: 'Tokens Per Second (TPS)', color: '#94a3b8' }, grid: { drawOnChartArea: false } }
                }
            }
        });
    </script>
</body>
</html>
"""

    k_labels_str = ", ".join(f"'{k}'" for k in results.get("k_sweeps", {}).keys())
    env = jinja2.Environment()
    template = env.from_string(html_template)
    rendered_html = template.render(results=results, k_labels_str=k_labels_str)

    with open(path, "w", encoding="utf-8") as f:
        f.write(rendered_html)

    console.print(f"📊 Visual HTML report saved to: [underline cyan]{filepath}[/underline cyan]")
