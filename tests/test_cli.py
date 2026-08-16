"""
Integration tests for spec_bench.cli Click subcommands.
"""

from click.testing import CliRunner
from spec_bench.cli import main


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_cli_check_env():
    runner = CliRunner()
    result = runner.invoke(main, ["check-env"])
    assert result.exit_code == 0
    assert "System Environment Diagnostics" in result.output


def test_cli_prompts():
    runner = CliRunner()
    result = runner.invoke(main, ["prompts", "--category", "code"])
    assert result.exit_code == 0
    assert "Evaluation Prompts Suite" in result.output


def test_cli_validate():
    runner = CliRunner()
    result = runner.invoke(main, ["validate", "--target", "dummy_t", "--draft", "dummy_d", "--force"])
    assert result.exit_code == 0
    assert "Pre-Flight Check" in result.output


def test_cli_run_synthetic(tmp_path):
    runner = CliRunner()
    json_out = str(tmp_path / "results.json")
    md_out = str(tmp_path / "summary.md")
    html_out = str(tmp_path / "report.html")

    result = runner.invoke(main, [
        "run",
        "--target", "dummy_target",
        "--draft", "dummy_draft",
        "--num-draft-tokens", "3,5",
        "--category", "code",
        "--export-json", json_out,
        "--export-markdown", md_out,
        "--export-html", html_out,
        "--skip-preflight"
    ])
    assert result.exit_code == 0
    assert "Speculative Decoding Report" in result.output
    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "summary.md").exists()
    assert (tmp_path / "report.html").exists()


def test_cli_run_single_prompt():
    runner = CliRunner()
    result = runner.invoke(main, [
        "run",
        "--target", "dummy_target",
        "--draft", "dummy_draft",
        "--prompt-id", "code_quicksort",
        "--skip-preflight"
    ])
    assert result.exit_code == 0
    assert "Prompts: 1" in result.output



def test_cli_run_adaptive_flags(tmp_path):
    import json
    runner = CliRunner()
    json_out = str(tmp_path / "results_adaptive.json")

    result = runner.invoke(main, [
        "run",
        "--target", "dummy_target",
        "--draft", "dummy_draft",
        "--num-draft-tokens", "3,5",
        "--adaptive",
        "--min-k", "2",
        "--max-k", "6",
        "--initial-k", "4",
        "--export-json", json_out,
        "--skip-preflight"
    ])
    assert result.exit_code == 0
    assert "Head-to-Head" in result.output
    assert "Adaptive" in result.output

    with open(json_out, "r") as f:
        data = json.load(f)
    assert data["adaptive_summary"] is not None
    assert "head_to_head" in data
    assert "output_determinism" in data


def test_cli_run_no_adaptive():
    runner = CliRunner()
    result = runner.invoke(main, [
        "run",
        "--target", "dummy_target",
        "--draft", "dummy_draft",
        "--no-adaptive",
        "--skip-preflight"
    ])
    assert result.exit_code == 0
    assert "Adaptive: False" in result.output


def test_cli_clean():
    runner = CliRunner()
    result = runner.invoke(main, ["clean"])
    assert result.exit_code == 0
    assert "Cleaning up mlx-specdec-eval artifacts" in result.output



