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


def test_cli_clean():
    runner = CliRunner()
    result = runner.invoke(main, ["clean"])
    assert result.exit_code == 0
    assert "Cleaning up spec-bench artifacts" in result.output


