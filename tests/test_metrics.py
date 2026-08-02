"""
Unit tests for spec_bench.metrics module.
"""

from spec_bench.metrics import (
    calculate_acceptance_rate,
    calculate_tps,
    calculate_speedup,
    aggregate_benchmark_results
)


def test_calculate_acceptance_rate():
    assert calculate_acceptance_rate(50, 100) == 0.5
    assert calculate_acceptance_rate(0, 100) == 0.0
    assert calculate_acceptance_rate(100, 100) == 1.0
    assert calculate_acceptance_rate(10, 0) == 0.0


def test_calculate_tps():
    assert calculate_tps(100, 2.0) == 50.0
    assert calculate_tps(0, 2.0) == 0.0
    assert calculate_tps(100, 0.0) == 0.0


def test_calculate_speedup():
    assert calculate_speedup(50.0, 25.0) == 2.0
    assert calculate_speedup(25.0, 25.0) == 1.0
    assert calculate_speedup(12.5, 25.0) == 0.5
    assert calculate_speedup(50.0, 0.0) == 1.0


def test_aggregate_benchmark_results():
    baseline_runs = [{"tps": 20.0, "elapsed_time_s": 2.0}]
    runs_by_k = {
        3: [{"tps": 30.0, "alpha": 0.8, "peak_memory_mb": 4000.0}],
        5: [{"tps": 40.0, "alpha": 0.7, "peak_memory_mb": 4500.0}]
    }
    results = aggregate_benchmark_results("target_m", "draft_m", [3, 5], runs_by_k, baseline_runs)

    assert results["target_model"] == "target_m"
    assert results["draft_model"] == "draft_m"
    assert results["optimal_k"] == 5
    assert results["best_speedup"] == 2.0
    assert "3" in results["k_sweeps"]
    assert "5" in results["k_sweeps"]
    assert results["k_sweeps"]["5"]["speedup_factor"] == 2.0
