"""
Unit tests for spec_bench.metrics module.
"""

from spec_bench.metrics import (
    calculate_acceptance_rate,
    calculate_tps,
    calculate_speedup,
    calculate_wasted_draft_ratio,
    aggregate_adaptive_results,
    verify_output_determinism,
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


def test_calculate_wasted_draft_ratio():
    # 100 proposed, 80 accepted, 128 output tokens -> 20 / 128 = 0.1562
    assert calculate_wasted_draft_ratio(100, 80, 128) == 0.1562
    assert calculate_wasted_draft_ratio(50, 50, 100) == 0.0
    assert calculate_wasted_draft_ratio(50, 0, 100) == 0.5
    assert calculate_wasted_draft_ratio(10, 5, 0) == 0.0


def test_verify_output_determinism():
    baseline_runs = [{"prompt_id": "p1", "text": "Hello world"}]
    fixed_runs_by_k = {
        3: [{"prompt_id": "p1", "text": "Hello world"}]
    }
    adaptive_runs = [{"prompt_id": "p1", "text": "Hello world"}]

    det = verify_output_determinism(baseline_runs, fixed_runs_by_k, adaptive_runs)
    assert det["all_identical"] is True
    assert det["match_percentage"] == 100.0
    assert det["divergences_count"] == 0

    # Test divergence
    divergent_runs = [{"prompt_id": "p1", "text": "Different response"}]
    det_div = verify_output_determinism(baseline_runs, fixed_runs_by_k, divergent_runs)
    assert det_div["all_identical"] is False
    assert det_div["divergences_count"] == 1


def test_aggregate_benchmark_results():
    baseline_runs = [{"prompt_id": "p1", "tps": 20.0, "elapsed_time_s": 2.0, "text": "Resp"}]
    runs_by_k = {
        3: [{"prompt_id": "p1", "tps": 30.0, "alpha": 0.8, "peak_memory_mb": 4000.0, "total_draft_proposed": 30, "accepted_draft_tokens": 24, "output_tokens": 128, "text": "Resp"}],
        5: [{"prompt_id": "p1", "tps": 40.0, "alpha": 0.7, "peak_memory_mb": 4500.0, "total_draft_proposed": 50, "accepted_draft_tokens": 35, "output_tokens": 128, "text": "Resp"}]
    }
    adaptive_runs = [
        {"prompt_id": "p1", "tps": 38.0, "alpha": 0.82, "mean_effective_k": 3.5, "wasted_draft_ratio": 0.12, "peak_memory_mb": 4200.0, "text": "Resp"}
    ]

    results = aggregate_benchmark_results("target_m", "draft_m", [3, 5], runs_by_k, baseline_runs, adaptive_runs)

    assert results["target_model"] == "target_m"
    assert results["draft_model"] == "draft_m"
    assert results["optimal_k"] == 5
    assert results["best_speedup"] == 2.0
    assert "3" in results["k_sweeps"]
    assert "5" in results["k_sweeps"]
    assert results["adaptive_summary"] is not None
    assert results["adaptive_summary"]["avg_adaptive_tps"] == 38.0
    assert len(results["head_to_head"]) == 4  # Baseline + K=3 + K=5 + Adaptive K
    assert results["output_determinism"]["all_identical"] is True

