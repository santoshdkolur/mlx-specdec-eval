"""
Metrics Computation Engine for mlx-specdec-eval.
Calculates acceptance rate (alpha), TPS, Speedup factor (S), per-category task breakdowns,
and aggregates multi-run benchmark data.
"""

from typing import Dict, Any, List
import statistics


def calculate_acceptance_rate(accepted_draft_tokens: int, total_draft_proposed: int) -> float:
    """Calculates alpha = accepted / proposed."""
    if total_draft_proposed <= 0:
        return 0.0
    return max(0.0, min(1.0, accepted_draft_tokens / total_draft_proposed))


def calculate_tps(output_tokens: int, elapsed_seconds: float) -> float:
    """Calculates Tokens Per Second (TPS)."""
    if elapsed_seconds <= 0:
        return 0.0
    return output_tokens / elapsed_seconds


def calculate_speedup(speculative_tps: float, baseline_tps: float) -> float:
    """Calculates Speedup Factor S = speculative_tps / baseline_tps."""
    if baseline_tps <= 0:
        return 1.0
    return speculative_tps / baseline_tps


def aggregate_benchmark_results(
    target_model: str,
    draft_model: str,
    K_values: List[int],
    runs_by_k: Dict[int, List[Dict[str, Any]]],
    baseline_runs: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Aggregates per-prompt baseline and speculative runs across multiple draft token counts K,
    including detailed per-category task domain breakdowns.
    """
    base_tps_list = [r["tps"] for r in baseline_runs if r.get("tps", 0) > 0]
    avg_baseline_tps = statistics.mean(base_tps_list) if base_tps_list else 0.0

    k_summaries = {}

    for k in K_values:
        spec_runs = runs_by_k.get(k, [])
        spec_tps_list = [r["tps"] for r in spec_runs if r.get("tps", 0) > 0]
        alpha_list = [r.get("alpha", 0.0) for r in spec_runs]
        mem_list = [r.get("peak_memory_mb", 0.0) for r in spec_runs]

        avg_spec_tps = statistics.mean(spec_tps_list) if spec_tps_list else 0.0
        avg_alpha = statistics.mean(alpha_list) if alpha_list else 0.0
        avg_peak_mem_mb = statistics.mean(mem_list) if mem_list else 0.0

        # Check text output equality between baseline and speculative runs
        exact_matches = 0
        total_compared = 0
        base_map = {r.get("prompt_id"): r.get("text", "") for r in baseline_runs if "prompt_id" in r}

        for spec_r in spec_runs:
            p_id = spec_r.get("prompt_id")
            if p_id in base_map and base_map[p_id]:
                total_compared += 1
                if spec_r.get("text", "").strip() == base_map[p_id].strip():
                    exact_matches += 1

        match_ratio = (exact_matches / total_compared) if total_compared > 0 else 1.0
        speedup = calculate_speedup(avg_spec_tps, avg_baseline_tps)

        # Per-Category Task Breakdown for this K
        categories = {}
        for spec_r in spec_runs:
            # Use the explicit "category" field passed from cli.py
            cat = spec_r.get("category", "general").lower()
            if cat not in categories:
                categories[cat] = {
                    "spec_tps": [],
                    "base_tps": [],
                    "alpha": [],
                    "matches": 0,
                    "count": 0
                }
            categories[cat]["spec_tps"].append(spec_r.get("tps", 0.0))
            categories[cat]["alpha"].append(spec_r.get("alpha", 0.0))
            p_id = spec_r.get("prompt_id")
            if p_id in base_map and base_map[p_id]:
                categories[cat]["count"] += 1
                if spec_r.get("text", "").strip() == base_map[p_id].strip():
                    categories[cat]["matches"] += 1

        # Match baseline runs for category base_tps using explicit category field
        for base_r in baseline_runs:
            cat = base_r.get("category", "general").lower()
            if cat in categories and base_r.get("tps", 0.0) > 0:
                categories[cat]["base_tps"].append(base_r.get("tps", 0.0))


        cat_breakdown = {}
        for cat_name, cat_data in categories.items():
            c_spec_tps = statistics.mean(cat_data["spec_tps"]) if cat_data["spec_tps"] else 0.0
            c_base_tps = statistics.mean(cat_data["base_tps"]) if cat_data["base_tps"] else avg_baseline_tps
            c_alpha = statistics.mean(cat_data["alpha"]) if cat_data["alpha"] else 0.0
            c_speedup = calculate_speedup(c_spec_tps, c_base_tps)
            c_match = (cat_data["matches"] / cat_data["count"]) if cat_data["count"] > 0 else 1.0

            cat_breakdown[cat_name] = {
                "category": cat_name,
                "baseline_tps": round(c_base_tps, 2),
                "speculative_tps": round(c_spec_tps, 2),
                "alpha": round(c_alpha, 4),
                "speedup_factor": round(c_speedup, 3),
                "exact_match_rate": round(c_match, 4)
            }

        k_summaries[str(k)] = {
            "num_draft_tokens": k,
            "avg_speculative_tps": round(avg_spec_tps, 2),
            "avg_baseline_tps": round(avg_baseline_tps, 2),
            "speedup_factor": round(speedup, 3),
            "avg_acceptance_rate_alpha": round(avg_alpha, 4),
            "output_exact_match_rate": round(match_ratio, 4),
            "avg_peak_memory_mb": round(avg_peak_mem_mb, 2),
            "is_accelerated": speedup > 1.0,
            "category_breakdown": cat_breakdown,
            "prompt_runs": spec_runs
        }

    # Find optimal K
    best_k = max(
        K_values,
        key=lambda k: k_summaries[str(k)]["speedup_factor"]
    ) if K_values else (K_values[0] if K_values else 5)

    return {
        "target_model": target_model,
        "draft_model": draft_model,
        "baseline_summary": {
            "avg_baseline_tps": round(avg_baseline_tps, 2),
            "prompt_runs": baseline_runs
        },
        "k_sweeps": k_summaries,
        "optimal_k": best_k,
        "best_speedup": k_summaries[str(best_k)]["speedup_factor"] if k_summaries else 1.0,
        "category_breakdown": k_summaries[str(best_k)]["category_breakdown"] if str(best_k) in k_summaries else {}
    }
