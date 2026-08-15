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


def calculate_wasted_draft_ratio(total_proposed: int, total_accepted: int, output_tokens: int) -> float:
    """Calculates Wasted Draft Ratio (WDR) = (Proposed - Accepted) / Output Tokens."""
    if output_tokens <= 0:
        return 0.0
    wasted = max(0, total_proposed - total_accepted)
    return round(wasted / output_tokens, 4)


def aggregate_adaptive_results(
    adaptive_runs: List[Dict[str, Any]],
    baseline_runs: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Aggregates per-prompt adaptive speculative runs, including mean effective K,
    wasted draft ratio, alpha, and per-category task domain breakdowns.
    """
    if not adaptive_runs:
        return {}

    base_tps_list = [r["tps"] for r in baseline_runs if r.get("tps", 0) > 0]
    avg_baseline_tps = statistics.mean(base_tps_list) if base_tps_list else 0.0

    adapt_tps_list = [r["tps"] for r in adaptive_runs if r.get("tps", 0) > 0]
    alpha_list = [r.get("alpha", 0.0) for r in adaptive_runs]
    effective_k_list = [r.get("mean_effective_k", 3.0) for r in adaptive_runs]
    wdr_list = [r.get("wasted_draft_ratio", 0.0) for r in adaptive_runs]
    mem_list = [r.get("peak_memory_mb", 0.0) for r in adaptive_runs]

    avg_adaptive_tps = statistics.mean(adapt_tps_list) if adapt_tps_list else 0.0
    avg_alpha = statistics.mean(alpha_list) if alpha_list else 0.0
    mean_effective_k = statistics.mean(effective_k_list) if effective_k_list else 3.0
    avg_wdr = statistics.mean(wdr_list) if wdr_list else 0.0
    avg_peak_mem_mb = statistics.mean(mem_list) if mem_list else 0.0

    # Output exact match comparison against baseline
    exact_matches = 0
    total_compared = 0
    base_map = {r.get("prompt_id"): r.get("text", "") for r in baseline_runs if "prompt_id" in r}

    for r in adaptive_runs:
        p_id = r.get("prompt_id")
        if p_id in base_map and base_map[p_id]:
            total_compared += 1
            if r.get("text", "").strip() == base_map[p_id].strip():
                exact_matches += 1

    match_ratio = (exact_matches / total_compared) if total_compared > 0 else 1.0
    speedup = calculate_speedup(avg_adaptive_tps, avg_baseline_tps)

    # Per-Category Task Breakdown
    categories = {}
    for r in adaptive_runs:
        cat = r.get("category", "general").lower()
        if cat not in categories:
            categories[cat] = {
                "spec_tps": [],
                "base_tps": [],
                "alpha": [],
                "effective_k": [],
                "wdr": [],
                "matches": 0,
                "count": 0
            }
        categories[cat]["spec_tps"].append(r.get("tps", 0.0))
        categories[cat]["alpha"].append(r.get("alpha", 0.0))
        categories[cat]["effective_k"].append(r.get("mean_effective_k", 3.0))
        categories[cat]["wdr"].append(r.get("wasted_draft_ratio", 0.0))
        p_id = r.get("prompt_id")
        if p_id in base_map and base_map[p_id]:
            categories[cat]["count"] += 1
            if r.get("text", "").strip() == base_map[p_id].strip():
                categories[cat]["matches"] += 1

    for base_r in baseline_runs:
        cat = base_r.get("category", "general").lower()
        if cat in categories and base_r.get("tps", 0.0) > 0:
            categories[cat]["base_tps"].append(base_r.get("tps", 0.0))

    cat_breakdown = {}
    for cat_name, cat_data in categories.items():
        c_spec_tps = statistics.mean(cat_data["spec_tps"]) if cat_data["spec_tps"] else 0.0
        c_base_tps = statistics.mean(cat_data["base_tps"]) if cat_data["base_tps"] else avg_baseline_tps
        c_alpha = statistics.mean(cat_data["alpha"]) if cat_data["alpha"] else 0.0
        c_eff_k = statistics.mean(cat_data["effective_k"]) if cat_data["effective_k"] else mean_effective_k
        c_wdr = statistics.mean(cat_data["wdr"]) if cat_data["wdr"] else 0.0
        c_speedup = calculate_speedup(c_spec_tps, c_base_tps)
        c_match = (cat_data["matches"] / cat_data["count"]) if cat_data["count"] > 0 else 1.0

        cat_breakdown[cat_name] = {
            "category": cat_name,
            "baseline_tps": round(c_base_tps, 2),
            "speculative_tps": round(c_spec_tps, 2),
            "alpha": round(c_alpha, 4),
            "mean_effective_k": round(c_eff_k, 2),
            "wasted_draft_ratio": round(c_wdr, 4),
            "speedup_factor": round(c_speedup, 3),
            "exact_match_rate": round(c_match, 4)
        }

    return {
        "mode": "adaptive",
        "avg_adaptive_tps": round(avg_adaptive_tps, 2),
        "avg_baseline_tps": round(avg_baseline_tps, 2),
        "speedup_factor": round(speedup, 3),
        "avg_acceptance_rate_alpha": round(avg_alpha, 4),
        "mean_effective_k": round(mean_effective_k, 2),
        "avg_wasted_draft_ratio": round(avg_wdr, 4),
        "output_exact_match_rate": round(match_ratio, 4),
        "avg_peak_memory_mb": round(avg_peak_mem_mb, 2),
        "is_accelerated": speedup > 1.0,
        "category_breakdown": cat_breakdown,
        "prompt_runs": adaptive_runs
    }


def verify_output_determinism(
    baseline_runs: List[Dict[str, Any]],
    fixed_runs_by_k: Dict[int, List[Dict[str, Any]]],
    adaptive_runs: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Verifies that speculative decoding outputs (Fixed K and Adaptive K)
    are 100% token-for-token identical to Baseline Target generation at temperature 0.0.
    """
    base_map = {r.get("prompt_id", f"p_{i}"): r.get("text", "") for i, r in enumerate(baseline_runs)}
    total_comparisons = 0
    exact_matches = 0
    divergences = []

    # Check Fixed K runs
    for k, runs in fixed_runs_by_k.items():
        for i, r in enumerate(runs):
            p_id = r.get("prompt_id", f"p_{i}")
            if p_id in base_map and base_map[p_id]:
                total_comparisons += 1
                base_text = base_map[p_id].strip()
                spec_text = r.get("text", "").strip()
                if base_text == spec_text:
                    exact_matches += 1
                else:
                    divergences.append({
                        "mode": f"fixed_k_{k}",
                        "prompt_id": p_id,
                        "baseline_len": len(base_text),
                        "speculative_len": len(spec_text)
                    })

    # Check Adaptive runs
    if adaptive_runs:
        for i, r in enumerate(adaptive_runs):
            p_id = r.get("prompt_id", f"p_{i}")
            if p_id in base_map and base_map[p_id]:
                total_comparisons += 1
                base_text = base_map[p_id].strip()
                spec_text = r.get("text", "").strip()
                if base_text == spec_text:
                    exact_matches += 1
                else:
                    divergences.append({
                        "mode": "adaptive",
                        "prompt_id": p_id,
                        "baseline_len": len(base_text),
                        "speculative_len": len(spec_text)
                    })

    match_percentage = (exact_matches / total_comparisons * 100.0) if total_comparisons > 0 else 100.0

    return {
        "total_comparisons": total_comparisons,
        "exact_matches": exact_matches,
        "divergences_count": len(divergences),
        "match_percentage": round(match_percentage, 2),
        "all_identical": len(divergences) == 0,
        "divergences": divergences
    }


def aggregate_benchmark_results(
    target_model: str,
    draft_model: str,
    K_values: List[int],
    runs_by_k: Dict[int, List[Dict[str, Any]]],
    baseline_runs: List[Dict[str, Any]],
    adaptive_runs: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Aggregates per-prompt baseline and speculative runs across multiple draft token counts K,
    including Adaptive K runs and detailed per-category task domain breakdowns.
    """
    base_tps_list = [r["tps"] for r in baseline_runs if r.get("tps", 0) > 0]
    avg_baseline_tps = statistics.mean(base_tps_list) if base_tps_list else 0.0

    k_summaries = {}

    for k in K_values:
        spec_runs = runs_by_k.get(k, [])
        spec_tps_list = [r["tps"] for r in spec_runs if r.get("tps", 0) > 0]
        alpha_list = [r.get("alpha", 0.0) for r in spec_runs]
        mem_list = [r.get("peak_memory_mb", 0.0) for r in spec_runs]

        # Calculate Wasted Draft Ratio for this K
        wdr_list = []
        for r in spec_runs:
            prop = r.get("total_draft_proposed", 0)
            acc = r.get("accepted_draft_tokens", 0)
            toks = r.get("output_tokens", 1)
            wdr_list.append(calculate_wasted_draft_ratio(prop, acc, toks))

        avg_spec_tps = statistics.mean(spec_tps_list) if spec_tps_list else 0.0
        avg_alpha = statistics.mean(alpha_list) if alpha_list else 0.0
        avg_wdr = statistics.mean(wdr_list) if wdr_list else 0.0
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
            cat = spec_r.get("category", "general").lower()
            if cat not in categories:
                categories[cat] = {
                    "spec_tps": [],
                    "base_tps": [],
                    "alpha": [],
                    "wdr": [],
                    "matches": 0,
                    "count": 0
                }
            categories[cat]["spec_tps"].append(spec_r.get("tps", 0.0))
            categories[cat]["alpha"].append(spec_r.get("alpha", 0.0))
            prop = spec_r.get("total_draft_proposed", 0)
            acc = spec_r.get("accepted_draft_tokens", 0)
            toks = spec_r.get("output_tokens", 1)
            categories[cat]["wdr"].append(calculate_wasted_draft_ratio(prop, acc, toks))

            p_id = spec_r.get("prompt_id")
            if p_id in base_map and base_map[p_id]:
                categories[cat]["count"] += 1
                if spec_r.get("text", "").strip() == base_map[p_id].strip():
                    categories[cat]["matches"] += 1

        for base_r in baseline_runs:
            cat = base_r.get("category", "general").lower()
            if cat in categories and base_r.get("tps", 0.0) > 0:
                categories[cat]["base_tps"].append(base_r.get("tps", 0.0))

        cat_breakdown = {}
        for cat_name, cat_data in categories.items():
            c_spec_tps = statistics.mean(cat_data["spec_tps"]) if cat_data["spec_tps"] else 0.0
            c_base_tps = statistics.mean(cat_data["base_tps"]) if cat_data["base_tps"] else avg_baseline_tps
            c_alpha = statistics.mean(cat_data["alpha"]) if cat_data["alpha"] else 0.0
            c_wdr = statistics.mean(cat_data["wdr"]) if cat_data["wdr"] else 0.0
            c_speedup = calculate_speedup(c_spec_tps, c_base_tps)
            c_match = (cat_data["matches"] / cat_data["count"]) if cat_data["count"] > 0 else 1.0

            cat_breakdown[cat_name] = {
                "category": cat_name,
                "baseline_tps": round(c_base_tps, 2),
                "speculative_tps": round(c_spec_tps, 2),
                "alpha": round(c_alpha, 4),
                "wasted_draft_ratio": round(c_wdr, 4),
                "speedup_factor": round(c_speedup, 3),
                "exact_match_rate": round(c_match, 4)
            }

        k_summaries[str(k)] = {
            "num_draft_tokens": k,
            "avg_speculative_tps": round(avg_spec_tps, 2),
            "avg_baseline_tps": round(avg_baseline_tps, 2),
            "speedup_factor": round(speedup, 3),
            "avg_acceptance_rate_alpha": round(avg_alpha, 4),
            "avg_wasted_draft_ratio": round(avg_wdr, 4),
            "output_exact_match_rate": round(match_ratio, 4),
            "avg_peak_memory_mb": round(avg_peak_mem_mb, 2),
            "is_accelerated": speedup > 1.0,
            "category_breakdown": cat_breakdown,
            "prompt_runs": spec_runs
        }

    # Process Adaptive summary if present
    adaptive_summary = aggregate_adaptive_results(adaptive_runs, baseline_runs) if adaptive_runs else None

    # Determinism verification check
    determinism = verify_output_determinism(baseline_runs, runs_by_k, adaptive_runs)

    # Find optimal Fixed K
    best_k = max(
        K_values,
        key=lambda k: k_summaries[str(k)]["speedup_factor"]
    ) if K_values else 3

    # Build Head-to-Head comparison list
    head_to_head = []
    head_to_head.append({
        "mode": "Baseline (M_T only)",
        "decode_tps": round(avg_baseline_tps, 2),
        "speedup": 1.0,
        "alpha": None,
        "effective_k": 0.0,
        "wasted_draft_ratio": 0.0,
        "output_match": 1.0,
        "status": "Baseline"
    })

    for k in K_values:
        k_str = str(k)
        if k_str in k_summaries:
            s_data = k_summaries[k_str]
            s_val = s_data["speedup_factor"]
            head_to_head.append({
                "mode": f"Fixed K={k}",
                "decode_tps": s_data["avg_speculative_tps"],
                "speedup": s_val,
                "alpha": s_data["avg_acceptance_rate_alpha"],
                "effective_k": float(k),
                "wasted_draft_ratio": s_data["avg_wasted_draft_ratio"],
                "output_match": s_data["output_exact_match_rate"],
                "status": "Accelerated" if s_val > 1.0 else ("Neutral" if s_val == 1.0 else "Degraded")
            })

    if adaptive_summary:
        a_spd = adaptive_summary["speedup_factor"]
        head_to_head.append({
            "mode": "Adaptive K (Dynamic)",
            "decode_tps": adaptive_summary["avg_adaptive_tps"],
            "speedup": a_spd,
            "alpha": adaptive_summary["avg_acceptance_rate_alpha"],
            "effective_k": adaptive_summary["mean_effective_k"],
            "wasted_draft_ratio": adaptive_summary["avg_wasted_draft_ratio"],
            "output_match": adaptive_summary["output_exact_match_rate"],
            "status": "Accelerated" if a_spd > 1.0 else ("Neutral" if a_spd == 1.0 else "Degraded")
        })

    return {
        "target_model": target_model,
        "draft_model": draft_model,
        "baseline_summary": {
            "avg_baseline_tps": round(avg_baseline_tps, 2),
            "prompt_runs": baseline_runs
        },
        "k_sweeps": k_summaries,
        "adaptive_summary": adaptive_summary,
        "head_to_head": head_to_head,
        "output_determinism": determinism,
        "optimal_k": best_k,
        "best_speedup": k_summaries[str(best_k)]["speedup_factor"] if k_summaries and str(best_k) in k_summaries else 1.0,
        "category_breakdown": k_summaries[str(best_k)]["category_breakdown"] if k_summaries and str(best_k) in k_summaries else {}
    }

