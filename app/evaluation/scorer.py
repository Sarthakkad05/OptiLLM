"""
Efficiency Scorer — Cost-vs-Quality Composite Metric
Calculates cost-vs-quality efficiency ratio and efficiency rating tiers.
"""

from typing import Any, Dict



def compute_efficiency_score(quality_score: float, cost_usd: float) -> float:
    """
    Calculate composite cost-vs-quality efficiency score:
    efficiency_score = quality_score / max(cost_usd, 0.0001)
    """
    safe_cost = max(cost_usd, 0.0001)
    efficiency = quality_score / safe_cost
    return round(efficiency, 2)


def categorize_efficiency(efficiency_score: float) -> str:
    """
    Categorize efficiency score into qualitative value tiers.
    """
    if efficiency_score >= 10000.0:
        return "ultra_high"
    elif efficiency_score >= 2000.0:
        return "high"
    elif efficiency_score >= 500.0:
        return "moderate"
    else:
        return "low"


def evaluate_cost_efficiency(quality_score: float, cost_usd: float) -> Dict[str, Any]:
    eff_score = compute_efficiency_score(quality_score, cost_usd)
    rating = categorize_efficiency(eff_score)
    return {
        "efficiency_score": eff_score,
        "efficiency_rating": rating,
    }
