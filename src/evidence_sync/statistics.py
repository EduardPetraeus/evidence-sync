"""Meta-analysis statistical engine — DerSimonian-Laird random-effects model."""

from __future__ import annotations

import logging
import math
from datetime import date
from typing import Optional

import numpy as np
from scipy import stats

from evidence_sync.models import AnalysisResult, EffectMeasure, ReviewStatus, Study

logger = logging.getLogger(__name__)


def run_meta_analysis(
    studies: list[Study],
    effect_measure: EffectMeasure,
    topic: str = "",
    require_approval: bool = True,
) -> Optional[AnalysisResult]:
    """Run a random-effects meta-analysis using DerSimonian-Laird method.

    Args:
        studies: List of studies with extracted effect sizes and CIs.
        effect_measure: The effect measure type.
        topic: Topic identifier for the result.
        require_approval: If True, only include approved/corrected studies.
            Set to False for backward compatibility with existing workflows.

    Returns:
        AnalysisResult or None if insufficient data.
    """
    # Filter to studies with extractable data
    with_data = [s for s in studies if s.has_extractable_data and s.se_from_ci]

    if require_approval:
        # Auto-detect: if no studies have been explicitly reviewed, skip approval gate
        any_reviewed = any(
            s.review_status != ReviewStatus.PENDING for s in with_data
        )
        if any_reviewed:
            valid = [
                s for s in with_data
                if s.review_status in (ReviewStatus.APPROVED, ReviewStatus.CORRECTED)
            ]
            n_filtered = len(with_data) - len(valid)
            if n_filtered > 0:
                logger.info(
                    f"Review filter: {len(valid)} approved of {len(with_data)} "
                    f"with data ({n_filtered} pending/rejected)"
                )
        else:
            valid = with_data
            logger.info(
                "No studies reviewed yet — including all studies with data"
            )
    else:
        valid = with_data

    if len(valid) < 2:
        logger.warning(f"Need at least 2 studies with data, got {len(valid)}")
        return None

    effects = np.array([s.effect_size for s in valid], dtype=np.float64)
    ses = np.array([s.se_from_ci for s in valid], dtype=np.float64)
    variances = ses**2

    # Fixed-effect weights (inverse variance)
    w_fixed = 1.0 / variances

    # Fixed-effect pooled estimate
    theta_fixed = np.sum(w_fixed * effects) / np.sum(w_fixed)

    # Q statistic for heterogeneity
    q_stat = float(np.sum(w_fixed * (effects - theta_fixed) ** 2))
    df = len(valid) - 1
    q_p_value = float(1.0 - stats.chi2.cdf(q_stat, df)) if df > 0 else 1.0

    # DerSimonian-Laird tau-squared
    c = float(np.sum(w_fixed) - np.sum(w_fixed**2) / np.sum(w_fixed))
    tau_sq = max(0.0, (q_stat - df) / c) if c > 0 else 0.0

    # Random-effects weights
    w_random = 1.0 / (variances + tau_sq)

    # Pooled effect (random effects)
    pooled_effect = float(np.sum(w_random * effects) / np.sum(w_random))
    pooled_se = float(math.sqrt(1.0 / np.sum(w_random)))

    # 95% CI
    z = 1.96
    pooled_ci_lower = pooled_effect - z * pooled_se
    pooled_ci_upper = pooled_effect + z * pooled_se

    # P-value (two-tailed)
    z_score = pooled_effect / pooled_se if pooled_se > 0 else 0.0
    pooled_p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(z_score))))

    # I-squared
    i_squared = max(0.0, (q_stat - df) / q_stat * 100.0) if q_stat > 0 else 0.0

    # Egger's test for publication bias (requires >= 3 studies)
    egger_intercept = None
    egger_p_value = None
    if len(valid) >= 3:
        egger_intercept, egger_p_value = _eggers_test(effects, ses)

    return AnalysisResult(
        topic=topic,
        n_studies=len(valid),
        pooled_effect=pooled_effect,
        pooled_ci_lower=pooled_ci_lower,
        pooled_ci_upper=pooled_ci_upper,
        pooled_p_value=pooled_p_value,
        effect_measure=effect_measure,
        i_squared=i_squared,
        q_statistic=q_stat,
        q_p_value=q_p_value,
        tau_squared=tau_sq,
        egger_intercept=egger_intercept,
        egger_p_value=egger_p_value,
        analysis_date=date.today(),
        studies_included=[s.pmid for s in valid],
    )


def _eggers_test(effects: np.ndarray, ses: np.ndarray) -> tuple[Optional[float], Optional[float]]:
    """Egger's regression test for publication bias.

    Regresses standardized effect (effect/SE) on precision (1/SE).
    The intercept indicates asymmetry in the funnel plot.
    """
    if len(effects) < 3:
        return None, None

    precision = 1.0 / ses
    standardized = effects / ses

    try:
        slope, intercept, _, p_value, _ = stats.linregress(precision, standardized)
        return float(intercept), float(p_value)
    except Exception:
        logger.warning("Egger's test failed", exc_info=True)
        return None, None


def compute_study_weights(
    studies: list[Study],
    tau_squared: float = 0.0,
) -> list[tuple[Study, float]]:
    """Compute random-effects weights for each study.

    Returns list of (study, weight_pct) tuples where weights sum to 100.
    """
    valid = [(s, s.se_from_ci) for s in studies if s.has_extractable_data and s.se_from_ci]
    if not valid:
        return []

    weights = []
    for study, se in valid:
        w = 1.0 / (se**2 + tau_squared)
        weights.append((study, w))

    total_w = sum(w for _, w in weights)
    if total_w == 0:
        return []

    return [(s, (w / total_w) * 100.0) for s, w in weights]
