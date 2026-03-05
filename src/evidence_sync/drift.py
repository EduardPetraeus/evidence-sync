"""Evidence drift detection — compare analysis snapshots and trigger alerts."""

from __future__ import annotations

import logging
from typing import Optional

from evidence_sync.models import AnalysisResult, DriftResult, ReviewConfig

logger = logging.getLogger(__name__)


def detect_drift(
    current: AnalysisResult,
    previous: Optional[AnalysisResult],
    config: ReviewConfig,
) -> DriftResult:
    """Compare current analysis against previous snapshot.

    Args:
        current: The latest analysis result.
        previous: The previous analysis result (None if first run).
        config: Review config with drift thresholds.

    Returns:
        DriftResult with change metrics and alert status.
    """
    if previous is None:
        return DriftResult(
            topic=current.topic,
            previous_effect=0.0,
            current_effect=current.pooled_effect,
            effect_change_pct=0.0,
            significance_flipped=False,
            heterogeneity_change=0.0,
            alert_triggered=False,
            alert_reasons=["First analysis — no previous snapshot to compare"],
        )

    # Effect size change
    if previous.pooled_effect != 0:
        effect_change_pct = abs(
            (current.pooled_effect - previous.pooled_effect) / previous.pooled_effect * 100.0
        )
    else:
        effect_change_pct = abs(current.pooled_effect) * 100.0

    # Significance flip
    significance_flipped = current.significant != previous.significant

    # Heterogeneity change
    heterogeneity_change = abs(current.i_squared - previous.i_squared)

    # Determine alerts
    alert_reasons = []

    if effect_change_pct > config.effect_change_threshold_pct:
        alert_reasons.append(
            f"Effect size changed by {effect_change_pct:.1f}% "
            f"(threshold: {config.effect_change_threshold_pct}%)"
        )

    if significance_flipped:
        prev_sig = "significant" if previous.significant else "non-significant"
        curr_sig = "significant" if current.significant else "non-significant"
        alert_reasons.append(f"Significance flipped: {prev_sig} -> {curr_sig}")

    if heterogeneity_change > config.heterogeneity_change_threshold:
        alert_reasons.append(
            f"Heterogeneity (I²) changed by {heterogeneity_change:.1f}pp "
            f"(threshold: {config.heterogeneity_change_threshold}pp)"
        )

    alert_triggered = len(alert_reasons) > 0

    if alert_triggered:
        logger.warning(f"Drift alert for {current.topic}: {'; '.join(alert_reasons)}")

    return DriftResult(
        topic=current.topic,
        previous_effect=previous.pooled_effect,
        current_effect=current.pooled_effect,
        effect_change_pct=effect_change_pct,
        significance_flipped=significance_flipped,
        heterogeneity_change=heterogeneity_change,
        alert_triggered=alert_triggered,
        alert_reasons=alert_reasons,
    )
