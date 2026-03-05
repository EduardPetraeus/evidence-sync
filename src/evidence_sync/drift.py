"""Evidence drift detection — compare analysis snapshots and trigger alerts."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

import httpx

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


def _validate_webhook_url(url: str) -> str:
    """Validate webhook URL scheme to prevent SSRF to internal services.

    Only https:// URLs are accepted.

    Args:
        url: The webhook URL to validate.

    Returns:
        The validated URL.

    Raises:
        ValueError: If the URL scheme is not https.
    """
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(
            f"Webhook URL must use https:// scheme, got '{parsed.scheme}://'. "
            "Plain http is rejected to prevent SSRF to internal services."
        )
    return url


def send_alert(drift: DriftResult, config: ReviewConfig) -> bool:
    """Send drift alert via configured channels (webhook, email).

    Args:
        drift: The drift detection result.
        config: Review config with alert_webhook and/or alert_email.

    Returns:
        True if an alert was sent successfully, False otherwise.
    """
    sent = False

    if config.alert_webhook:
        try:
            url = _validate_webhook_url(config.alert_webhook)
            payload = {
                "topic": drift.topic,
                "previous_effect": drift.previous_effect,
                "current_effect": drift.current_effect,
                "change_pct": drift.effect_change_pct,
                "reasons": drift.alert_reasons,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            response = httpx.post(url, json=payload, timeout=10.0)
            response.raise_for_status()
            logger.info(f"Webhook alert sent for {drift.topic}")
            sent = True
        except ValueError as e:
            logger.error(f"Invalid webhook URL: {e}")
        except httpx.HTTPError as e:
            logger.error(f"Failed to send webhook alert for {drift.topic}: {e}")

    if config.alert_email:
        logger.warning(
            f"Email alerts not implemented — would send to {config.alert_email} "
            f"for topic {drift.topic}"
        )

    return sent
