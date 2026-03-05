"""Tests for evidence drift detection."""

from __future__ import annotations

import pytest

from evidence_sync.drift import detect_drift
from evidence_sync.models import AnalysisResult, EffectMeasure, ReviewConfig


@pytest.fixture
def config():
    return ReviewConfig(
        topic_id="test",
        topic_name="Test",
        search_query="test",
        effect_measure=EffectMeasure.ODDS_RATIO,
        primary_outcome="test",
        effect_change_threshold_pct=10.0,
        heterogeneity_change_threshold=15.0,
    )


def _make_result(
    pooled_effect: float = 1.5,
    p_value: float = 0.01,
    i_squared: float = 30.0,
) -> AnalysisResult:
    return AnalysisResult(
        topic="test",
        n_studies=5,
        pooled_effect=pooled_effect,
        pooled_ci_lower=pooled_effect - 0.3,
        pooled_ci_upper=pooled_effect + 0.3,
        pooled_p_value=p_value,
        effect_measure=EffectMeasure.ODDS_RATIO,
        i_squared=i_squared,
        q_statistic=5.0,
        q_p_value=0.3,
        tau_squared=0.01,
    )


class TestDetectDrift:
    def test_first_run_no_alert(self, config):
        current = _make_result()
        drift = detect_drift(current, None, config)
        assert drift.alert_triggered is False
        assert drift.previous_effect == 0.0

    def test_no_drift_similar_results(self, config):
        previous = _make_result(pooled_effect=1.50)
        current = _make_result(pooled_effect=1.52)
        drift = detect_drift(current, previous, config)
        assert drift.alert_triggered is False
        assert drift.effect_change_pct < 10.0

    def test_effect_size_drift_alert(self, config):
        previous = _make_result(pooled_effect=1.50)
        current = _make_result(pooled_effect=1.80)  # 20% change
        drift = detect_drift(current, previous, config)
        assert drift.alert_triggered is True
        assert drift.effect_change_pct > 10.0
        assert any("Effect size" in r for r in drift.alert_reasons)

    def test_significance_flip_alert(self, config):
        previous = _make_result(p_value=0.04)  # significant
        current = _make_result(p_value=0.06)  # not significant
        drift = detect_drift(current, previous, config)
        assert drift.significance_flipped is True
        assert drift.alert_triggered is True
        assert any("Significance flipped" in r for r in drift.alert_reasons)

    def test_heterogeneity_change_alert(self, config):
        previous = _make_result(i_squared=30.0)
        current = _make_result(i_squared=50.0)  # 20pp change
        drift = detect_drift(current, previous, config)
        assert drift.alert_triggered is True
        assert drift.heterogeneity_change == 20.0
        assert any("Heterogeneity" in r for r in drift.alert_reasons)

    def test_multiple_alerts(self, config):
        previous = _make_result(pooled_effect=1.50, p_value=0.04, i_squared=30.0)
        current = _make_result(pooled_effect=1.80, p_value=0.06, i_squared=50.0)
        drift = detect_drift(current, previous, config)
        assert drift.alert_triggered is True
        assert len(drift.alert_reasons) == 3  # effect + sig flip + heterogeneity
