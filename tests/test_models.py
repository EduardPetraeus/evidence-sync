"""Tests for core data models."""

from __future__ import annotations

from datetime import date

from evidence_sync.models import (
    AnalysisResult,
    BiasRisk,
    EffectMeasure,
    RiskOfBias,
    Study,
)


class TestStudy:
    def test_sample_size_total(self):
        study = Study(
            pmid="1", title="", authors=[], journal="", publication_date=date(2020, 1, 1),
            abstract="", sample_size_treatment=100, sample_size_control=100,
        )
        assert study.sample_size_total == 200

    def test_sample_size_total_none(self):
        study = Study(
            pmid="1", title="", authors=[], journal="", publication_date=date(2020, 1, 1),
            abstract="",
        )
        assert study.sample_size_total is None

    def test_has_extractable_data(self):
        study = Study(
            pmid="1", title="", authors=[], journal="", publication_date=date(2020, 1, 1),
            abstract="", effect_size=1.5, ci_lower=1.1, ci_upper=2.1,
        )
        assert study.has_extractable_data is True

    def test_has_extractable_data_missing_ci(self):
        study = Study(
            pmid="1", title="", authors=[], journal="", publication_date=date(2020, 1, 1),
            abstract="", effect_size=1.5,
        )
        assert study.has_extractable_data is False

    def test_se_from_ci(self):
        study = Study(
            pmid="1", title="", authors=[], journal="", publication_date=date(2020, 1, 1),
            abstract="", ci_lower=1.0, ci_upper=2.0,
        )
        # SE = (2.0 - 1.0) / (2 * 1.96) = 0.2551...
        se = study.se_from_ci
        assert se is not None
        assert abs(se - 0.2551) < 0.001

    def test_weight_inverse_variance(self):
        study = Study(
            pmid="1", title="", authors=[], journal="", publication_date=date(2020, 1, 1),
            abstract="", ci_lower=1.0, ci_upper=2.0,
        )
        w = study.weight_inverse_variance
        assert w is not None
        assert w > 0


class TestRiskOfBias:
    def test_overall_low(self):
        rob = RiskOfBias(
            random_sequence_generation=BiasRisk.LOW,
            allocation_concealment=BiasRisk.LOW,
            blinding_participants=BiasRisk.LOW,
            blinding_outcome=BiasRisk.LOW,
            incomplete_outcome=BiasRisk.LOW,
            selective_reporting=BiasRisk.LOW,
        )
        assert rob.overall == BiasRisk.LOW

    def test_overall_high_if_any_high(self):
        rob = RiskOfBias(
            random_sequence_generation=BiasRisk.LOW,
            allocation_concealment=BiasRisk.HIGH,
            blinding_participants=BiasRisk.LOW,
            blinding_outcome=BiasRisk.LOW,
            incomplete_outcome=BiasRisk.LOW,
            selective_reporting=BiasRisk.LOW,
        )
        assert rob.overall == BiasRisk.HIGH

    def test_overall_unclear_if_mixed(self):
        rob = RiskOfBias(
            random_sequence_generation=BiasRisk.LOW,
            allocation_concealment=BiasRisk.UNCLEAR,
            blinding_participants=BiasRisk.LOW,
            blinding_outcome=BiasRisk.LOW,
            incomplete_outcome=BiasRisk.LOW,
            selective_reporting=BiasRisk.LOW,
        )
        assert rob.overall == BiasRisk.UNCLEAR


class TestAnalysisResult:
    def test_significant(self):
        result = AnalysisResult(
            topic="test", n_studies=5, pooled_effect=1.5,
            pooled_ci_lower=1.1, pooled_ci_upper=2.0,
            pooled_p_value=0.01, effect_measure=EffectMeasure.ODDS_RATIO,
            i_squared=30.0, q_statistic=5.0, q_p_value=0.3, tau_squared=0.01,
        )
        assert result.significant is True

    def test_not_significant(self):
        result = AnalysisResult(
            topic="test", n_studies=2, pooled_effect=1.1,
            pooled_ci_lower=0.8, pooled_ci_upper=1.5,
            pooled_p_value=0.10, effect_measure=EffectMeasure.ODDS_RATIO,
            i_squared=10.0, q_statistic=1.0, q_p_value=0.8, tau_squared=0.0,
        )
        assert result.significant is False

    def test_high_heterogeneity(self):
        result = AnalysisResult(
            topic="test", n_studies=5, pooled_effect=1.5,
            pooled_ci_lower=1.1, pooled_ci_upper=2.0,
            pooled_p_value=0.01, effect_measure=EffectMeasure.ODDS_RATIO,
            i_squared=80.0, q_statistic=20.0, q_p_value=0.001, tau_squared=0.5,
        )
        assert result.high_heterogeneity is True
