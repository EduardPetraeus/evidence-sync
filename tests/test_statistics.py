"""Tests for the meta-analysis statistical engine."""

from __future__ import annotations

from evidence_sync.models import EffectMeasure
from evidence_sync.statistics import compute_study_weights, run_meta_analysis
from tests.conftest import make_study


class TestRunMetaAnalysis:
    def test_basic_meta_analysis(self, sample_studies):
        result = run_meta_analysis(
            sample_studies, EffectMeasure.ODDS_RATIO, topic="test"
        )
        assert result is not None
        assert result.n_studies == 5
        assert result.topic == "test"
        assert result.effect_measure == EffectMeasure.ODDS_RATIO

        # Pooled effect should be between min and max individual effects
        effects = [s.effect_size for s in sample_studies]
        assert min(effects) <= result.pooled_effect <= max(effects)

        # CI should contain the pooled effect
        assert result.pooled_ci_lower < result.pooled_effect < result.pooled_ci_upper

        # Should be significant (all studies trend same direction with moderate effects)
        assert result.pooled_p_value < 0.05

    def test_heterogeneity_metrics(self, sample_studies):
        result = run_meta_analysis(sample_studies, EffectMeasure.ODDS_RATIO)
        assert result is not None

        # I-squared should be between 0 and 100
        assert 0.0 <= result.i_squared <= 100.0

        # Q statistic should be positive
        assert result.q_statistic >= 0.0

        # Tau-squared should be non-negative
        assert result.tau_squared >= 0.0

    def test_eggers_test_runs(self, sample_studies):
        result = run_meta_analysis(sample_studies, EffectMeasure.ODDS_RATIO)
        assert result is not None
        # With 5 studies, Egger's test should run
        assert result.egger_intercept is not None
        assert result.egger_p_value is not None

    def test_insufficient_studies(self):
        studies = [make_study(pmid="1", effect_size=1.5, ci_lower=1.0, ci_upper=2.0)]
        result = run_meta_analysis(studies, EffectMeasure.ODDS_RATIO)
        assert result is None

    def test_no_extractable_data(self):
        studies = [make_study(pmid="1"), make_study(pmid="2")]
        result = run_meta_analysis(studies, EffectMeasure.ODDS_RATIO)
        assert result is None

    def test_two_studies_minimal(self):
        studies = [
            make_study(pmid="1", effect_size=1.5, ci_lower=1.0, ci_upper=2.0),
            make_study(pmid="2", effect_size=2.0, ci_lower=1.2, ci_upper=3.0),
        ]
        result = run_meta_analysis(studies, EffectMeasure.ODDS_RATIO)
        assert result is not None
        assert result.n_studies == 2
        # With only 2 studies, Egger's test should still attempt
        # but may not be reliable

    def test_identical_studies_zero_heterogeneity(self):
        studies = [
            make_study(pmid="1", effect_size=1.5, ci_lower=1.0, ci_upper=2.0),
            make_study(pmid="2", effect_size=1.5, ci_lower=1.0, ci_upper=2.0),
            make_study(pmid="3", effect_size=1.5, ci_lower=1.0, ci_upper=2.0),
        ]
        result = run_meta_analysis(studies, EffectMeasure.ODDS_RATIO)
        assert result is not None
        assert result.i_squared == 0.0
        assert result.tau_squared == 0.0
        assert abs(result.pooled_effect - 1.5) < 0.001

    def test_studies_included_list(self, sample_studies):
        result = run_meta_analysis(sample_studies, EffectMeasure.ODDS_RATIO)
        assert result is not None
        assert set(result.studies_included) == {"10001", "10002", "10003", "10004", "10005"}


class TestComputeStudyWeights:
    def test_weights_sum_to_100(self, sample_studies):
        weights = compute_study_weights(sample_studies, tau_squared=0.0)
        total = sum(w for _, w in weights)
        assert abs(total - 100.0) < 0.01

    def test_weights_with_tau(self, sample_studies):
        weights = compute_study_weights(sample_studies, tau_squared=0.1)
        total = sum(w for _, w in weights)
        assert abs(total - 100.0) < 0.01

    def test_larger_studies_get_more_weight(self):
        """Studies with narrower CIs (more precise) should get more weight."""
        narrow = make_study(pmid="1", effect_size=1.5, ci_lower=1.3, ci_upper=1.7)
        wide = make_study(pmid="2", effect_size=1.5, ci_lower=0.5, ci_upper=2.5)

        weights = compute_study_weights([narrow, wide], tau_squared=0.0)
        w_narrow = next(w for s, w in weights if s.pmid == "1")
        w_wide = next(w for s, w in weights if s.pmid == "2")

        assert w_narrow > w_wide

    def test_empty_studies(self):
        weights = compute_study_weights([], tau_squared=0.0)
        assert weights == []
