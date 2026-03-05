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


class TestHighHeterogeneity:
    """Tests with very heterogeneous studies (I-squared > 75%)."""

    def test_high_heterogeneity_detection(self):
        """Wildly different effect sizes should produce high I-squared."""
        studies = [
            make_study(pmid="1", effect_size=0.5, ci_lower=0.3, ci_upper=0.7),
            make_study(pmid="2", effect_size=3.0, ci_lower=2.5, ci_upper=3.5),
            make_study(pmid="3", effect_size=1.0, ci_lower=0.8, ci_upper=1.2),
            make_study(pmid="4", effect_size=5.0, ci_lower=4.0, ci_upper=6.0),
            make_study(pmid="5", effect_size=0.2, ci_lower=0.1, ci_upper=0.3),
        ]
        result = run_meta_analysis(studies, EffectMeasure.ODDS_RATIO)

        assert result is not None
        assert result.i_squared > 75.0, (
            f"I² = {result.i_squared:.1f}% should be > 75% for highly heterogeneous studies"
        )
        assert result.high_heterogeneity is True
        assert result.tau_squared > 0.0

    def test_high_heterogeneity_q_significant(self):
        """Q statistic should be significant with high heterogeneity."""
        studies = [
            make_study(pmid="1", effect_size=0.3, ci_lower=0.2, ci_upper=0.4),
            make_study(pmid="2", effect_size=4.0, ci_lower=3.5, ci_upper=4.5),
            make_study(pmid="3", effect_size=1.5, ci_lower=1.3, ci_upper=1.7),
            make_study(pmid="4", effect_size=0.1, ci_lower=0.05, ci_upper=0.15),
        ]
        result = run_meta_analysis(studies, EffectMeasure.ODDS_RATIO)

        assert result is not None
        assert result.q_p_value < 0.05, (
            f"Q p-value {result.q_p_value:.4f} should be < 0.05"
        )

    def test_tau_squared_increases_with_spread(self):
        """Tau-squared should be larger when effects are more spread out."""
        # Narrow spread
        narrow = [
            make_study(pmid="1", effect_size=1.5, ci_lower=1.3, ci_upper=1.7),
            make_study(pmid="2", effect_size=1.6, ci_lower=1.4, ci_upper=1.8),
            make_study(pmid="3", effect_size=1.4, ci_lower=1.2, ci_upper=1.6),
        ]
        # Wide spread
        wide = [
            make_study(pmid="4", effect_size=0.5, ci_lower=0.3, ci_upper=0.7),
            make_study(pmid="5", effect_size=3.0, ci_lower=2.8, ci_upper=3.2),
            make_study(pmid="6", effect_size=1.5, ci_lower=1.3, ci_upper=1.7),
        ]

        result_narrow = run_meta_analysis(narrow, EffectMeasure.ODDS_RATIO)
        result_wide = run_meta_analysis(wide, EffectMeasure.ODDS_RATIO)

        assert result_narrow is not None
        assert result_wide is not None
        assert result_wide.tau_squared > result_narrow.tau_squared


class TestNegativeEffectSizes:
    """Tests with negative effect sizes (mean differences)."""

    def test_negative_mean_differences(self):
        """Meta-analysis should handle negative effect sizes (e.g., symptom reduction)."""
        studies = [
            make_study(pmid="1", effect_size=-2.5, ci_lower=-3.5, ci_upper=-1.5),
            make_study(pmid="2", effect_size=-3.0, ci_lower=-4.0, ci_upper=-2.0),
            make_study(pmid="3", effect_size=-1.8, ci_lower=-2.8, ci_upper=-0.8),
            make_study(pmid="4", effect_size=-2.2, ci_lower=-3.0, ci_upper=-1.4),
        ]
        result = run_meta_analysis(studies, EffectMeasure.MEAN_DIFFERENCE)

        assert result is not None
        assert result.pooled_effect < 0, (
            f"Pooled effect {result.pooled_effect:.3f} should be negative"
        )
        assert result.pooled_ci_upper < 0, (
            "Upper CI should also be negative (all studies show benefit)"
        )

    def test_mixed_positive_negative(self):
        """Studies with both positive and negative effects should pool correctly."""
        studies = [
            make_study(pmid="1", effect_size=-1.5, ci_lower=-2.5, ci_upper=-0.5),
            make_study(pmid="2", effect_size=0.5, ci_lower=-0.5, ci_upper=1.5),
            make_study(pmid="3", effect_size=-0.8, ci_lower=-1.8, ci_upper=0.2),
            make_study(pmid="4", effect_size=1.2, ci_lower=0.2, ci_upper=2.2),
        ]
        result = run_meta_analysis(studies, EffectMeasure.MEAN_DIFFERENCE)

        assert result is not None
        # With mixed results, pooled effect should be somewhere in the middle
        assert -2.0 < result.pooled_effect < 2.0
        # CI should span zero (non-significant)
        # (Not strictly guaranteed, but likely with this balanced mix)

    def test_zero_centered_effects(self):
        """Effects centered around zero should yield pooled effect near zero."""
        studies = [
            make_study(pmid="1", effect_size=0.1, ci_lower=-0.5, ci_upper=0.7),
            make_study(pmid="2", effect_size=-0.1, ci_lower=-0.7, ci_upper=0.5),
            make_study(pmid="3", effect_size=0.0, ci_lower=-0.6, ci_upper=0.6),
        ]
        result = run_meta_analysis(studies, EffectMeasure.MEAN_DIFFERENCE)

        assert result is not None
        assert abs(result.pooled_effect) < 0.3, (
            f"Pooled effect {result.pooled_effect:.3f} should be near zero"
        )


class TestSmallSampleSizes:
    """Tests with very small sample sizes (wide confidence intervals)."""

    def test_very_wide_cis(self):
        """Studies with extremely wide CIs should still pool without errors."""
        studies = [
            make_study(pmid="1", effect_size=2.0, ci_lower=0.1, ci_upper=40.0),
            make_study(pmid="2", effect_size=1.5, ci_lower=0.2, ci_upper=11.0),
            make_study(pmid="3", effect_size=3.0, ci_lower=0.3, ci_upper=30.0),
        ]
        result = run_meta_analysis(studies, EffectMeasure.ODDS_RATIO)

        assert result is not None
        assert result.pooled_ci_lower < result.pooled_effect < result.pooled_ci_upper

    def test_asymmetric_precision(self):
        """Mix of very precise and very imprecise studies should be handled."""
        studies = [
            # Very precise (large trial)
            make_study(pmid="1", effect_size=1.50, ci_lower=1.45, ci_upper=1.55),
            # Very imprecise (tiny trial)
            make_study(pmid="2", effect_size=2.00, ci_lower=0.10, ci_upper=40.0),
            make_study(pmid="3", effect_size=1.60, ci_lower=1.40, ci_upper=1.80),
        ]
        result = run_meta_analysis(studies, EffectMeasure.ODDS_RATIO)

        assert result is not None
        # Pooled effect should be pulled toward the precise study
        assert abs(result.pooled_effect - 1.50) < 0.20, (
            f"Pooled {result.pooled_effect:.3f} should be close to precise study (1.50)"
        )


class TestNumericalStability:
    """Tests for numerical stability with extreme values."""

    def test_very_large_effect_sizes(self):
        """Engine should handle very large effect sizes without overflow."""
        studies = [
            make_study(pmid="1", effect_size=100.0, ci_lower=80.0, ci_upper=120.0),
            make_study(pmid="2", effect_size=150.0, ci_lower=130.0, ci_upper=170.0),
            make_study(pmid="3", effect_size=120.0, ci_lower=100.0, ci_upper=140.0),
        ]
        result = run_meta_analysis(studies, EffectMeasure.MEAN_DIFFERENCE)

        assert result is not None
        assert 80.0 <= result.pooled_effect <= 170.0
        assert result.pooled_ci_lower < result.pooled_effect < result.pooled_ci_upper

    def test_very_small_effect_sizes(self):
        """Engine should handle effect sizes near zero."""
        studies = [
            make_study(pmid="1", effect_size=0.001, ci_lower=-0.01, ci_upper=0.012),
            make_study(pmid="2", effect_size=0.002, ci_lower=-0.008, ci_upper=0.012),
            make_study(pmid="3", effect_size=-0.001, ci_lower=-0.011, ci_upper=0.009),
        ]
        result = run_meta_analysis(studies, EffectMeasure.MEAN_DIFFERENCE)

        assert result is not None
        assert abs(result.pooled_effect) < 0.01
        assert result.pooled_ci_lower < result.pooled_effect < result.pooled_ci_upper

    def test_very_narrow_cis(self):
        """Extremely narrow CIs (high precision) should not cause division issues."""
        studies = [
            make_study(pmid="1", effect_size=1.50, ci_lower=1.499, ci_upper=1.501),
            make_study(pmid="2", effect_size=1.51, ci_lower=1.509, ci_upper=1.511),
            make_study(pmid="3", effect_size=1.50, ci_lower=1.499, ci_upper=1.501),
        ]
        result = run_meta_analysis(studies, EffectMeasure.ODDS_RATIO)

        assert result is not None
        assert abs(result.pooled_effect - 1.50) < 0.02

    def test_identical_effect_different_precision(self):
        """Same effect size but varying precision should yield that effect."""
        studies = [
            make_study(pmid="1", effect_size=2.0, ci_lower=1.9, ci_upper=2.1),
            make_study(pmid="2", effect_size=2.0, ci_lower=1.0, ci_upper=3.0),
            make_study(pmid="3", effect_size=2.0, ci_lower=1.5, ci_upper=2.5),
            make_study(pmid="4", effect_size=2.0, ci_lower=0.5, ci_upper=3.5),
        ]
        result = run_meta_analysis(studies, EffectMeasure.ODDS_RATIO)

        assert result is not None
        assert abs(result.pooled_effect - 2.0) < 0.001, (
            f"Pooled effect {result.pooled_effect:.4f} should be exactly 2.0"
        )
        assert result.i_squared == 0.0
        assert result.tau_squared == 0.0

    def test_many_studies_stability(self):
        """Meta-analysis should remain stable with a large number of studies."""
        import random

        random.seed(42)
        studies = [
            make_study(
                pmid=str(i),
                effect_size=1.5 + random.gauss(0, 0.3),
                ci_lower=1.5 + random.gauss(0, 0.3) - 0.5,
                ci_upper=1.5 + random.gauss(0, 0.3) + 0.5,
            )
            for i in range(50)
        ]
        # Fix CI to be centered on effect_size
        for s in studies:
            s.ci_lower = s.effect_size - 0.5
            s.ci_upper = s.effect_size + 0.5

        result = run_meta_analysis(studies, EffectMeasure.ODDS_RATIO)

        assert result is not None
        assert result.n_studies == 50
        assert 1.0 < result.pooled_effect < 2.0
        assert result.pooled_ci_lower < result.pooled_effect < result.pooled_ci_upper
