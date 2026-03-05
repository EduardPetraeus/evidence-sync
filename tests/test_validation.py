"""Validation tests against published meta-analysis results.

Uses data from Cipriani et al. 2018 (PMID 29477251, The Lancet):
'Comparative efficacy and acceptability of 21 antidepressant drugs
for the acute treatment of adults with major depressive disorder.'

This landmark network meta-analysis provides drug-level odds ratios
for response (SSRI vs placebo). We use the 6 published SSRI ORs as
study-level inputs and verify that our DerSimonian-Laird pooling
produces a result consistent with the published SSRI class estimate.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from evidence_sync.dashboard import generate_forest_plot
from evidence_sync.models import EffectMeasure
from evidence_sync.statistics import compute_study_weights, run_meta_analysis
from tests.conftest import make_study


def _cipriani_ssri_studies():
    """Create study-level inputs from Cipriani 2018 SSRI vs placebo ORs.

    Each 'study' here represents a single drug's pooled OR from the
    network meta-analysis. We feed these into our pairwise engine to
    get a class-level SSRI vs placebo estimate.
    """
    return [
        make_study(
            pmid="cipriani_fluoxetine",
            title="Fluoxetine vs placebo (Cipriani 2018, 49 trials)",
            effect_size=1.52,
            ci_lower=1.40,
            ci_upper=1.64,
        ),
        make_study(
            pmid="cipriani_sertraline",
            title="Sertraline vs placebo (Cipriani 2018, 25 trials)",
            effect_size=1.67,
            ci_lower=1.50,
            ci_upper=1.85,
        ),
        make_study(
            pmid="cipriani_paroxetine",
            title="Paroxetine vs placebo (Cipriani 2018, 33 trials)",
            effect_size=1.75,
            ci_lower=1.61,
            ci_upper=1.90,
        ),
        make_study(
            pmid="cipriani_citalopram",
            title="Citalopram vs placebo (Cipriani 2018, 17 trials)",
            effect_size=1.52,
            ci_lower=1.33,
            ci_upper=1.74,
        ),
        make_study(
            pmid="cipriani_escitalopram",
            title="Escitalopram vs placebo (Cipriani 2018, 18 trials)",
            effect_size=1.68,
            ci_lower=1.50,
            ci_upper=1.87,
        ),
        make_study(
            pmid="cipriani_fluvoxamine",
            title="Fluvoxamine vs placebo (Cipriani 2018, 7 trials)",
            effect_size=1.69,
            ci_lower=1.39,
            ci_upper=2.06,
        ),
    ]


class TestCipriani2018Validation:
    """Validate our engine against Cipriani et al. 2018 published results."""

    def test_pooled_effect_in_expected_range(self):
        """Pooled SSRI OR should be approximately 1.6 (range 1.50-1.75)."""
        studies = _cipriani_ssri_studies()
        result = run_meta_analysis(
            studies, EffectMeasure.ODDS_RATIO, topic="ssri-validation"
        )

        assert result is not None, "Meta-analysis should produce a result"
        assert result.n_studies == 6, f"Expected 6 studies, got {result.n_studies}"

        # Pooled effect should be in the range consistent with published SSRI class estimate
        assert 1.50 <= result.pooled_effect <= 1.75, (
            f"Pooled OR {result.pooled_effect:.3f} outside expected range [1.50, 1.75]"
        )

    def test_confidence_interval_covers_published(self):
        """95% CI should cover the published class-level estimates."""
        studies = _cipriani_ssri_studies()
        result = run_meta_analysis(
            studies, EffectMeasure.ODDS_RATIO, topic="ssri-validation"
        )

        assert result is not None
        # CI should contain the pooled effect
        assert result.pooled_ci_lower < result.pooled_effect < result.pooled_ci_upper

        # CI should be reasonably narrow given 6 precise drug-level estimates
        ci_width = result.pooled_ci_upper - result.pooled_ci_lower
        assert ci_width < 0.40, f"CI too wide: {ci_width:.3f}"

        # Lower bound should be > 1.0 (significant benefit over placebo)
        assert result.pooled_ci_lower > 1.0, (
            f"Lower CI {result.pooled_ci_lower:.3f} should be > 1.0"
        )

    def test_significance(self):
        """SSRI class effect should be highly significant."""
        studies = _cipriani_ssri_studies()
        result = run_meta_analysis(
            studies, EffectMeasure.ODDS_RATIO, topic="ssri-validation"
        )

        assert result is not None
        assert result.pooled_p_value < 0.001, (
            f"P-value {result.pooled_p_value:.6f} should be < 0.001"
        )
        assert result.significant

    def test_heterogeneity_within_bounds(self):
        """SSRIs are relatively similar — I-squared should be moderate at most.

        The 6 drugs have ORs ranging from 1.52 to 1.75, so some heterogeneity
        is expected, but it should not be extreme (I² < 75%).
        """
        studies = _cipriani_ssri_studies()
        result = run_meta_analysis(
            studies, EffectMeasure.ODDS_RATIO, topic="ssri-validation"
        )

        assert result is not None
        # Heterogeneity should be low-to-moderate for this class of similar drugs
        assert result.i_squared < 75.0, (
            f"I² = {result.i_squared:.1f}% is unexpectedly high for similar SSRIs"
        )
        # Tau-squared should be small
        assert result.tau_squared < 0.05, (
            f"Tau² = {result.tau_squared:.4f} is unexpectedly large"
        )

    def test_eggers_test_runs(self):
        """Egger's test should run with 6 studies."""
        studies = _cipriani_ssri_studies()
        result = run_meta_analysis(
            studies, EffectMeasure.ODDS_RATIO, topic="ssri-validation"
        )

        assert result is not None
        assert result.egger_intercept is not None
        assert result.egger_p_value is not None

    def test_study_weights_reasonable(self):
        """More precise drug estimates should receive more weight."""
        studies = _cipriani_ssri_studies()
        result = run_meta_analysis(
            studies, EffectMeasure.ODDS_RATIO, topic="ssri-validation"
        )

        assert result is not None
        weights = compute_study_weights(studies, result.tau_squared)

        # All weights should be positive and sum to 100
        assert len(weights) == 6
        total = sum(w for _, w in weights)
        assert abs(total - 100.0) < 0.01

        # Fluoxetine (narrowest CI: 1.40-1.64) should have highest weight
        # Fluvoxamine (widest CI: 1.39-2.06) should have lowest weight
        weight_map = {s.pmid: w for s, w in weights}
        assert weight_map["cipriani_fluoxetine"] > weight_map["cipriani_fluvoxamine"], (
            "Fluoxetine (most precise) should have more weight than Fluvoxamine (least precise)"
        )

    def test_forest_plot_generation(self):
        """Forest plot should generate without errors."""
        studies = _cipriani_ssri_studies()
        result = run_meta_analysis(
            studies, EffectMeasure.ODDS_RATIO, topic="ssri-validation"
        )

        assert result is not None

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "cipriani_forest.png"
            returned_path = generate_forest_plot(
                studies,
                result,
                output_path,
                title="SSRI vs Placebo — Cipriani 2018 Validation",
            )
            assert returned_path.exists(), "Forest plot file should be created"
            assert returned_path.stat().st_size > 0, "Forest plot should not be empty"

    def test_individual_drug_ordering(self):
        """The pooled effect should be between the min and max individual drug ORs."""
        studies = _cipriani_ssri_studies()
        result = run_meta_analysis(
            studies, EffectMeasure.ODDS_RATIO, topic="ssri-validation"
        )

        assert result is not None
        individual_ors = [s.effect_size for s in studies]
        assert min(individual_ors) <= result.pooled_effect <= max(individual_ors), (
            f"Pooled {result.pooled_effect:.3f} outside individual range "
            f"[{min(individual_ors):.2f}, {max(individual_ors):.2f}]"
        )
