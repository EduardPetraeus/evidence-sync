"""Shared test fixtures for Evidence Sync."""

from __future__ import annotations

from datetime import date

import pytest

from evidence_sync.models import (
    BiasRisk,
    EffectMeasure,
    ReviewConfig,
    ReviewStatus,
    RiskOfBias,
    Study,
    StudyDesign,
)


@pytest.fixture
def sample_review_config() -> ReviewConfig:
    """A sample review config for SSRI depression topic."""
    return ReviewConfig(
        topic_id="ssri-depression",
        topic_name="SSRIs vs. Placebo for MDD",
        search_query='(SSRI OR fluoxetine) AND depression AND placebo',
        effect_measure=EffectMeasure.ODDS_RATIO,
        primary_outcome="Response rate",
        publication_types=["Randomized Controlled Trial"],
        effect_change_threshold_pct=10.0,
        heterogeneity_change_threshold=15.0,
    )


def make_study(
    pmid: str = "12345678",
    effect_size: float | None = None,
    ci_lower: float | None = None,
    ci_upper: float | None = None,
    **kwargs,
) -> Study:
    """Helper to create a study with sensible defaults."""
    defaults = dict(
        pmid=pmid,
        title=f"Test Study {pmid}",
        authors=["Smith J", "Doe A"],
        journal="Test Journal",
        publication_date=date(2020, 1, 1),
        abstract="This is a test abstract.",
        effect_size=effect_size,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        effect_measure=EffectMeasure.ODDS_RATIO if effect_size else None,
        study_design=StudyDesign.RCT,
        # Default to APPROVED when effect data is provided so existing tests
        # pass without changes (meta-analysis requires approval by default).
        review_status=ReviewStatus.APPROVED if effect_size is not None else ReviewStatus.PENDING,
    )
    defaults.update(kwargs)
    return Study(**defaults)


@pytest.fixture
def sample_studies() -> list[Study]:
    """A set of sample studies with realistic SSRI trial data.

    Based on approximate values from well-known SSRI trials.
    These simulate odds ratios for response (SSRI vs placebo).
    OR < 1 means SSRI is better (more likely to respond).
    """
    return [
        make_study(
            pmid="10001",
            title="Fluoxetine vs placebo in MDD (Study A)",
            effect_size=1.52,
            ci_lower=1.10,
            ci_upper=2.10,
            sample_size_treatment=100,
            sample_size_control=100,
            risk_of_bias=RiskOfBias(
                random_sequence_generation=BiasRisk.LOW,
                allocation_concealment=BiasRisk.LOW,
                blinding_participants=BiasRisk.LOW,
                blinding_outcome=BiasRisk.LOW,
                incomplete_outcome=BiasRisk.UNCLEAR,
                selective_reporting=BiasRisk.LOW,
            ),
        ),
        make_study(
            pmid="10002",
            title="Sertraline vs placebo in MDD (Study B)",
            effect_size=1.68,
            ci_lower=1.22,
            ci_upper=2.31,
            sample_size_treatment=150,
            sample_size_control=150,
            risk_of_bias=RiskOfBias(
                random_sequence_generation=BiasRisk.LOW,
                allocation_concealment=BiasRisk.LOW,
                blinding_participants=BiasRisk.LOW,
                blinding_outcome=BiasRisk.LOW,
                incomplete_outcome=BiasRisk.LOW,
                selective_reporting=BiasRisk.LOW,
            ),
        ),
        make_study(
            pmid="10003",
            title="Paroxetine vs placebo in MDD (Study C)",
            effect_size=1.45,
            ci_lower=0.98,
            ci_upper=2.14,
            sample_size_treatment=80,
            sample_size_control=80,
            risk_of_bias=RiskOfBias(
                random_sequence_generation=BiasRisk.LOW,
                allocation_concealment=BiasRisk.UNCLEAR,
                blinding_participants=BiasRisk.LOW,
                blinding_outcome=BiasRisk.LOW,
                incomplete_outcome=BiasRisk.HIGH,
                selective_reporting=BiasRisk.UNCLEAR,
            ),
        ),
        make_study(
            pmid="10004",
            title="Citalopram vs placebo in MDD (Study D)",
            effect_size=1.80,
            ci_lower=1.30,
            ci_upper=2.49,
            sample_size_treatment=120,
            sample_size_control=120,
            risk_of_bias=RiskOfBias(
                random_sequence_generation=BiasRisk.LOW,
                allocation_concealment=BiasRisk.LOW,
                blinding_participants=BiasRisk.LOW,
                blinding_outcome=BiasRisk.LOW,
                incomplete_outcome=BiasRisk.LOW,
                selective_reporting=BiasRisk.LOW,
            ),
        ),
        make_study(
            pmid="10005",
            title="Escitalopram vs placebo in MDD (Study E)",
            effect_size=1.95,
            ci_lower=1.42,
            ci_upper=2.68,
            sample_size_treatment=200,
            sample_size_control=200,
            risk_of_bias=RiskOfBias(
                random_sequence_generation=BiasRisk.LOW,
                allocation_concealment=BiasRisk.LOW,
                blinding_participants=BiasRisk.LOW,
                blinding_outcome=BiasRisk.LOW,
                incomplete_outcome=BiasRisk.LOW,
                selective_reporting=BiasRisk.LOW,
            ),
        ),
    ]
