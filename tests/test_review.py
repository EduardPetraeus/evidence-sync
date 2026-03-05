"""Tests for the human-in-the-loop review workflow."""

from __future__ import annotations

from datetime import date

import pytest

from evidence_sync.models import EffectMeasure, ReviewStatus
from evidence_sync.review import (
    approve_study,
    correct_study,
    get_pending_studies,
    get_review_summary,
    reject_study,
)
from evidence_sync.statistics import run_meta_analysis
from evidence_sync.versioning import load_study, save_study
from tests.conftest import make_study


class TestApproveStudy:
    def test_approve_study(self):
        study = make_study(pmid="1", review_status=ReviewStatus.PENDING)
        result = approve_study(study, reviewer="dr-smith", notes="Looks good")

        assert result.review_status == ReviewStatus.APPROVED
        assert result.reviewer == "dr-smith"
        assert result.review_date == date.today()
        assert result.review_notes == "Looks good"

    def test_approve_without_notes(self):
        study = make_study(pmid="1", review_status=ReviewStatus.PENDING)
        result = approve_study(study, reviewer="dr-smith")

        assert result.review_status == ReviewStatus.APPROVED
        assert result.review_notes is None


class TestRejectStudy:
    def test_reject_study(self):
        study = make_study(pmid="1", review_status=ReviewStatus.PENDING)
        result = reject_study(study, reviewer="dr-jones", notes="Data looks wrong")

        assert result.review_status == ReviewStatus.REJECTED
        assert result.reviewer == "dr-jones"
        assert result.review_date == date.today()
        assert result.review_notes == "Data looks wrong"


class TestCorrectStudy:
    def test_correct_study(self):
        study = make_study(
            pmid="1",
            effect_size=1.5,
            ci_lower=1.0,
            ci_upper=2.0,
            review_status=ReviewStatus.PENDING,
        )
        result = correct_study(
            study,
            reviewer="dr-smith",
            corrections={"effect_size": 1.6, "ci_lower": 1.1},
            notes="Corrected from table 3",
        )

        assert result.review_status == ReviewStatus.CORRECTED
        assert result.effect_size == 1.6
        assert result.ci_lower == 1.1
        assert result.ci_upper == 2.0  # unchanged
        assert result.reviewer == "dr-smith"
        assert result.review_notes == "Corrected from table 3"

    def test_correct_study_preserves_originals(self):
        study = make_study(
            pmid="1",
            effect_size=1.5,
            ci_lower=1.0,
            ci_upper=2.0,
            review_status=ReviewStatus.PENDING,
        )
        correct_study(
            study,
            reviewer="dr-smith",
            corrections={"effect_size": 1.6, "ci_lower": 1.1, "ci_upper": 2.2},
        )

        assert study.original_effect_size == 1.5
        assert study.original_ci_lower == 1.0
        assert study.original_ci_upper == 2.0

    def test_correct_study_does_not_overwrite_originals(self):
        """Second correction should not overwrite the original values."""
        study = make_study(
            pmid="1",
            effect_size=1.5,
            ci_lower=1.0,
            ci_upper=2.0,
            review_status=ReviewStatus.PENDING,
        )
        correct_study(study, reviewer="dr-a", corrections={"effect_size": 1.6})
        correct_study(study, reviewer="dr-b", corrections={"effect_size": 1.7})

        # Should still have the very first original value
        assert study.original_effect_size == 1.5
        assert study.effect_size == 1.7

    def test_correct_study_invalid_field(self):
        study = make_study(pmid="1", review_status=ReviewStatus.PENDING)
        with pytest.raises(ValueError, match="Cannot correct fields"):
            correct_study(study, reviewer="dr-smith", corrections={"pmid": "999"})


class TestGetPendingStudies:
    def test_get_pending_studies(self):
        studies = [
            make_study(pmid="1", review_status=ReviewStatus.PENDING),
            make_study(pmid="2", review_status=ReviewStatus.APPROVED),
            make_study(pmid="3", review_status=ReviewStatus.PENDING),
            make_study(pmid="4", review_status=ReviewStatus.REJECTED),
        ]
        pending = get_pending_studies(studies)

        assert len(pending) == 2
        assert all(s.review_status == ReviewStatus.PENDING for s in pending)

    def test_get_pending_none(self):
        studies = [
            make_study(pmid="1", review_status=ReviewStatus.APPROVED),
            make_study(pmid="2", review_status=ReviewStatus.REJECTED),
        ]
        assert get_pending_studies(studies) == []


class TestGetReviewSummary:
    def test_get_review_summary(self):
        studies = [
            make_study(pmid="1", review_status=ReviewStatus.PENDING),
            make_study(pmid="2", review_status=ReviewStatus.APPROVED),
            make_study(pmid="3", review_status=ReviewStatus.APPROVED),
            make_study(pmid="4", review_status=ReviewStatus.REJECTED),
            make_study(pmid="5", review_status=ReviewStatus.CORRECTED),
        ]
        summary = get_review_summary(studies)

        assert summary == {
            "pending": 1,
            "approved": 2,
            "rejected": 1,
            "corrected": 1,
            "total": 5,
        }

    def test_empty_summary(self):
        summary = get_review_summary([])
        assert summary == {
            "pending": 0,
            "approved": 0,
            "rejected": 0,
            "corrected": 0,
            "total": 0,
        }


class TestReviewRoundtripYaml:
    def test_review_roundtrip_yaml(self, tmp_path):
        """Save/load study with review fields preserves all data."""
        study = make_study(
            pmid="12345",
            effect_size=1.5,
            ci_lower=1.0,
            ci_upper=2.0,
            review_status=ReviewStatus.CORRECTED,
        )
        study.reviewer = "dr-smith"
        study.review_date = date(2026, 3, 5)
        study.review_notes = "Fixed from table 3"
        study.original_effect_size = 1.3
        study.original_ci_lower = 0.9
        study.original_ci_upper = 1.8

        save_study(study, tmp_path)
        loaded = load_study(tmp_path / "12345.yaml")

        assert loaded.review_status == ReviewStatus.CORRECTED
        assert loaded.reviewer == "dr-smith"
        assert loaded.review_date == date(2026, 3, 5)
        assert loaded.review_notes == "Fixed from table 3"
        assert loaded.original_effect_size == 1.3
        assert loaded.original_ci_lower == 0.9
        assert loaded.original_ci_upper == 1.8


class TestMetaAnalysisRequiresApproval:
    def test_meta_analysis_requires_approval(self):
        """Only approved/corrected studies should be included when require_approval=True."""
        studies = [
            make_study(pmid="1", effect_size=1.5, ci_lower=1.0, ci_upper=2.0,
                       review_status=ReviewStatus.APPROVED),
            make_study(pmid="2", effect_size=2.0, ci_lower=1.2, ci_upper=3.0,
                       review_status=ReviewStatus.APPROVED),
            make_study(pmid="3", effect_size=1.8, ci_lower=1.1, ci_upper=2.5,
                       review_status=ReviewStatus.PENDING),
            make_study(pmid="4", effect_size=1.6, ci_lower=1.0, ci_upper=2.2,
                       review_status=ReviewStatus.REJECTED),
        ]
        result = run_meta_analysis(
            studies, EffectMeasure.ODDS_RATIO, topic="test", require_approval=True
        )

        assert result is not None
        assert result.n_studies == 2
        assert set(result.studies_included) == {"1", "2"}

    def test_meta_analysis_includes_corrected(self):
        """Corrected studies should also be included."""
        studies = [
            make_study(pmid="1", effect_size=1.5, ci_lower=1.0, ci_upper=2.0,
                       review_status=ReviewStatus.APPROVED),
            make_study(pmid="2", effect_size=2.0, ci_lower=1.2, ci_upper=3.0,
                       review_status=ReviewStatus.CORRECTED),
        ]
        result = run_meta_analysis(
            studies, EffectMeasure.ODDS_RATIO, topic="test", require_approval=True
        )

        assert result is not None
        assert result.n_studies == 2

    def test_meta_analysis_backward_compat(self):
        """require_approval=False uses old behavior (all studies with data)."""
        studies = [
            make_study(pmid="1", effect_size=1.5, ci_lower=1.0, ci_upper=2.0,
                       review_status=ReviewStatus.PENDING),
            make_study(pmid="2", effect_size=2.0, ci_lower=1.2, ci_upper=3.0,
                       review_status=ReviewStatus.PENDING),
            make_study(pmid="3", effect_size=1.8, ci_lower=1.1, ci_upper=2.5,
                       review_status=ReviewStatus.REJECTED),
        ]
        result = run_meta_analysis(
            studies, EffectMeasure.ODDS_RATIO, topic="test", require_approval=False
        )

        assert result is not None
        assert result.n_studies == 3

    def test_meta_analysis_insufficient_approved(self):
        """Should return None if fewer than 2 approved studies."""
        studies = [
            make_study(pmid="1", effect_size=1.5, ci_lower=1.0, ci_upper=2.0,
                       review_status=ReviewStatus.APPROVED),
            make_study(pmid="2", effect_size=2.0, ci_lower=1.2, ci_upper=3.0,
                       review_status=ReviewStatus.PENDING),
        ]
        result = run_meta_analysis(
            studies, EffectMeasure.ODDS_RATIO, topic="test", require_approval=True
        )

        assert result is None
