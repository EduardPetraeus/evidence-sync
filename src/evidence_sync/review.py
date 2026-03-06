"""Human-in-the-loop review workflow for extracted study data."""

from __future__ import annotations

from datetime import date
from typing import Optional

from evidence_sync.models import ReviewStatus, Study

# Fields that can be corrected during review
CORRECTABLE_FIELDS = frozenset({
    "effect_size",
    "ci_lower",
    "ci_upper",
    "p_value",
    "sample_size_treatment",
    "sample_size_control",
    "effect_measure",
    "study_design",
    "primary_outcome",
})


def approve_study(
    study: Study,
    reviewer: str,
    notes: Optional[str] = None,
) -> Study:
    """Approve a study's extracted data for inclusion in meta-analysis.

    Args:
        study: The study to approve.
        reviewer: Name of the reviewer.
        notes: Optional review notes.

    Returns:
        The updated study with APPROVED status.
    """
    study.review_status = ReviewStatus.APPROVED
    study.reviewer = reviewer
    study.review_date = date.today()
    study.review_notes = notes
    return study


def reject_study(
    study: Study,
    reviewer: str,
    notes: Optional[str] = None,
) -> Study:
    """Reject a study's extracted data.

    Args:
        study: The study to reject.
        reviewer: Name of the reviewer.
        notes: Optional review notes explaining the rejection.

    Returns:
        The updated study with REJECTED status.
    """
    study.review_status = ReviewStatus.REJECTED
    study.reviewer = reviewer
    study.review_date = date.today()
    study.review_notes = notes
    return study


def correct_study(
    study: Study,
    reviewer: str,
    corrections: dict,
    notes: Optional[str] = None,
) -> Study:
    """Correct a study's extracted data and mark as CORRECTED.

    Saves original values to original_* fields before applying corrections,
    providing an audit trail.

    Args:
        study: The study to correct.
        reviewer: Name of the reviewer.
        corrections: Dict of field_name -> new_value. Only fields in
            CORRECTABLE_FIELDS are accepted.
        notes: Optional review notes explaining the corrections.

    Returns:
        The updated study with CORRECTED status.

    Raises:
        ValueError: If a correction key is not in CORRECTABLE_FIELDS.
    """
    # Validate correction keys
    invalid_keys = set(corrections.keys()) - CORRECTABLE_FIELDS
    if invalid_keys:
        raise ValueError(
            f"Cannot correct fields: {invalid_keys}. "
            f"Allowed: {sorted(CORRECTABLE_FIELDS)}"
        )

    # Save original values before correction (only for tracked fields)
    if "effect_size" in corrections and study.original_effect_size is None:
        study.original_effect_size = study.effect_size
    if "ci_lower" in corrections and study.original_ci_lower is None:
        study.original_ci_lower = study.ci_lower
    if "ci_upper" in corrections and study.original_ci_upper is None:
        study.original_ci_upper = study.ci_upper

    # Apply corrections
    for field_name, new_value in corrections.items():
        setattr(study, field_name, new_value)

    study.review_status = ReviewStatus.CORRECTED
    study.reviewer = reviewer
    study.review_date = date.today()
    study.review_notes = notes
    return study


def get_pending_studies(studies: list[Study]) -> list[Study]:
    """Filter to studies with PENDING review status.

    Args:
        studies: List of studies to filter.

    Returns:
        List of studies with review_status == PENDING.
    """
    return [s for s in studies if s.review_status == ReviewStatus.PENDING]


def get_review_summary(studies: list[Study]) -> dict:
    """Get a summary of review statuses across studies.

    Args:
        studies: List of studies to summarize.

    Returns:
        Dict with counts: pending, approved, rejected, corrected, total.
    """
    summary = {
        "pending": 0,
        "approved": 0,
        "rejected": 0,
        "corrected": 0,
        "total": len(studies),
    }
    for study in studies:
        key = study.review_status.value
        if key in summary:
            summary[key] += 1
    return summary
