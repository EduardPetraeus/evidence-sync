"""Extraction accuracy validation — compare extracted data against ground truth."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml

from evidence_sync.models import Study

logger = logging.getLogger(__name__)

# Fields to compare and their types
NUMERIC_INT_FIELDS = ("sample_size_treatment", "sample_size_control")
NUMERIC_FLOAT_FIELDS = ("effect_size", "ci_lower", "ci_upper", "p_value")
STRING_EXACT_FIELDS = ("study_design", "effect_measure")
FUZZY_STRING_FIELDS = ("primary_outcome",)

ALL_COMPARED_FIELDS = (
    NUMERIC_INT_FIELDS + NUMERIC_FLOAT_FIELDS + STRING_EXACT_FIELDS + FUZZY_STRING_FIELDS
)

# Tolerance settings for numeric comparisons
RELATIVE_TOLERANCE = 0.05  # 5% relative tolerance
ABSOLUTE_TOLERANCE = 0.05  # 0.05 absolute tolerance


def load_ground_truth(ground_truth_dir: Path) -> list[dict]:
    """Load all YAML ground truth files from a directory.

    Args:
        ground_truth_dir: Path to directory containing ground truth YAML files.

    Returns:
        List of ground truth dictionaries, each containing study metadata
        and a 'ground_truth' key with verified extraction values.
    """
    entries: list[dict] = []

    if not ground_truth_dir.exists():
        logger.warning(f"Ground truth directory does not exist: {ground_truth_dir}")
        return entries

    for filepath in sorted(ground_truth_dir.glob("*.yaml")):
        try:
            with open(filepath) as f:
                data = yaml.safe_load(f)
            if data and "ground_truth" in data:
                # Ensure pmid is always a string
                data["pmid"] = str(data.get("pmid", ""))
                entries.append(data)
            else:
                logger.warning(f"No ground_truth key in {filepath}, skipping")
        except Exception:
            logger.warning(f"Failed to load ground truth from {filepath}", exc_info=True)

    logger.info(f"Loaded {len(entries)} ground truth entries from {ground_truth_dir}")
    return entries


def compare_extraction(extracted: Study, ground_truth: dict) -> dict:
    """Compare one study's extracted values against ground truth.

    Args:
        extracted: Study object with extracted data.
        ground_truth: Dictionary with verified values (the 'ground_truth' sub-dict).

    Returns:
        Dictionary with per-field results:
        - field_results: dict mapping field name to {correct: bool, expected, actual, error}
        - n_correct: int
        - n_total: int
        - accuracy: float (0.0 to 1.0)
        - pmid: str
    """
    gt = ground_truth
    field_results: dict[str, dict] = {}

    for field_name in ALL_COMPARED_FIELDS:
        expected = gt.get(field_name)
        actual = _get_study_field(extracted, field_name)

        if field_name in NUMERIC_INT_FIELDS:
            correct, error = _compare_int(expected, actual)
        elif field_name in NUMERIC_FLOAT_FIELDS:
            correct, error = _compare_float(expected, actual)
        elif field_name in STRING_EXACT_FIELDS:
            correct, error = _compare_string_exact(expected, actual)
        elif field_name in FUZZY_STRING_FIELDS:
            correct, error = _compare_string_fuzzy(expected, actual)
        else:
            correct, error = False, "unknown field type"

        field_results[field_name] = {
            "correct": correct,
            "expected": expected,
            "actual": actual,
            "error": error,
        }

    n_correct = sum(1 for r in field_results.values() if r["correct"])
    n_total = len(field_results)

    return {
        "pmid": extracted.pmid,
        "field_results": field_results,
        "n_correct": n_correct,
        "n_total": n_total,
        "accuracy": n_correct / n_total if n_total > 0 else 0.0,
    }


def compute_accuracy_report(results: list[dict]) -> dict:
    """Aggregate accuracy across all studies.

    Args:
        results: List of comparison result dicts from compare_extraction.

    Returns:
        Dictionary with:
        - overall_accuracy: float (% of fields correct across all studies)
        - per_field_accuracy: dict mapping field name to accuracy float
        - numeric_error_rates: dict mapping numeric field name to mean absolute % error
        - problem_studies: list of PMIDs where accuracy < 50%
        - summary_text: human-readable summary string
    """
    if not results:
        return {
            "overall_accuracy": 0.0,
            "per_field_accuracy": {},
            "numeric_error_rates": {},
            "problem_studies": [],
            "summary_text": "No results to report.",
        }

    # Overall accuracy
    total_correct = sum(r["n_correct"] for r in results)
    total_fields = sum(r["n_total"] for r in results)
    overall_accuracy = total_correct / total_fields if total_fields > 0 else 0.0

    # Per-field accuracy
    per_field_accuracy: dict[str, float] = {}
    for field_name in ALL_COMPARED_FIELDS:
        field_correct = 0
        field_total = 0
        for r in results:
            if field_name in r["field_results"]:
                field_total += 1
                if r["field_results"][field_name]["correct"]:
                    field_correct += 1
        per_field_accuracy[field_name] = field_correct / field_total if field_total > 0 else 0.0

    # Numeric error rates (mean absolute percentage error for numeric float fields)
    numeric_error_rates: dict[str, float] = {}
    for field_name in NUMERIC_FLOAT_FIELDS:
        errors: list[float] = []
        for r in results:
            fr = r["field_results"].get(field_name, {})
            error = fr.get("error")
            if isinstance(error, (int, float)):
                errors.append(abs(error))
        numeric_error_rates[field_name] = sum(errors) / len(errors) if errors else 0.0

    # Problem studies (accuracy < 50%)
    problem_studies = [r["pmid"] for r in results if r["accuracy"] < 0.5]

    # Summary text
    lines = [
        f"Extraction Accuracy Report ({len(results)} studies)",
        f"Overall accuracy: {overall_accuracy:.1%} ({total_correct}/{total_fields} fields correct)",
        "",
        "Per-field accuracy:",
    ]
    for field_name, acc in per_field_accuracy.items():
        lines.append(f"  {field_name}: {acc:.1%}")

    if numeric_error_rates:
        lines.append("")
        lines.append("Numeric error rates (mean absolute % error):")
        for field_name, err in numeric_error_rates.items():
            lines.append(f"  {field_name}: {err:.1%}")

    if problem_studies:
        lines.append("")
        lines.append(f"Problem studies ({len(problem_studies)}):")
        for pmid in problem_studies:
            lines.append(f"  - {pmid}")

    summary_text = "\n".join(lines)

    return {
        "overall_accuracy": overall_accuracy,
        "per_field_accuracy": per_field_accuracy,
        "numeric_error_rates": numeric_error_rates,
        "problem_studies": problem_studies,
        "summary_text": summary_text,
    }


def format_accuracy_report(report: dict) -> str:
    """Format an accuracy report as markdown text.

    Args:
        report: Report dict from compute_accuracy_report.

    Returns:
        Markdown-formatted string.
    """
    lines = [
        "# Extraction Accuracy Report",
        "",
        f"**Overall accuracy:** {report['overall_accuracy']:.1%}",
        "",
        "## Per-Field Accuracy",
        "",
        "| Field | Accuracy |",
        "|-------|----------|",
    ]

    for field_name, acc in report.get("per_field_accuracy", {}).items():
        lines.append(f"| {field_name} | {acc:.1%} |")

    numeric_errors = report.get("numeric_error_rates", {})
    if numeric_errors:
        lines.extend([
            "",
            "## Numeric Error Rates",
            "",
            "| Field | Mean Abs % Error |",
            "|-------|-----------------|",
        ])
        for field_name, err in numeric_errors.items():
            lines.append(f"| {field_name} | {err:.1%} |")

    problem_studies = report.get("problem_studies", [])
    if problem_studies:
        lines.extend([
            "",
            f"## Problem Studies ({len(problem_studies)})",
            "",
        ])
        for pmid in problem_studies:
            lines.append(f"- {pmid}")

    return "\n".join(lines)


# --- Internal comparison helpers ---


def _get_study_field(study: Study, field_name: str) -> object:
    """Get a field value from a Study, converting enums to their string value."""
    value = getattr(study, field_name, None)
    if value is not None and hasattr(value, "value"):
        # Convert enum to its string value
        return value.value
    return value


def _compare_int(expected: object, actual: object) -> tuple[bool, Optional[str]]:
    """Compare integer values. Exact match required.

    Returns:
        Tuple of (is_correct, error_description).
    """
    if expected is None and actual is None:
        return True, None
    if expected is None or actual is None:
        return False, "null mismatch"
    try:
        match = int(expected) == int(actual)
        return match, None if match else "value mismatch"
    except (TypeError, ValueError):
        return False, "type error"


def _compare_float(expected: object, actual: object) -> tuple[bool, Optional[float | str]]:
    """Compare float values with tolerance.

    Tolerance rules:
    - Within 5% relative tolerance OR 0.05 absolute tolerance

    Returns:
        Tuple of (is_correct, relative_error_or_description).
    """
    if expected is None and actual is None:
        return True, None
    if expected is None or actual is None:
        return False, "null mismatch"
    try:
        exp = float(expected)
        act = float(actual)
    except (TypeError, ValueError):
        return False, "type error"

    abs_diff = abs(exp - act)

    # Check absolute tolerance
    if abs_diff <= ABSOLUTE_TOLERANCE:
        return True, 0.0

    # Check relative tolerance
    if exp != 0.0:
        rel_error = abs_diff / abs(exp)
        if rel_error <= RELATIVE_TOLERANCE:
            return True, rel_error
        return False, rel_error

    # exp is 0 but diff > absolute tolerance
    return False, abs_diff


def _compare_string_exact(expected: object, actual: object) -> tuple[bool, Optional[str]]:
    """Compare string values with case-insensitive exact match.

    Returns:
        Tuple of (is_correct, error_description).
    """
    if expected is None and actual is None:
        return True, None
    if expected is None or actual is None:
        return False, "null mismatch"
    exp_str = str(expected).lower().strip()
    act_str = str(actual).lower().strip()
    return exp_str == act_str, None if exp_str == act_str else "value mismatch"


def _compare_string_fuzzy(expected: object, actual: object) -> tuple[bool, Optional[str]]:
    """Compare strings with fuzzy matching — check if key terms overlap.

    Splits both strings into words, filters common stop words, and checks
    if there is substantial overlap (>=50% of expected terms found in actual).

    Returns:
        Tuple of (is_correct, error_description).
    """
    if expected is None and actual is None:
        return True, None
    if expected is None or actual is None:
        return False, "null mismatch"

    stop_words = {
        "a", "an", "the", "in", "of", "for", "and", "or", "to", "from",
        "with", "by", "on", "at", "is", "was", "are", "were", "be", "been",
    }

    exp_terms = {
        w.lower().strip("().,;:") for w in str(expected).split() if len(w) > 1
    } - stop_words
    act_terms = {
        w.lower().strip("().,;:") for w in str(actual).split() if len(w) > 1
    } - stop_words

    if not exp_terms:
        return True, None

    overlap = exp_terms & act_terms
    overlap_ratio = len(overlap) / len(exp_terms)

    if overlap_ratio >= 0.5:
        return True, None
    return False, f"low overlap ({overlap_ratio:.0%})"
