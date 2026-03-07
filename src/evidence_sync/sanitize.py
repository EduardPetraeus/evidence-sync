"""Input sanitization for LLM prompt injection defense."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# XML-like tags used in prompt templates that must not appear in user input
_DANGEROUS_TAG_PATTERN = re.compile(
    r"</?(?:study_title|study_abstract|study_authors|study_journal"
    r"|inclusion_criteria|exclusion_criteria|primary_outcome"
    r"|system|human|assistant)\b[^>]*>",
    re.IGNORECASE,
)


def sanitize_prompt_input(text: str, max_length: int = 5000) -> str:
    """Sanitize untrusted text before embedding in an LLM prompt.

    Truncates to max_length and strips XML-like tags that match
    the prompt template structure to prevent tag injection.

    Args:
        text: Raw input text (e.g. study title, abstract).
        max_length: Maximum allowed length after sanitization.

    Returns:
        Sanitized text safe for prompt embedding.
    """
    if not text:
        return text

    result = text[:max_length]

    cleaned = _DANGEROUS_TAG_PATTERN.sub("", result)
    if cleaned != result:
        logger.warning(
            "Stripped %d dangerous XML tag(s) from prompt input",
            len(_DANGEROUS_TAG_PATTERN.findall(result)),
        )

    return cleaned


def validate_extraction_output(data: dict) -> dict:
    """Range-check extracted values from LLM output.

    Sets out-of-range values to None and truncates overlong strings.
    This is an additional defense layer on top of extractor's _safe_int/_safe_float.

    Args:
        data: Parsed extraction dict from LLM response.

    Returns:
        The same dict with out-of-range values set to None.
    """
    _check_range(data, "sample_size_treatment", 0, 10_000_000)
    _check_range(data, "sample_size_control", 0, 10_000_000)
    _check_range(data, "effect_size", -100.0, 100.0)
    _check_range(data, "p_value", 0.0, 1.0)
    _check_range(data, "extraction_confidence", 0.0, 1.0)

    primary = data.get("primary_outcome")
    if primary is not None and len(str(primary)) > 1000:
        logger.warning("primary_outcome truncated from %d chars", len(str(primary)))
        data["primary_outcome"] = str(primary)[:1000]

    return data


def _check_range(data: dict, key: str, lo: float, hi: float) -> None:
    """Null out a numeric field if it falls outside [lo, hi]."""
    val = data.get(key)
    if val is not None and isinstance(val, (int, float)) and not isinstance(val, bool):
        if not (lo <= float(val) <= hi):
            logger.warning("%s=%s out of range [%s, %s], setting to None", key, val, lo, hi)
            data[key] = None
