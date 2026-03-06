"""Intelligent study screening with relevance scoring."""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Optional

from evidence_sync.models import ReviewConfig, ScreeningResult, Study

logger = logging.getLogger(__name__)

SCREENING_PROMPT = """You are a systematic review screening specialist. \
Assess the relevance of the following study for inclusion in a living \
meta-analysis.

<study_title>{title}</study_title>

<study_abstract>{abstract}</study_abstract>

<study_journal>{journal}</study_journal>

<inclusion_criteria>
{inclusion_criteria}
</inclusion_criteria>

<exclusion_criteria>
{exclusion_criteria}
</exclusion_criteria>

<primary_outcome>{primary_outcome}</primary_outcome>

Evaluate the study and respond with ONLY a JSON object, no other text.

{{
    "relevance_score": <float 0.0-1.0 — how relevant is this study>,
    "decision": <"include" if score > 0.7, "exclude" if score < 0.3, \
"uncertain" otherwise>,
    "reasons": [<list of strings explaining the decision>],
    "population": <string — the study population, or null>,
    "intervention": <string — the intervention studied, or null>,
    "comparator": <string — the comparator/control, or null>,
    "outcome": <string — the primary outcome measured, or null>
}}

Important:
- Base your assessment on the abstract content and the criteria provided
- A relevance score > 0.7 means the study clearly meets inclusion criteria
- A relevance score < 0.3 means the study clearly meets exclusion criteria
- Scores between 0.3 and 0.7 indicate uncertainty requiring human review
- Extract PICO elements (Population, Intervention, Comparator, Outcome) \
from the abstract"""


def _build_screening_prompt(
    study: Study,
    config: ReviewConfig,
) -> str:
    """Build the screening prompt with study and config data.

    Args:
        study: Study to screen.
        config: Review configuration with inclusion/exclusion criteria.

    Returns:
        Formatted prompt string.
    """
    inclusion = "\n".join(
        f"- {c}" for c in config.inclusion_criteria
    ) or "- Not specified"
    exclusion = "\n".join(
        f"- {c}" for c in config.exclusion_criteria
    ) or "- Not specified"

    return SCREENING_PROMPT.format(
        title=study.title,
        abstract=study.abstract,
        journal=study.journal,
        inclusion_criteria=inclusion,
        exclusion_criteria=exclusion,
        primary_outcome=config.primary_outcome,
    )


def _parse_screening_response(response_text: str) -> Optional[dict]:
    """Parse JSON response from Claude screening.

    Args:
        response_text: Raw text response from Claude.

    Returns:
        Parsed dict or None if parsing fails.
    """
    text = response_text.strip()

    # Handle markdown code blocks
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [
            line for line in lines
            if not line.strip().startswith("```")
        ]
        text = "\n".join(lines)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.error(
            f"Failed to parse screening response: {text[:200]}"
        )
        return None


def _clamp_score(score: float) -> float:
    """Clamp a relevance score to [0.0, 1.0]."""
    if score is None:
        return 0.0
    return max(0.0, min(1.0, float(score)))


def _validate_decision(
    decision: str, relevance_score: float,
) -> str:
    """Validate and normalize the screening decision.

    Args:
        decision: Raw decision string from response.
        relevance_score: The relevance score (0.0-1.0).

    Returns:
        Normalized decision: "include", "exclude", or "uncertain".
    """
    decision = str(decision).lower().strip()
    if decision in ("include", "exclude", "uncertain"):
        return decision
    # Fall back to score-based decision
    if relevance_score > 0.7:
        return "include"
    if relevance_score < 0.3:
        return "exclude"
    return "uncertain"


def screen_study(
    study: Study,
    config: ReviewConfig,
    anthropic_client: object,
    model: str = "claude-sonnet-4-20250514",
) -> ScreeningResult:
    """Screen a study for relevance using Claude.

    Args:
        study: Study with abstract text populated.
        config: Review configuration with criteria.
        anthropic_client: An initialized anthropic.Anthropic() client.
        model: Claude model to use for screening.

    Returns:
        ScreeningResult with relevance score and decision.
    """
    if not study.abstract or len(study.abstract.strip()) < 50:
        logger.warning(
            f"Study {study.pmid} has no/short abstract — skipping screening"
        )
        return ScreeningResult(
            pmid=study.pmid,
            relevance_score=0.5,
            decision="uncertain",
            reasons=["Abstract too short for meaningful screening"],
            screening_date=date.today(),
            screening_model=model,
        )

    prompt = _build_screening_prompt(study, config)

    response = anthropic_client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = response.content[0].text
    parsed = _parse_screening_response(response_text)

    if parsed:
        return _apply_screening(study, parsed, model)

    # Fallback: could not parse response
    logger.warning(
        f"Screening failed for {study.pmid}, defaulting to uncertain"
    )
    return ScreeningResult(
        pmid=study.pmid,
        relevance_score=0.5,
        decision="uncertain",
        reasons=["Failed to parse screening response"],
        screening_date=date.today(),
        screening_model=model,
    )


def _apply_screening(
    study: Study,
    data: dict,
    model: str,
) -> ScreeningResult:
    """Apply parsed screening data to study and create result.

    Args:
        study: Study to update with PICO data.
        data: Parsed screening response dict.
        model: Model name used for screening.

    Returns:
        ScreeningResult with extracted data.
    """
    score = _clamp_score(data.get("relevance_score", 0.5))
    decision = _validate_decision(
        data.get("decision", "uncertain"), score,
    )
    reasons = data.get("reasons", [])
    if not isinstance(reasons, list):
        reasons = [str(reasons)]

    # Populate PICO fields on the study
    pop = data.get("population")
    if pop is not None:
        study.population = str(pop)[:500]
    interv = data.get("intervention")
    if interv is not None:
        study.intervention = str(interv)[:500]
    comp = data.get("comparator")
    if comp is not None:
        study.comparator = str(comp)[:500]
    outc = data.get("outcome")
    if outc is not None:
        study.outcome = str(outc)[:500]

    return ScreeningResult(
        pmid=study.pmid,
        relevance_score=score,
        decision=decision,
        reasons=[str(r) for r in reasons],
        screening_date=date.today(),
        screening_model=model,
    )


def screen_study_from_dict(
    study: Study,
    screening_data: dict,
    model: str = "manual",
) -> ScreeningResult:
    """Apply pre-computed screening data (for testing).

    Args:
        study: Study to update with PICO data.
        screening_data: Dict with screening fields.
        model: Model name to record.

    Returns:
        ScreeningResult with applied data.
    """
    return _apply_screening(study, screening_data, model)


def get_screening_summary(results: list[ScreeningResult]) -> dict:
    """Summarize screening results.

    Args:
        results: List of screening results.

    Returns:
        Dict with counts by decision and average relevance score.
    """
    return {
        "total": len(results),
        "include": sum(
            1 for r in results if r.decision == "include"
        ),
        "exclude": sum(
            1 for r in results if r.decision == "exclude"
        ),
        "uncertain": sum(
            1 for r in results if r.decision == "uncertain"
        ),
        "avg_relevance": (
            sum(r.relevance_score for r in results)
            / max(len(results), 1)
        ),
    }


def rank_by_relevance(
    results: list[ScreeningResult],
) -> list[ScreeningResult]:
    """Sort screening results by relevance score (highest first).

    Args:
        results: List of screening results to sort.

    Returns:
        New list sorted by relevance_score descending.
    """
    return sorted(
        results, key=lambda r: r.relevance_score, reverse=True,
    )
