"""PRISMA 2020 compliance module — flow diagrams, checklists, decision logs, and reporting."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from evidence_sync.models import (
    AnalysisResult,
    BiasRisk,
    ReviewConfig,
    ReviewStatus,
    ScreeningResult,
    Study,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# D1: PRISMA Flow Diagram
# ---------------------------------------------------------------------------


@dataclass
class PRISMAFlowData:
    """PRISMA 2020 flow diagram counts."""

    # Identification
    records_identified: int = 0
    records_from_databases: int = 0
    records_from_registers: int = 0
    records_from_other: int = 0

    # Screening
    duplicates_removed: int = 0
    records_screened: int = 0
    records_excluded_screening: int = 0
    screening_exclusion_reasons: dict[str, int] = field(default_factory=dict)

    # Eligibility
    reports_sought: int = 0
    reports_not_retrieved: int = 0
    reports_assessed: int = 0
    reports_excluded_eligibility: int = 0
    eligibility_exclusion_reasons: dict[str, int] = field(default_factory=dict)

    # Included
    studies_included: int = 0
    studies_in_meta_analysis: int = 0


def generate_prisma_flow(
    studies: list[Study],
    screening_results: list[ScreeningResult],
    config: ReviewConfig,
    total_identified: int | None = None,
    duplicates_removed: int = 0,
) -> PRISMAFlowData:
    """Generate PRISMA 2020 flow data from pipeline state.

    Args:
        studies: All studies (included, excluded, pending).
        screening_results: Screening results with decisions and reasons.
        config: Review configuration.
        total_identified: Override for total records identified. If None,
            inferred from studies + excluded screening results.
        duplicates_removed: Number of duplicates removed during dedup.

    Returns:
        PRISMAFlowData with counts for each stage.
    """
    # Build screening decision lists
    excluded_screening = [sr for sr in screening_results if sr.decision == "exclude"]
    included_or_uncertain = [
        sr for sr in screening_results if sr.decision in ("include", "uncertain")
    ]

    # Count screening exclusion reasons
    screening_reasons: dict[str, int] = {}
    for sr in excluded_screening:
        for reason in sr.reasons:
            screening_reasons[reason] = screening_reasons.get(reason, 0) + 1

    # Eligibility: studies that were rejected during review
    rejected_studies = [s for s in studies if s.review_status == ReviewStatus.REJECTED]
    eligibility_reasons: dict[str, int] = {}
    for s in rejected_studies:
        reason = s.review_notes or "No reason provided"
        eligibility_reasons[reason] = eligibility_reasons.get(reason, 0) + 1

    # Included studies
    included_studies = [
        s for s in studies if s.review_status in (ReviewStatus.APPROVED, ReviewStatus.CORRECTED)
    ]

    # If no studies are explicitly reviewed, treat all with data as included
    any_reviewed = any(s.review_status != ReviewStatus.PENDING for s in studies)
    if not any_reviewed:
        included_studies = [s for s in studies if s.has_extractable_data]

    # Studies in meta-analysis (must have extractable data)
    in_meta = [s for s in included_studies if s.has_extractable_data]

    # Counts
    records_screened = len(screening_results)
    records_from_registers = sum(1 for s in studies if s.nct_id is not None)
    records_from_databases = len(studies) - records_from_registers

    # Total identified
    if total_identified is None:
        total_identified = len(studies) + len(excluded_screening) + duplicates_removed

    reports_assessed = len(studies) - len(excluded_screening)
    reports_not_retrieved = max(0, len(included_or_uncertain) - reports_assessed)

    return PRISMAFlowData(
        records_identified=total_identified,
        records_from_databases=max(
            records_from_databases, total_identified - records_from_registers
        ),
        records_from_registers=records_from_registers,
        records_from_other=0,
        duplicates_removed=duplicates_removed,
        records_screened=records_screened if records_screened > 0 else len(studies),
        records_excluded_screening=len(excluded_screening),
        screening_exclusion_reasons=screening_reasons,
        reports_sought=reports_assessed if reports_assessed > 0 else len(studies),
        reports_not_retrieved=reports_not_retrieved,
        reports_assessed=reports_assessed if reports_assessed > 0 else len(studies),
        reports_excluded_eligibility=len(rejected_studies),
        eligibility_exclusion_reasons=eligibility_reasons,
        studies_included=len(included_studies),
        studies_in_meta_analysis=len(in_meta),
    )


def format_prisma_flow_text(flow: PRISMAFlowData) -> str:
    """Format PRISMA flow data as a readable text diagram.

    Args:
        flow: PRISMA flow data.

    Returns:
        Multi-line string with flow diagram.
    """
    lines = [
        "PRISMA 2020 Flow Diagram",
        "=" * 60,
        "",
        "IDENTIFICATION",
        f"  Records identified from databases: {flow.records_from_databases}",
        f"  Records identified from registers: {flow.records_from_registers}",
        f"  Records from other sources: {flow.records_from_other}",
        f"  Total records identified: {flow.records_identified}",
        f"  Duplicates removed: {flow.duplicates_removed}",
        "",
        "SCREENING",
        f"  Records screened: {flow.records_screened}",
        f"  Records excluded: {flow.records_excluded_screening}",
    ]

    if flow.screening_exclusion_reasons:
        for reason, count in sorted(
            flow.screening_exclusion_reasons.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            lines.append(f"    - {reason}: {count}")

    lines.extend(
        [
            "",
            "ELIGIBILITY",
            f"  Reports sought for retrieval: {flow.reports_sought}",
            f"  Reports not retrieved: {flow.reports_not_retrieved}",
            f"  Reports assessed for eligibility: {flow.reports_assessed}",
            f"  Reports excluded: {flow.reports_excluded_eligibility}",
        ]
    )

    if flow.eligibility_exclusion_reasons:
        for reason, count in sorted(
            flow.eligibility_exclusion_reasons.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            lines.append(f"    - {reason}: {count}")

    lines.extend(
        [
            "",
            "INCLUDED",
            f"  Studies included in review: {flow.studies_included}",
            f"  Studies included in meta-analysis: {flow.studies_in_meta_analysis}",
            "",
            "=" * 60,
        ]
    )

    return "\n".join(lines)


def format_prisma_flow_dict(flow: PRISMAFlowData) -> dict:
    """Format PRISMA flow data as a dict for JSON/YAML export.

    Args:
        flow: PRISMA flow data.

    Returns:
        Dict representation of flow data.
    """
    return {
        "identification": {
            "records_identified": flow.records_identified,
            "records_from_databases": flow.records_from_databases,
            "records_from_registers": flow.records_from_registers,
            "records_from_other": flow.records_from_other,
            "duplicates_removed": flow.duplicates_removed,
        },
        "screening": {
            "records_screened": flow.records_screened,
            "records_excluded": flow.records_excluded_screening,
            "exclusion_reasons": flow.screening_exclusion_reasons,
        },
        "eligibility": {
            "reports_sought": flow.reports_sought,
            "reports_not_retrieved": flow.reports_not_retrieved,
            "reports_assessed": flow.reports_assessed,
            "reports_excluded": flow.reports_excluded_eligibility,
            "exclusion_reasons": flow.eligibility_exclusion_reasons,
        },
        "included": {
            "studies_included": flow.studies_included,
            "studies_in_meta_analysis": flow.studies_in_meta_analysis,
        },
    }


# ---------------------------------------------------------------------------
# D3: Inclusion/Exclusion Decision Log
# ---------------------------------------------------------------------------


@dataclass
class DecisionLogEntry:
    """A single inclusion/exclusion decision."""

    pmid: str
    stage: str  # "screening" or "eligibility"
    decision: str  # "include", "exclude", "uncertain"
    reasons: list[str] = field(default_factory=list)
    timestamp: str = ""
    reviewer: str = "automated"


def build_decision_log(
    studies: list[Study],
    screening_results: list[ScreeningResult],
) -> list[DecisionLogEntry]:
    """Build complete decision log from screening and review data.

    Args:
        studies: All studies with review status.
        screening_results: Screening results with decisions.

    Returns:
        List of DecisionLogEntry objects, ordered by stage then PMID.
    """
    entries: list[DecisionLogEntry] = []

    # Screening decisions
    for sr in screening_results:
        entries.append(
            DecisionLogEntry(
                pmid=sr.pmid,
                stage="screening",
                decision=sr.decision,
                reasons=list(sr.reasons),
                timestamp=sr.screening_date.isoformat() if sr.screening_date else "",
                reviewer=sr.screening_model or "automated",
            )
        )

    # Eligibility decisions (from review workflow)
    for study in studies:
        if study.review_status == ReviewStatus.PENDING:
            continue

        if study.review_status == ReviewStatus.REJECTED:
            decision = "exclude"
        elif study.review_status in (ReviewStatus.APPROVED, ReviewStatus.CORRECTED):
            decision = "include"
        else:
            continue

        reasons = []
        if study.review_notes:
            reasons.append(study.review_notes)

        entries.append(
            DecisionLogEntry(
                pmid=study.pmid,
                stage="eligibility",
                decision=decision,
                reasons=reasons,
                timestamp=study.review_date.isoformat() if study.review_date else "",
                reviewer=study.reviewer or "unknown",
            )
        )

    # Sort by stage (screening first) then by PMID
    stage_order = {"screening": 0, "eligibility": 1}
    entries.sort(key=lambda e: (stage_order.get(e.stage, 2), e.pmid))
    return entries


def format_decision_log_text(entries: list[DecisionLogEntry]) -> str:
    """Format decision log as readable text.

    Args:
        entries: Decision log entries.

    Returns:
        Multi-line formatted string.
    """
    if not entries:
        return "No decisions recorded."

    lines = [
        "Inclusion/Exclusion Decision Log",
        "=" * 60,
        "",
        f"{'PMID':<15} {'Stage':<15} {'Decision':<12} {'Reviewer':<20} Reasons",
        "-" * 90,
    ]

    for entry in entries:
        reasons_str = "; ".join(entry.reasons) if entry.reasons else "-"
        lines.append(
            f"{entry.pmid:<15} {entry.stage:<15} {entry.decision:<12} "
            f"{entry.reviewer:<20} {reasons_str}"
        )

    lines.append("")
    lines.append(f"Total entries: {len(entries)}")
    return "\n".join(lines)


def get_exclusion_reasons_summary(
    entries: list[DecisionLogEntry],
) -> dict[str, dict[str, int]]:
    """Summarize exclusion reasons by stage.

    Args:
        entries: Decision log entries.

    Returns:
        Dict mapping stage -> {reason: count}.
    """
    summary: dict[str, dict[str, int]] = {}
    for entry in entries:
        if entry.decision != "exclude":
            continue
        if entry.stage not in summary:
            summary[entry.stage] = {}
        for reason in entry.reasons:
            summary[entry.stage][reason] = summary[entry.stage].get(reason, 0) + 1
    return summary


# ---------------------------------------------------------------------------
# D2: PRISMA 27-Item Checklist
# ---------------------------------------------------------------------------

# PRISMA 2020 checklist items
# (number, section, topic, description)
_PRISMA_ITEMS: list[tuple[int, str, str, str]] = [
    (1, "Title", "Title", "Identify the report as a systematic review."),
    (
        2,
        "Abstract",
        "Abstract",
        (
            "Provide a structured abstract including background, "
            "objectives, methods, results, and conclusions."
        ),
    ),
    (
        3,
        "Introduction",
        "Rationale",
        ("Describe the rationale for the review in the context of existing knowledge."),
    ),
    (
        4,
        "Introduction",
        "Objectives",
        ("Provide an explicit statement of the objectives or questions the review addresses."),
    ),
    (5, "Methods", "Eligibility criteria", ("Specify the inclusion and exclusion criteria.")),
    (
        6,
        "Methods",
        "Information sources",
        ("Specify all databases, registers, and other sources searched or consulted."),
    ),
    (
        7,
        "Methods",
        "Search strategy",
        ("Present the full search strategies for all databases and registers."),
    ),
    (
        8,
        "Methods",
        "Selection process",
        ("Specify the methods used to decide whether a study met the inclusion criteria."),
    ),
    (
        9,
        "Methods",
        "Data collection process",
        ("Specify the methods used to collect data from reports."),
    ),
    (
        10,
        "Methods",
        "Data items",
        ("List and define all outcomes and other variables for which data were sought."),
    ),
    (
        11,
        "Methods",
        "Study risk of bias assessment",
        ("Specify the methods used for assessing risk of bias in included studies."),
    ),
    (
        12,
        "Methods",
        "Effect measures",
        ("Specify for each outcome the effect measure used in the synthesis."),
    ),
    (
        13,
        "Methods",
        "Synthesis methods",
        ("Describe the processes used to decide which studies were eligible for each synthesis."),
    ),
    (
        14,
        "Methods",
        "Reporting bias assessment",
        ("Describe any methods used to assess risk of bias due to missing results."),
    ),
    (
        15,
        "Methods",
        "Certainty assessment",
        ("Describe any methods used to assess certainty in the body of evidence."),
    ),
    (
        16,
        "Results",
        "Study selection",
        ("Describe the results of the search and selection process with a flow diagram."),
    ),
    (
        17,
        "Results",
        "Study characteristics",
        ("Cite each included study and present its characteristics."),
    ),
    (
        18,
        "Results",
        "Risk of bias in studies",
        ("Present assessments of risk of bias for each included study."),
    ),
    (
        19,
        "Results",
        "Results of individual studies",
        ("For all outcomes, present for each study data and effect estimates with CIs."),
    ),
    (
        20,
        "Results",
        "Results of syntheses",
        ("Present results of each meta-analysis, with CIs and measures of heterogeneity."),
    ),
    (
        21,
        "Results",
        "Reporting biases",
        ("Present assessments of risk of bias due to missing results."),
    ),
    (
        22,
        "Results",
        "Certainty of evidence",
        ("Present assessments of certainty in the body of evidence for each outcome."),
    ),
    (
        23,
        "Discussion",
        "Discussion",
        ("Provide a general interpretation of the results and implications."),
    ),
    (
        24,
        "Other",
        "Registration and protocol",
        ("Provide registration information including register name and number."),
    ),
    (
        25,
        "Other",
        "Support",
        ("Describe sources of financial or non-financial support for the review."),
    ),
    (26, "Other", "Competing interests", ("Declare any competing interests of review authors.")),
    (
        27,
        "Other",
        "Availability of data",
        ("Report which of the following are publicly available and where they can be found."),
    ),
]


@dataclass
class PRISMAChecklistItem:
    """A single PRISMA checklist item with compliance status."""

    number: int
    section: str
    topic: str
    description: str
    compliant: bool
    evidence: str
    notes: str = ""


def generate_prisma_checklist(
    studies: list[Study],
    config: ReviewConfig,
    result: AnalysisResult | None = None,
    flow: PRISMAFlowData | None = None,
    protocol: dict | None = None,
) -> list[PRISMAChecklistItem]:
    """Generate PRISMA 2020 checklist with compliance status.

    Auto-assesses each of the 27 items based on available data.

    Args:
        studies: All studies in the review.
        config: Review configuration.
        result: Meta-analysis result (if available).
        flow: PRISMA flow data (if available).
        protocol: Protocol dict (if available).

    Returns:
        List of 27 PRISMAChecklistItem objects.
    """
    has_studies = len(studies) > 0
    has_data = any(s.has_extractable_data for s in studies)
    has_pico = any(s.population is not None for s in studies)
    has_rob = any(s.risk_of_bias is not None for s in studies)
    has_criteria = bool(config.inclusion_criteria or config.exclusion_criteria)
    has_query = bool(config.search_query)
    has_result = result is not None
    has_flow = flow is not None
    has_protocol = protocol is not None
    any_reviewed = any(s.review_status != ReviewStatus.PENDING for s in studies)

    # Build compliance map
    compliance: dict[int, tuple[bool, str, str]] = {
        1: (
            bool(config.topic_name),
            f"Topic: {config.topic_name}" if config.topic_name else "",
            "Auto-detected from config.topic_name",
        ),
        2: (
            False,
            "",
            "Abstract must be written manually for publication",
        ),
        3: (
            False,
            "",
            "Rationale must be written manually for publication",
        ),
        4: (
            bool(config.primary_outcome),
            f"Primary outcome: {config.primary_outcome}" if config.primary_outcome else "",
            "Auto-detected from config.primary_outcome",
        ),
        5: (
            has_criteria,
            f"Inclusion: {len(config.inclusion_criteria)} criteria, "
            f"Exclusion: {len(config.exclusion_criteria)} criteria"
            if has_criteria
            else "",
            "Auto-detected from config criteria",
        ),
        6: (
            True,
            "PubMed E-utilities API",
            "Evidence Sync uses PubMed as primary source",
        ),
        7: (
            has_query,
            f"Query: {config.search_query}" if has_query else "",
            "Auto-detected from config.search_query",
        ),
        8: (
            has_studies,
            f"Automated screening pipeline with {len(studies)} studies",
            "Selection via AI screening + human review"
            if any_reviewed
            else "Selection via AI screening",
        ),
        9: (
            has_data,
            "Claude API structured extraction from abstracts/full-text",
            "Automated data collection with human review option",
        ),
        10: (
            bool(config.effect_measure and config.primary_outcome),
            f"Effect measure: {config.effect_measure.value}, Outcome: {config.primary_outcome}"
            if config.effect_measure
            else "",
            "Auto-detected from config",
        ),
        11: (
            has_rob,
            f"{sum(1 for s in studies if s.risk_of_bias is not None)} studies with RoB assessment",
            "Cochrane Risk of Bias tool (6 domains)",
        ),
        12: (
            bool(config.effect_measure),
            f"Effect measure: {config.effect_measure.value}" if config.effect_measure else "",
            "Auto-detected from config.effect_measure",
        ),
        13: (
            has_result,
            "DerSimonian-Laird random-effects model" if has_result else "",
            "Automated meta-analysis engine",
        ),
        14: (
            has_result and result is not None and result.egger_p_value is not None,
            f"Egger's test: p={result.egger_p_value:.3f}"
            if has_result and result is not None and result.egger_p_value is not None
            else "",
            "Egger's regression test for funnel plot asymmetry",
        ),
        15: (
            False,
            "",
            "GRADE assessment available via generate_summary_of_findings()",
        ),
        16: (
            has_flow,
            "PRISMA flow diagram generated" if has_flow else "",
            "Auto-generated from pipeline data",
        ),
        17: (
            has_pico,
            f"{sum(1 for s in studies if s.population)} studies with PICO data",
            "PICO extracted during screening",
        ),
        18: (
            has_rob,
            f"{sum(1 for s in studies if s.risk_of_bias is not None)} studies assessed",
            "Risk of bias heatmap available in dashboard",
        ),
        19: (
            has_data,
            f"{sum(1 for s in studies if s.has_extractable_data)} studies with effect estimates",
            "Effect sizes with 95% CIs",
        ),
        20: (
            has_result,
            f"Pooled effect: {result.pooled_effect:.4f} "
            f"[{result.pooled_ci_lower:.4f}, {result.pooled_ci_upper:.4f}], "
            f"I²={result.i_squared:.1f}%"
            if has_result and result is not None
            else "",
            "Random-effects meta-analysis with heterogeneity metrics",
        ),
        21: (
            has_result and result is not None and result.egger_p_value is not None,
            "Funnel plot + Egger's test" if has_result else "",
            "Publication bias assessment",
        ),
        22: (
            False,
            "",
            "GRADE certainty assessment available via generate_summary_of_findings()",
        ),
        23: (
            False,
            "",
            "Discussion must be written manually for publication",
        ),
        24: (
            has_protocol,
            "Protocol template generated" if has_protocol else "",
            "PROSPERO-compatible protocol available",
        ),
        25: (
            False,
            "",
            "Support/funding information must be provided manually",
        ),
        26: (
            False,
            "",
            "Competing interests must be declared manually",
        ),
        27: (
            True,
            "Git-versioned YAML datasets, open-source pipeline",
            "All data available in repository datasets/ directory",
        ),
    }

    items = []
    for num, section, topic, description in _PRISMA_ITEMS:
        compliant, evidence, item_notes = compliance.get(num, (False, "", ""))
        items.append(
            PRISMAChecklistItem(
                number=num,
                section=section,
                topic=topic,
                description=description,
                compliant=compliant,
                evidence=evidence,
                notes=item_notes,
            )
        )
    return items


def format_checklist_text(items: list[PRISMAChecklistItem]) -> str:
    """Format PRISMA checklist as a readable text table.

    Args:
        items: Checklist items.

    Returns:
        Multi-line formatted string.
    """
    lines = [
        "PRISMA 2020 Checklist",
        "=" * 80,
        "",
        f"{'#':<4} {'Section':<15} {'Topic':<30} {'Status':<10} Evidence",
        "-" * 80,
    ]

    compliant_count = 0
    for item in items:
        status = "YES" if item.compliant else "NO"
        if item.compliant:
            compliant_count += 1
        evidence_short = item.evidence[:40] if item.evidence else "-"
        lines.append(
            f"{item.number:<4} {item.section:<15} {item.topic:<30} {status:<10} {evidence_short}"
        )

    lines.extend(
        [
            "",
            "-" * 80,
            f"Compliance: {compliant_count}/{len(items)} items "
            f"({compliant_count / max(len(items), 1) * 100:.0f}%)",
        ]
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# D5: Auto-Generated Methods Section
# ---------------------------------------------------------------------------


def generate_methods_section(
    config: ReviewConfig,
    flow: PRISMAFlowData | None = None,
    result: AnalysisResult | None = None,
) -> str:
    """Generate a draft Methods section for a systematic review paper.

    Covers: search strategy, eligibility criteria, data extraction,
    quality assessment, statistical methods.

    Args:
        config: Review configuration.
        flow: PRISMA flow data (if available).
        result: Meta-analysis result (if available).

    Returns:
        Multi-paragraph Methods section text.
    """
    paragraphs: list[str] = []

    # Search strategy
    search_para = (
        f"A systematic search was conducted using PubMed/MEDLINE "
        f'with the following search strategy: "{config.search_query}".'
    )
    if config.min_date or config.max_date:
        date_range = ""
        if config.min_date and config.max_date:
            date_range = f" from {config.min_date} to {config.max_date}"
        elif config.min_date:
            date_range = f" from {config.min_date} onwards"
        elif config.max_date:
            date_range = f" up to {config.max_date}"
        search_para += f" The search was limited to publications{date_range}."
    if config.publication_types:
        types_str = ", ".join(config.publication_types)
        search_para += f" Publication types were restricted to: {types_str}."
    paragraphs.append(search_para)

    # Eligibility criteria
    if config.inclusion_criteria or config.exclusion_criteria:
        criteria_para = "Studies were eligible for inclusion if they met the following criteria: "
        if config.inclusion_criteria:
            criteria_list = "; ".join(config.inclusion_criteria)
            criteria_para += f"{criteria_list}. "
        if config.exclusion_criteria:
            excl_list = "; ".join(config.exclusion_criteria)
            criteria_para += f"Studies were excluded if: {excl_list}."
        paragraphs.append(criteria_para)

    # Data extraction
    extraction_para = (
        "Data extraction was performed using an automated pipeline powered by "
        "Claude (Anthropic) for structured data extraction from study abstracts "
        "and full-text articles where available. Extracted data items included: "
        f"effect sizes ({config.effect_measure.value.replace('_', ' ')}), "
        "95% confidence intervals, sample sizes, study design, and "
        f"the primary outcome ({config.primary_outcome})."
    )
    paragraphs.append(extraction_para)

    # Quality assessment
    quality_para = (
        "Risk of bias was assessed using the Cochrane Risk of Bias tool, "
        "evaluating six domains: random sequence generation, allocation "
        "concealment, blinding of participants and personnel, blinding of "
        "outcome assessment, incomplete outcome data, and selective reporting."
    )
    paragraphs.append(quality_para)

    # Statistical methods
    stats_para = (
        "Meta-analysis was performed using the DerSimonian-Laird random-effects model. "
        "Heterogeneity was assessed using the I-squared statistic and Cochran's Q test. "
        "Publication bias was evaluated using funnel plot visual inspection and "
        "Egger's regression test."
    )
    if result is not None:
        stats_para += (
            f" The analysis included {result.n_studies} studies with "
            f"a pooled {config.effect_measure.value.replace('_', ' ')} of "
            f"{result.pooled_effect:.3f} "
            f"(95% CI: {result.pooled_ci_lower:.3f} to {result.pooled_ci_upper:.3f})."
        )
    paragraphs.append(stats_para)

    # Flow summary
    if flow is not None:
        flow_para = (
            f"The search identified {flow.records_identified} records. "
            f"After removing {flow.duplicates_removed} duplicates, "
            f"{flow.records_screened} records were screened, of which "
            f"{flow.records_excluded_screening} were excluded. "
            f"{flow.reports_assessed} full-text reports were assessed "
            f"for eligibility, with {flow.reports_excluded_eligibility} excluded. "
            f"A total of {flow.studies_included} studies were included in "
            f"the qualitative synthesis, and {flow.studies_in_meta_analysis} "
            f"were included in the quantitative meta-analysis."
        )
        paragraphs.append(flow_para)

    return "\n\n".join(paragraphs)


# ---------------------------------------------------------------------------
# D6: Summary of Findings (GRADE format)
# ---------------------------------------------------------------------------


@dataclass
class GRADEAssessment:
    """GRADE certainty assessment for an outcome."""

    outcome: str = ""
    n_studies: int = 0
    n_participants: int = 0
    effect_estimate: str = ""
    certainty: str = "low"  # "high", "moderate", "low", "very_low"
    importance: str = "critical"  # "critical", "important", "not_important"
    risk_of_bias: str = "not_serious"  # "serious", "not_serious"
    inconsistency: str = "not_serious"
    indirectness: str = "not_serious"
    imprecision: str = "not_serious"
    publication_bias: str = "undetected"  # "detected", "undetected"


def generate_summary_of_findings(
    studies: list[Study],
    result: AnalysisResult,
    config: ReviewConfig,
) -> GRADEAssessment:
    """Generate GRADE Summary of Findings assessment.

    Auto-assesses each GRADE domain based on available data.

    Args:
        studies: Included studies.
        result: Meta-analysis result.
        config: Review configuration.

    Returns:
        GRADEAssessment with certainty rating.
    """
    included = [s for s in studies if s.has_extractable_data]

    # Total participants
    n_participants = sum(s.sample_size_total or 0 for s in included)

    # Effect estimate string
    effect_str = (
        f"{result.pooled_effect:.3f} "
        f"(95% CI: {result.pooled_ci_lower:.3f} to {result.pooled_ci_upper:.3f})"
    )

    # Risk of bias assessment
    rob_serious = "not_serious"
    studies_with_rob = [s for s in included if s.risk_of_bias is not None]
    if studies_with_rob:
        high_rob_count = sum(
            1
            for s in studies_with_rob
            if s.risk_of_bias is not None and s.risk_of_bias.overall == BiasRisk.HIGH
        )
        if high_rob_count > len(studies_with_rob) / 2:
            rob_serious = "serious"

    # Inconsistency (based on I-squared)
    inconsistency = "not_serious"
    if result.i_squared > 75.0:
        inconsistency = "serious"
    elif result.i_squared > 50.0:
        inconsistency = "not_serious"  # moderate, but not serious

    # Imprecision (based on CI width and sample size)
    imprecision = "not_serious"
    ci_width = abs(result.pooled_ci_upper - result.pooled_ci_lower)
    if n_participants < 300 or ci_width > abs(result.pooled_effect) * 2:
        imprecision = "serious"

    # Publication bias
    pub_bias = "undetected"
    if result.egger_p_value is not None and result.egger_p_value < 0.05:
        pub_bias = "detected"

    # Calculate certainty
    downgrades = 0
    if rob_serious == "serious":
        downgrades += 1
    if inconsistency == "serious":
        downgrades += 1
    if imprecision == "serious":
        downgrades += 1
    if pub_bias == "detected":
        downgrades += 1
    # indirectness stays not_serious unless manually assessed

    certainty_levels = ["high", "moderate", "low", "very_low"]
    certainty_index = min(downgrades, 3)
    certainty = certainty_levels[certainty_index]

    return GRADEAssessment(
        outcome=config.primary_outcome,
        n_studies=result.n_studies,
        n_participants=n_participants,
        effect_estimate=effect_str,
        certainty=certainty,
        importance="critical",
        risk_of_bias=rob_serious,
        inconsistency=inconsistency,
        indirectness="not_serious",
        imprecision=imprecision,
        publication_bias=pub_bias,
    )


def format_sof_text(assessment: GRADEAssessment) -> str:
    """Format Summary of Findings as a readable text table.

    Args:
        assessment: GRADE assessment.

    Returns:
        Multi-line formatted string.
    """
    lines = [
        "Summary of Findings (GRADE)",
        "=" * 70,
        "",
        f"Outcome: {assessment.outcome}",
        f"Number of studies: {assessment.n_studies}",
        f"Number of participants: {assessment.n_participants}",
        f"Effect estimate: {assessment.effect_estimate}",
        "",
        "GRADE Domains:",
        f"  Risk of bias:      {assessment.risk_of_bias}",
        f"  Inconsistency:     {assessment.inconsistency}",
        f"  Indirectness:      {assessment.indirectness}",
        f"  Imprecision:       {assessment.imprecision}",
        f"  Publication bias:  {assessment.publication_bias}",
        "",
        f"Overall certainty:   {assessment.certainty.upper().replace('_', ' ')}",
        f"Importance:          {assessment.importance}",
        "",
        "=" * 70,
    ]
    return "\n".join(lines)
