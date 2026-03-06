"""Protocol template generation for PROSPERO-compatible systematic review registration."""

from __future__ import annotations

from evidence_sync.models import AnalysisResult, ReviewConfig


def generate_protocol(
    config: ReviewConfig,
    result: AnalysisResult | None = None,
) -> dict:
    """Generate a PROSPERO-compatible protocol template.

    Creates a structured protocol document from the review configuration,
    suitable for registration with PROSPERO or similar registries.

    Args:
        config: Review configuration with search strategy and criteria.
        result: Meta-analysis result (if available, for amendments section).

    Returns:
        Dict with protocol sections: title, background, objectives,
        search_strategy, eligibility_criteria, data_extraction,
        quality_assessment, data_synthesis, amendments.
    """
    # Build eligibility criteria text
    inclusion_text = ""
    if config.inclusion_criteria:
        inclusion_text = "\n".join(f"- {c}" for c in config.inclusion_criteria)
    else:
        inclusion_text = "Not yet specified."

    exclusion_text = ""
    if config.exclusion_criteria:
        exclusion_text = "\n".join(f"- {c}" for c in config.exclusion_criteria)
    else:
        exclusion_text = "Not yet specified."

    # Publication types
    pub_types = (
        ", ".join(config.publication_types) if config.publication_types else "Not restricted"
    )

    # Date range
    date_range = "No date restrictions"
    if config.min_date and config.max_date:
        date_range = f"{config.min_date} to {config.max_date}"
    elif config.min_date:
        date_range = f"{config.min_date} onwards"
    elif config.max_date:
        date_range = f"Up to {config.max_date}"

    # Amendments (if result exists, note it)
    amendments = "No amendments to date."
    if result is not None:
        amendments = (
            f"Analysis completed with {result.n_studies} studies. "
            f"Pooled effect: {result.pooled_effect:.4f} "
            f"(95% CI: {result.pooled_ci_lower:.4f} to {result.pooled_ci_upper:.4f}). "
            f"I-squared: {result.i_squared:.1f}%."
        )

    return {
        "title": f"Living systematic review and meta-analysis: {config.topic_name}",
        "background": (
            f"This protocol describes a living systematic review examining "
            f"{config.topic_name}. The primary outcome is {config.primary_outcome}. "
            f"This review uses an automated pipeline (Evidence Sync) for continuous "
            f"monitoring of new evidence, with the effect measure being "
            f"{config.effect_measure.value.replace('_', ' ')}."
        ),
        "objectives": (
            f"To systematically identify, appraise, and synthesize evidence on "
            f"{config.topic_name}, measuring the primary outcome "
            f"({config.primary_outcome}) using {config.effect_measure.value.replace('_', ' ')}."
        ),
        "search_strategy": {
            "databases": ["PubMed/MEDLINE"],
            "search_query": config.search_query,
            "date_range": date_range,
            "publication_types": pub_types,
            "update_frequency": config.schedule,
        },
        "eligibility_criteria": {
            "inclusion": inclusion_text,
            "exclusion": exclusion_text,
            "population": "As defined by inclusion criteria",
            "intervention": "As defined by search strategy",
            "comparator": "As defined by inclusion criteria",
            "outcome": config.primary_outcome,
        },
        "data_extraction": {
            "method": "Automated extraction using Claude (Anthropic) from abstracts and full-text",
            "items": [
                "Effect size and 95% confidence interval",
                "Sample sizes (treatment and control)",
                "Study design",
                "Primary outcome measure",
                "Risk of bias domains",
                "PICO elements",
            ],
            "verification": "Human review of all extracted data before inclusion in meta-analysis",
        },
        "quality_assessment": {
            "tool": "Cochrane Risk of Bias tool",
            "domains": [
                "Random sequence generation",
                "Allocation concealment",
                "Blinding of participants and personnel",
                "Blinding of outcome assessment",
                "Incomplete outcome data",
                "Selective reporting",
            ],
        },
        "data_synthesis": {
            "model": "DerSimonian-Laird random-effects model",
            "effect_measure": config.effect_measure.value.replace("_", " "),
            "heterogeneity": "I-squared, Cochran's Q, tau-squared",
            "publication_bias": "Funnel plot + Egger's regression test",
            "sensitivity": "Leave-one-out analysis (planned)",
            "subgroup": "By risk of bias level (planned)",
        },
        "amendments": amendments,
    }


def format_protocol_text(protocol: dict) -> str:
    """Format protocol as readable markdown text.

    Args:
        protocol: Protocol dict from generate_protocol().

    Returns:
        Multi-line markdown string.
    """
    lines = [
        f"# {protocol['title']}",
        "",
        "## Background",
        protocol["background"],
        "",
        "## Objectives",
        protocol["objectives"],
        "",
        "## Search Strategy",
    ]

    search = protocol["search_strategy"]
    lines.append(f"- **Databases:** {', '.join(search['databases'])}")
    lines.append(f"- **Search query:** {search['search_query']}")
    lines.append(f"- **Date range:** {search['date_range']}")
    lines.append(f"- **Publication types:** {search['publication_types']}")
    lines.append(f"- **Update frequency:** {search['update_frequency']}")

    lines.extend(["", "## Eligibility Criteria", ""])
    elig = protocol["eligibility_criteria"]
    lines.append("### Inclusion Criteria")
    lines.append(elig["inclusion"])
    lines.append("")
    lines.append("### Exclusion Criteria")
    lines.append(elig["exclusion"])
    lines.append("")
    lines.append(f"- **Population:** {elig['population']}")
    lines.append(f"- **Outcome:** {elig['outcome']}")

    lines.extend(["", "## Data Extraction", ""])
    extraction = protocol["data_extraction"]
    lines.append(f"**Method:** {extraction['method']}")
    lines.append("")
    lines.append("**Data items:**")
    for item in extraction["items"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append(f"**Verification:** {extraction['verification']}")

    lines.extend(["", "## Quality Assessment", ""])
    qa = protocol["quality_assessment"]
    lines.append(f"**Tool:** {qa['tool']}")
    lines.append("")
    lines.append("**Domains:**")
    for domain in qa["domains"]:
        lines.append(f"- {domain}")

    lines.extend(["", "## Data Synthesis", ""])
    synthesis = protocol["data_synthesis"]
    lines.append(f"- **Model:** {synthesis['model']}")
    lines.append(f"- **Effect measure:** {synthesis['effect_measure']}")
    lines.append(f"- **Heterogeneity:** {synthesis['heterogeneity']}")
    lines.append(f"- **Publication bias:** {synthesis['publication_bias']}")

    lines.extend(["", "## Amendments", ""])
    lines.append(protocol["amendments"])

    return "\n".join(lines)
