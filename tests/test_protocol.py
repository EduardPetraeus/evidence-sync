"""Tests for protocol template generation."""

from __future__ import annotations

from datetime import date

from evidence_sync.models import AnalysisResult, EffectMeasure, ReviewConfig
from evidence_sync.protocol import format_protocol_text, generate_protocol


def _make_config() -> ReviewConfig:
    return ReviewConfig(
        topic_id="ssri-depression",
        topic_name="SSRI vs Placebo for Depression",
        search_query="SSRI AND depression AND RCT",
        effect_measure=EffectMeasure.STANDARDIZED_MEAN_DIFFERENCE,
        primary_outcome="Depression severity (HAM-D)",
        inclusion_criteria=["Adults with MDD", "RCT design"],
        exclusion_criteria=["Animal studies", "Pediatric population"],
    )


def _make_result() -> AnalysisResult:
    return AnalysisResult(
        topic="ssri-depression",
        n_studies=5,
        pooled_effect=-0.31,
        pooled_ci_lower=-0.45,
        pooled_ci_upper=-0.17,
        pooled_p_value=0.001,
        effect_measure=EffectMeasure.STANDARDIZED_MEAN_DIFFERENCE,
        i_squared=42.0,
        q_statistic=6.9,
        q_p_value=0.14,
        tau_squared=0.02,
        analysis_date=date(2024, 6, 15),
    )


class TestProtocolGeneration:
    def test_basic_protocol(self):
        protocol = generate_protocol(_make_config())

        assert "title" in protocol
        assert "SSRI vs Placebo" in protocol["title"]
        assert "background" in protocol
        assert "objectives" in protocol
        assert "search_strategy" in protocol
        assert "eligibility_criteria" in protocol
        assert "data_extraction" in protocol
        assert "quality_assessment" in protocol
        assert "data_synthesis" in protocol
        assert "amendments" in protocol

    def test_search_strategy(self):
        protocol = generate_protocol(_make_config())
        search = protocol["search_strategy"]

        assert "PubMed/MEDLINE" in search["databases"]
        assert search["search_query"] == "SSRI AND depression AND RCT"

    def test_eligibility_criteria(self):
        protocol = generate_protocol(_make_config())
        elig = protocol["eligibility_criteria"]

        assert "Adults with MDD" in elig["inclusion"]
        assert "Animal studies" in elig["exclusion"]

    def test_with_result_amendments(self):
        protocol = generate_protocol(_make_config(), result=_make_result())

        assert "5 studies" in protocol["amendments"]
        assert "-0.3100" in protocol["amendments"]

    def test_without_result(self):
        protocol = generate_protocol(_make_config())

        assert "No amendments" in protocol["amendments"]

    def test_date_range(self):
        config = _make_config()
        config.min_date = "2020-01-01"
        config.max_date = "2024-12-31"

        protocol = generate_protocol(config)
        search = protocol["search_strategy"]

        assert "2020-01-01" in search["date_range"]
        assert "2024-12-31" in search["date_range"]

    def test_min_date_only(self):
        config = _make_config()
        config.min_date = "2020-01-01"

        protocol = generate_protocol(config)

        assert "onwards" in protocol["search_strategy"]["date_range"]

    def test_no_criteria(self):
        config = _make_config()
        config.inclusion_criteria = []
        config.exclusion_criteria = []

        protocol = generate_protocol(config)
        elig = protocol["eligibility_criteria"]

        assert "Not yet specified" in elig["inclusion"]

    def test_quality_assessment(self):
        protocol = generate_protocol(_make_config())
        qa = protocol["quality_assessment"]

        assert qa["tool"] == "Cochrane Risk of Bias tool"
        assert len(qa["domains"]) == 6


class TestProtocolFormatting:
    def test_format_text(self):
        protocol = generate_protocol(_make_config())
        text = format_protocol_text(protocol)

        assert "# " in text  # markdown heading
        assert "SSRI vs Placebo" in text
        assert "## Search Strategy" in text
        assert "## Eligibility Criteria" in text
        assert "## Data Extraction" in text
        assert "## Quality Assessment" in text
        assert "## Data Synthesis" in text

    def test_format_includes_search_query(self):
        protocol = generate_protocol(_make_config())
        text = format_protocol_text(protocol)

        assert "SSRI AND depression AND RCT" in text
