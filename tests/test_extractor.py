"""Tests for Claude-powered study data extraction."""

from __future__ import annotations

from datetime import date

import pytest

from evidence_sync.extractor import _parse_extraction_response, extract_study_data_from_dict
from evidence_sync.models import BiasRisk, EffectMeasure, Study, StudyDesign


@pytest.fixture
def basic_study():
    return Study(
        pmid="12345",
        title="Fluoxetine vs placebo for depression",
        authors=["Smith J", "Doe A"],
        journal="Test Journal",
        publication_date=date(2020, 6, 1),
        abstract="A randomized controlled trial of 200 patients...",
    )


class TestParseExtractionResponse:
    def test_parse_clean_json(self):
        response = '{"effect_size": 1.5, "ci_lower": 1.1, "ci_upper": 2.1}'
        result = _parse_extraction_response(response)
        assert result is not None
        assert result["effect_size"] == 1.5

    def test_parse_markdown_json(self):
        response = '```json\n{"effect_size": 1.5, "ci_lower": 1.1}\n```'
        result = _parse_extraction_response(response)
        assert result is not None
        assert result["effect_size"] == 1.5

    def test_parse_invalid_json(self):
        response = "This is not JSON at all"
        result = _parse_extraction_response(response)
        assert result is None

    def test_parse_full_extraction(self):
        response = """{
            "sample_size_treatment": 100,
            "sample_size_control": 100,
            "effect_size": 1.52,
            "effect_measure": "odds_ratio",
            "ci_lower": 1.10,
            "ci_upper": 2.10,
            "p_value": 0.009,
            "study_design": "rct",
            "primary_outcome": "Response rate (50% reduction in HAM-D)",
            "risk_of_bias": {
                "random_sequence_generation": "low",
                "allocation_concealment": "low",
                "blinding_participants": "low",
                "blinding_outcome": "low",
                "incomplete_outcome": "unclear",
                "selective_reporting": "low"
            },
            "extraction_confidence": 0.85
        }"""
        result = _parse_extraction_response(response)
        assert result is not None
        assert result["sample_size_treatment"] == 100
        assert result["effect_measure"] == "odds_ratio"
        assert result["risk_of_bias"]["random_sequence_generation"] == "low"


class TestExtractStudyDataFromDict:
    def test_apply_basic_extraction(self, basic_study):
        data = {
            "sample_size_treatment": 100,
            "sample_size_control": 100,
            "effect_size": 1.52,
            "effect_measure": "odds_ratio",
            "ci_lower": 1.10,
            "ci_upper": 2.10,
            "p_value": 0.009,
            "study_design": "rct",
            "primary_outcome": "Response rate",
            "extraction_confidence": 0.85,
        }
        result = extract_study_data_from_dict(basic_study, data)

        assert result.sample_size_treatment == 100
        assert result.sample_size_control == 100
        assert result.effect_size == 1.52
        assert result.effect_measure == EffectMeasure.ODDS_RATIO
        assert result.ci_lower == 1.10
        assert result.ci_upper == 2.10
        assert result.p_value == 0.009
        assert result.study_design == StudyDesign.RCT
        assert result.extraction_date == date.today()
        assert result.extraction_model == "manual"

    def test_apply_with_risk_of_bias(self, basic_study):
        data = {
            "effect_size": 1.5,
            "risk_of_bias": {
                "random_sequence_generation": "low",
                "allocation_concealment": "high",
                "blinding_participants": "low",
                "blinding_outcome": "unclear",
                "incomplete_outcome": "low",
                "selective_reporting": "low",
            },
        }
        result = extract_study_data_from_dict(basic_study, data)

        assert result.risk_of_bias is not None
        assert result.risk_of_bias.random_sequence_generation == BiasRisk.LOW
        assert result.risk_of_bias.allocation_concealment == BiasRisk.HIGH
        assert result.risk_of_bias.overall == BiasRisk.HIGH

    def test_apply_with_null_fields(self, basic_study):
        data = {
            "sample_size_treatment": None,
            "effect_size": None,
            "ci_lower": None,
            "ci_upper": None,
        }
        result = extract_study_data_from_dict(basic_study, data)
        assert result.effect_size is None
        assert result.has_extractable_data is False

    def test_apply_unknown_study_design(self, basic_study):
        data = {"study_design": "some_weird_design"}
        result = extract_study_data_from_dict(basic_study, data)
        assert result.study_design == StudyDesign.UNKNOWN
