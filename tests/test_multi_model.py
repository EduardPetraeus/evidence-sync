"""Tests for multi-model extraction (Gemini support)."""

from __future__ import annotations

import json
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from evidence_sync.extractor import (
    extract_study_data_fulltext,
    extract_study_data_gemini,
)
from evidence_sync.models import EffectMeasure, Study, StudyDesign


@pytest.fixture
def basic_study() -> Study:
    return Study(
        pmid="12345",
        title="Fluoxetine vs placebo for depression",
        authors=["Smith J", "Doe A"],
        journal="Test Journal",
        publication_date=date(2020, 6, 1),
        abstract="A randomized controlled trial of 200 patients...",
    )


GEMINI_EXTRACTION_JSON = {
    "sample_size_treatment": 100,
    "sample_size_control": 100,
    "effect_size": 1.52,
    "effect_measure": "odds_ratio",
    "ci_lower": 1.10,
    "ci_upper": 2.10,
    "p_value": 0.009,
    "study_design": "rct",
    "primary_outcome": "Response rate",
    "risk_of_bias": {
        "random_sequence_generation": "low",
        "allocation_concealment": "low",
        "blinding_participants": "low",
        "blinding_outcome": "low",
        "incomplete_outcome": "unclear",
        "selective_reporting": "low",
    },
    "extraction_confidence": 0.85,
}


class TestGeminiExtraction:
    def test_gemini_extraction_parses_json(self, basic_study):
        """Mock Gemini API, verify extraction populates study fields."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": json.dumps(GEMINI_EXTRACTION_JSON)}
                        ]
                    }
                }
            ]
        }

        with patch(
            "evidence_sync.extractor.httpx.post",
            return_value=mock_response,
        ):
            result = extract_study_data_gemini(
                basic_study, api_key="test-key"
            )

        assert result.effect_size == 1.52
        assert result.ci_lower == 1.10
        assert result.ci_upper == 2.10
        assert result.effect_measure == EffectMeasure.ODDS_RATIO
        assert result.study_design == StudyDesign.RCT
        assert result.sample_size_treatment == 100
        assert result.extraction_model == "gemini:gemini-2.0-flash"

    def test_gemini_extraction_markdown_response(self, basic_study):
        """Handle markdown code blocks in Gemini response."""
        markdown_response = (
            "```json\n" + json.dumps(GEMINI_EXTRACTION_JSON) + "\n```"
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": markdown_response}]
                    }
                }
            ]
        }

        with patch(
            "evidence_sync.extractor.httpx.post",
            return_value=mock_response,
        ):
            result = extract_study_data_gemini(
                basic_study, api_key="test-key"
            )

        assert result.effect_size == 1.52
        assert result.extraction_model == "gemini:gemini-2.0-flash"

    def test_gemini_extraction_error_handling(self, basic_study):
        """API error results in graceful failure (no crash, no data)."""
        import httpx

        with patch(
            "evidence_sync.extractor.httpx.post",
            side_effect=httpx.HTTPError("API Error"),
        ):
            result = extract_study_data_gemini(
                basic_study, api_key="test-key"
            )

        # Study returned unchanged (no extraction data)
        assert result.effect_size is None
        assert result.extraction_model is None

    def test_extraction_model_recorded(self, basic_study):
        """study.extraction_model shows 'gemini:model-name'."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": json.dumps(GEMINI_EXTRACTION_JSON)}
                        ]
                    }
                }
            ]
        }

        with patch(
            "evidence_sync.extractor.httpx.post",
            return_value=mock_response,
        ):
            extract_study_data_gemini(
                basic_study,
                api_key="test-key",
                model="gemini-1.5-pro",
            )

        assert basic_study.extraction_model == "gemini:gemini-1.5-pro"

    def test_gemini_custom_model(self, basic_study):
        """Custom model name passed through correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": json.dumps(GEMINI_EXTRACTION_JSON)}
                        ]
                    }
                }
            ]
        }

        with patch(
            "evidence_sync.extractor.httpx.post",
            return_value=mock_response,
        ) as mock_post:
            extract_study_data_gemini(
                basic_study,
                api_key="my-api-key",
                model="gemini-2.5-pro",
            )

        # Verify the URL contains the model name
        call_args = mock_post.call_args
        url = call_args[0][0]
        assert "gemini-2.5-pro" in url


class TestFulltextExtraction:
    def test_extract_fulltext_truncation(self, basic_study):
        """Full text is truncated to 4000 chars in the prompt."""
        long_text = "A" * 10000  # 10k chars
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text=json.dumps(
                    {
                        "effect_size": 1.5,
                        "ci_lower": 1.1,
                        "ci_upper": 2.0,
                    }
                )
            )
        ]
        mock_client.messages.create.return_value = mock_response

        result = extract_study_data_fulltext(
            basic_study, long_text, mock_client
        )

        # Verify the prompt was called and text was truncated
        call_args = mock_client.messages.create.call_args
        messages = call_args[1]["messages"]
        prompt_text = messages[0]["content"]
        # The truncated text (4000 A's) should be in the prompt,
        # but not 10000 A's
        assert "A" * 4000 in prompt_text
        assert "A" * 4001 not in prompt_text

        assert result.effect_size == 1.5
        assert result.data_source == "full_text"

    def test_extract_fulltext_sets_data_source(self, basic_study):
        """Full-text extraction sets data_source to 'full_text'."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text=json.dumps(
                    {
                        "effect_size": 2.0,
                        "ci_lower": 1.5,
                        "ci_upper": 2.8,
                    }
                )
            )
        ]
        mock_client.messages.create.return_value = mock_response

        result = extract_study_data_fulltext(
            basic_study, "Some full text content", mock_client
        )

        assert result.data_source == "full_text"


class TestCliExtractProviderOption:
    def test_cli_extract_provider_option(self, tmp_path):
        """--provider flag works in CLI extract command."""
        from click.testing import CliRunner

        from evidence_sync.cli import cli

        runner = CliRunner()

        # Should show gemini as valid choice in help
        result = runner.invoke(cli, ["extract", "--help"])
        assert "provider" in result.output
        assert "claude" in result.output
        assert "gemini" in result.output

    def test_cli_extract_gemini_needs_api_key(self, tmp_path):
        """Gemini provider without API key shows error."""
        import yaml
        from click.testing import CliRunner

        from evidence_sync.cli import cli
        from evidence_sync.versioning import save_study

        # Set up topic
        topic_dir = tmp_path / "datasets" / "test-topic"
        studies_dir = topic_dir / "studies"
        studies_dir.mkdir(parents=True)
        (topic_dir / "analysis").mkdir()

        config = {
            "topic_id": "test-topic",
            "topic_name": "Test Topic",
            "search_query": "test",
            "effect_measure": "odds_ratio",
            "primary_outcome": "Test outcome",
        }
        with open(topic_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)

        study = Study(
            pmid="11111111",
            title="Test Study",
            authors=["Author"],
            journal="Journal",
            publication_date=date(2023, 1, 1),
            abstract="Abstract.",
        )
        save_study(study, studies_dir)

        runner = CliRunner(env={"GEMINI_API_KEY": ""})
        result = runner.invoke(
            cli,
            [
                "--base-dir", str(tmp_path),
                "extract", "test-topic",
                "--provider", "gemini",
            ],
        )

        assert "GEMINI_API_KEY" in result.output
