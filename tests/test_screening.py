"""Tests for intelligent study screening and PICO extraction."""

from __future__ import annotations

import sys
from datetime import date
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from evidence_sync.models import (
    EffectMeasure,
    ReviewConfig,
    ScreeningResult,
    Study,
)
from evidence_sync.screening import (
    _build_screening_prompt,
    _parse_screening_response,
    get_screening_summary,
    rank_by_relevance,
    screen_study_from_dict,
)


@pytest.fixture
def basic_study() -> Study:
    """A basic study for screening tests."""
    return Study(
        pmid="99001",
        title="Fluoxetine vs placebo for major depression",
        authors=["Smith J", "Doe A"],
        journal="J Clin Psychiatry",
        publication_date=date(2021, 3, 15),
        abstract=(
            "Background: Depression is a common mental disorder. "
            "Methods: We randomized 200 adults with MDD to fluoxetine "
            "20mg (n=100) or placebo (n=100) for 8 weeks. "
            "Results: Response rate was 55% vs 35% (OR 2.27, 95% CI "
            "1.32-3.91, p=0.003). "
            "Conclusion: Fluoxetine was superior to placebo."
        ),
    )


@pytest.fixture
def screening_config() -> ReviewConfig:
    """A review config with inclusion/exclusion criteria."""
    return ReviewConfig(
        topic_id="ssri-depression",
        topic_name="SSRIs vs. Placebo for MDD",
        search_query="(SSRI OR fluoxetine) AND depression AND placebo",
        effect_measure=EffectMeasure.ODDS_RATIO,
        primary_outcome="Response rate",
        inclusion_criteria=[
            "Randomized controlled trial",
            "Adult patients with major depressive disorder",
            "SSRI vs placebo comparison",
        ],
        exclusion_criteria=[
            "Pediatric populations",
            "Non-placebo-controlled studies",
            "Case reports or reviews",
        ],
    )


class TestScreenStudyFromDict:
    """Test applying pre-computed screening data."""

    def test_basic_screening(self, basic_study):
        data = {
            "relevance_score": 0.85,
            "decision": "include",
            "reasons": ["RCT design", "SSRI vs placebo", "MDD population"],
            "population": "Adults with major depressive disorder",
            "intervention": "Fluoxetine 20mg daily",
            "comparator": "Placebo",
            "outcome": "Response rate (50% HAM-D reduction)",
        }
        result = screen_study_from_dict(basic_study, data)

        assert isinstance(result, ScreeningResult)
        assert result.pmid == "99001"
        assert result.relevance_score == 0.85
        assert result.decision == "include"
        assert len(result.reasons) == 3
        assert result.screening_date == date.today()
        assert result.screening_model == "manual"

        # Check PICO fields populated on study
        assert basic_study.population == "Adults with major depressive disorder"
        assert basic_study.intervention == "Fluoxetine 20mg daily"
        assert basic_study.comparator == "Placebo"
        assert basic_study.outcome == "Response rate (50% HAM-D reduction)"

    def test_exclude_decision(self, basic_study):
        data = {
            "relevance_score": 0.15,
            "decision": "exclude",
            "reasons": ["Not an RCT"],
        }
        result = screen_study_from_dict(basic_study, data)

        assert result.decision == "exclude"
        assert result.relevance_score == 0.15

    def test_uncertain_decision(self, basic_study):
        data = {
            "relevance_score": 0.5,
            "decision": "uncertain",
            "reasons": ["Mixed population"],
        }
        result = screen_study_from_dict(basic_study, data)

        assert result.decision == "uncertain"

    def test_score_clamping(self, basic_study):
        data = {"relevance_score": 1.5, "decision": "include"}
        result = screen_study_from_dict(basic_study, data)
        assert result.relevance_score == 1.0

        data2 = {"relevance_score": -0.3, "decision": "exclude"}
        result2 = screen_study_from_dict(basic_study, data2)
        assert result2.relevance_score == 0.0

    def test_pico_truncation(self, basic_study):
        data = {
            "relevance_score": 0.8,
            "decision": "include",
            "population": "X" * 600,
        }
        screen_study_from_dict(basic_study, data)
        assert len(basic_study.population) == 500

    def test_missing_pico_fields(self, basic_study):
        data = {
            "relevance_score": 0.6,
            "decision": "uncertain",
        }
        screen_study_from_dict(basic_study, data)

        assert basic_study.population is None
        assert basic_study.intervention is None
        assert basic_study.comparator is None
        assert basic_study.outcome is None

    def test_invalid_decision_falls_back_to_score(self, basic_study):
        data = {
            "relevance_score": 0.9,
            "decision": "maybe",
        }
        result = screen_study_from_dict(basic_study, data)
        assert result.decision == "include"

        data2 = {
            "relevance_score": 0.1,
            "decision": "xyz",
        }
        result2 = screen_study_from_dict(basic_study, data2)
        assert result2.decision == "exclude"


class TestGetScreeningSummary:
    """Test screening summary computation."""

    def test_basic_summary(self):
        results = [
            ScreeningResult(
                pmid="1", relevance_score=0.9, decision="include",
            ),
            ScreeningResult(
                pmid="2", relevance_score=0.8, decision="include",
            ),
            ScreeningResult(
                pmid="3", relevance_score=0.1, decision="exclude",
            ),
            ScreeningResult(
                pmid="4", relevance_score=0.5, decision="uncertain",
            ),
        ]
        summary = get_screening_summary(results)

        assert summary["total"] == 4
        assert summary["include"] == 2
        assert summary["exclude"] == 1
        assert summary["uncertain"] == 1
        assert summary["avg_relevance"] == pytest.approx(0.575, abs=0.001)

    def test_empty_summary(self):
        summary = get_screening_summary([])

        assert summary["total"] == 0
        assert summary["include"] == 0
        assert summary["exclude"] == 0
        assert summary["uncertain"] == 0
        assert summary["avg_relevance"] == 0.0

    def test_all_included(self):
        results = [
            ScreeningResult(
                pmid=str(i), relevance_score=0.9, decision="include",
            )
            for i in range(5)
        ]
        summary = get_screening_summary(results)

        assert summary["include"] == 5
        assert summary["exclude"] == 0
        assert summary["uncertain"] == 0


class TestRankByRelevance:
    """Test ranking screening results by relevance score."""

    def test_ranking_order(self):
        results = [
            ScreeningResult(
                pmid="1", relevance_score=0.3, decision="uncertain",
            ),
            ScreeningResult(
                pmid="2", relevance_score=0.9, decision="include",
            ),
            ScreeningResult(
                pmid="3", relevance_score=0.6, decision="uncertain",
            ),
        ]
        ranked = rank_by_relevance(results)

        assert ranked[0].pmid == "2"
        assert ranked[1].pmid == "3"
        assert ranked[2].pmid == "1"

    def test_empty_list(self):
        assert rank_by_relevance([]) == []

    def test_single_item(self):
        results = [
            ScreeningResult(
                pmid="1", relevance_score=0.5, decision="uncertain",
            ),
        ]
        ranked = rank_by_relevance(results)
        assert len(ranked) == 1
        assert ranked[0].pmid == "1"


class TestScreeningResultCreation:
    """Test ScreeningResult dataclass."""

    def test_all_fields_populated(self):
        result = ScreeningResult(
            pmid="12345",
            relevance_score=0.85,
            decision="include",
            reasons=["RCT", "Correct population"],
            screening_date=date(2026, 3, 5),
            screening_model="claude-sonnet-4-20250514",
        )

        assert result.pmid == "12345"
        assert result.relevance_score == 0.85
        assert result.decision == "include"
        assert result.reasons == ["RCT", "Correct population"]
        assert result.screening_date == date(2026, 3, 5)
        assert result.screening_model == "claude-sonnet-4-20250514"

    def test_default_fields(self):
        result = ScreeningResult(
            pmid="12345",
            relevance_score=0.5,
            decision="uncertain",
        )

        assert result.reasons == []
        assert result.screening_date is None
        assert result.screening_model is None


class TestScreenCLICommand:
    """Test the CLI screen command."""

    def test_screen_cli_all_screened(self, tmp_path):
        """CLI should exit early if all studies already screened."""
        from evidence_sync.cli import cli
        from evidence_sync.versioning import save_study

        # Set up topic directory
        topic_dir = tmp_path / "datasets" / "test-topic"
        topic_dir.mkdir(parents=True)
        studies_dir = topic_dir / "studies"
        studies_dir.mkdir()

        # Save config
        from evidence_sync.config import save_review_config

        config = ReviewConfig(
            topic_id="test-topic",
            topic_name="Test Topic",
            search_query="test",
            effect_measure=EffectMeasure.ODDS_RATIO,
            primary_outcome="Test outcome",
        )
        save_review_config(config, topic_dir / "config.yaml")

        # Save a study with PICO data already populated
        study = Study(
            pmid="11111",
            title="Already screened study",
            authors=["Test A"],
            journal="Test J",
            publication_date=date(2020, 1, 1),
            abstract="Test abstract.",
            population="Adults",
            intervention="Drug A",
            comparator="Placebo",
            outcome="Recovery",
        )
        save_study(study, studies_dir)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--base-dir", str(tmp_path), "screen", "test-topic"],
        )

        assert result.exit_code == 0
        assert "All studies already screened" in result.output

    def test_screen_cli_with_mock_screening(self, tmp_path):
        """CLI should screen unscreened studies using mock."""
        from evidence_sync.cli import cli
        from evidence_sync.config import save_review_config
        from evidence_sync.versioning import save_study

        # Set up topic
        topic_dir = tmp_path / "datasets" / "test-topic"
        topic_dir.mkdir(parents=True)
        studies_dir = topic_dir / "studies"
        studies_dir.mkdir()
        (topic_dir / "analysis").mkdir()

        config = ReviewConfig(
            topic_id="test-topic",
            topic_name="Test Topic",
            search_query="test",
            effect_measure=EffectMeasure.ODDS_RATIO,
            primary_outcome="Test outcome",
            inclusion_criteria=["RCT"],
        )
        save_review_config(config, topic_dir / "config.yaml")

        # Save unscreened study
        study = Study(
            pmid="22222",
            title="Unscreened study about something interesting",
            authors=["Doe B"],
            journal="Test J",
            publication_date=date(2021, 6, 1),
            abstract="A randomized trial of 100 patients.",
        )
        save_study(study, studies_dir)

        # Mock screen_study to avoid real API calls
        mock_result = ScreeningResult(
            pmid="22222",
            relevance_score=0.85,
            decision="include",
            reasons=["RCT"],
            screening_date=date.today(),
            screening_model="mock",
        )

        def mock_screen(study, config, client, model="mock"):
            study.population = "Adults"
            study.intervention = "Drug"
            study.comparator = "Placebo"
            study.outcome = "Response"
            return mock_result

        mock_anthropic_mod = MagicMock()
        mock_anthropic_mod.Anthropic.return_value = MagicMock()

        with patch.dict(
            sys.modules, {"anthropic": mock_anthropic_mod},
        ), patch(
            "evidence_sync.screening.screen_study",
            side_effect=mock_screen,
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "--base-dir",
                    str(tmp_path),
                    "screen",
                    "test-topic",
                ],
            )

        assert result.exit_code == 0
        assert "Screening 1 studies" in result.output
        assert "Screening complete" in result.output
        assert "Include: 1" in result.output


class TestPicoYamlRoundtrip:
    """Test PICO fields survive save/load cycle."""

    def test_roundtrip(self, tmp_path):
        from evidence_sync.versioning import load_study, save_study

        study = Study(
            pmid="33333",
            title="PICO Roundtrip Test",
            authors=["Test A"],
            journal="Test J",
            publication_date=date(2022, 1, 1),
            abstract="Test abstract.",
            population="Adults with hypertension",
            intervention="Lisinopril 10mg",
            comparator="Placebo",
            outcome="Blood pressure reduction",
        )

        filepath = save_study(study, tmp_path)
        loaded = load_study(filepath)

        assert loaded.population == "Adults with hypertension"
        assert loaded.intervention == "Lisinopril 10mg"
        assert loaded.comparator == "Placebo"
        assert loaded.outcome == "Blood pressure reduction"

    def test_roundtrip_none_pico(self, tmp_path):
        from evidence_sync.versioning import load_study, save_study

        study = Study(
            pmid="44444",
            title="No PICO Test",
            authors=["Test A"],
            journal="Test J",
            publication_date=date(2022, 1, 1),
            abstract="Test abstract.",
        )

        filepath = save_study(study, tmp_path)
        loaded = load_study(filepath)

        assert loaded.population is None
        assert loaded.intervention is None
        assert loaded.comparator is None
        assert loaded.outcome is None


class TestScreeningPromptConstruction:
    """Test that the screening prompt includes study data and criteria."""

    def test_prompt_includes_study_data(
        self, basic_study, screening_config,
    ):
        prompt = _build_screening_prompt(basic_study, screening_config)

        assert basic_study.title in prompt
        assert basic_study.abstract in prompt
        assert basic_study.journal in prompt

    def test_prompt_includes_criteria(
        self, basic_study, screening_config,
    ):
        prompt = _build_screening_prompt(basic_study, screening_config)

        for criterion in screening_config.inclusion_criteria:
            assert criterion in prompt
        for criterion in screening_config.exclusion_criteria:
            assert criterion in prompt

    def test_prompt_includes_primary_outcome(
        self, basic_study, screening_config,
    ):
        prompt = _build_screening_prompt(basic_study, screening_config)
        assert screening_config.primary_outcome in prompt

    def test_prompt_handles_empty_criteria(self, basic_study):
        config = ReviewConfig(
            topic_id="test",
            topic_name="Test",
            search_query="test",
            effect_measure=EffectMeasure.ODDS_RATIO,
            primary_outcome="Test outcome",
            inclusion_criteria=[],
            exclusion_criteria=[],
        )
        prompt = _build_screening_prompt(basic_study, config)
        assert "Not specified" in prompt


class TestParseScreeningResponse:
    """Test parsing of screening JSON responses."""

    def test_parse_clean_json(self):
        response = (
            '{"relevance_score": 0.85, "decision": "include", '
            '"reasons": ["Good match"]}'
        )
        result = _parse_screening_response(response)
        assert result is not None
        assert result["relevance_score"] == 0.85

    def test_parse_markdown_json(self):
        response = (
            '```json\n{"relevance_score": 0.5, '
            '"decision": "uncertain"}\n```'
        )
        result = _parse_screening_response(response)
        assert result is not None
        assert result["decision"] == "uncertain"

    def test_parse_invalid_json(self):
        result = _parse_screening_response("Not valid JSON")
        assert result is None
