"""Tests for the Streamlit dashboard and CLI commands (status, list, dashboard)."""

from __future__ import annotations

from datetime import date

import plotly.graph_objects as go
import pytest
from click.testing import CliRunner

from evidence_sync.app import (
    build_evidence_timeline,
    build_forest_plot,
    build_funnel_plot,
    build_rob_heatmap,
    build_study_table,
    discover_topics,
)
from evidence_sync.cli import cli
from evidence_sync.models import (
    AnalysisResult,
    BiasRisk,
    EffectMeasure,
    RiskOfBias,
)
from tests.conftest import make_study

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def analysis_result() -> AnalysisResult:
    """A sample analysis result for dashboard tests."""
    return AnalysisResult(
        topic="test",
        n_studies=5,
        pooled_effect=1.65,
        pooled_ci_lower=1.35,
        pooled_ci_upper=2.01,
        pooled_p_value=0.001,
        effect_measure=EffectMeasure.ODDS_RATIO,
        i_squared=25.0,
        q_statistic=5.3,
        q_p_value=0.26,
        tau_squared=0.02,
        egger_intercept=0.5,
        egger_p_value=0.42,
        analysis_date=date(2026, 3, 5),
        studies_included=["10001", "10002", "10003", "10004", "10005"],
    )


@pytest.fixture
def studies_with_rob() -> list:
    """Studies with risk of bias data for heatmap testing."""
    return [
        make_study(
            pmid="10001",
            effect_size=1.52,
            ci_lower=1.10,
            ci_upper=2.10,
            risk_of_bias=RiskOfBias(
                random_sequence_generation=BiasRisk.LOW,
                allocation_concealment=BiasRisk.LOW,
                blinding_participants=BiasRisk.LOW,
                blinding_outcome=BiasRisk.LOW,
                incomplete_outcome=BiasRisk.UNCLEAR,
                selective_reporting=BiasRisk.LOW,
            ),
        ),
        make_study(
            pmid="10002",
            effect_size=1.68,
            ci_lower=1.22,
            ci_upper=2.31,
            risk_of_bias=RiskOfBias(
                random_sequence_generation=BiasRisk.LOW,
                allocation_concealment=BiasRisk.HIGH,
                blinding_participants=BiasRisk.UNCLEAR,
                blinding_outcome=BiasRisk.LOW,
                incomplete_outcome=BiasRisk.HIGH,
                selective_reporting=BiasRisk.LOW,
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Import test
# ---------------------------------------------------------------------------


class TestAppImport:
    def test_app_module_imports(self):
        """Verify that app.py can be imported without errors."""
        import evidence_sync.app  # noqa: F401

    def test_main_function_exists(self):
        """Verify that the main() entry point exists."""
        from evidence_sync.app import main

        assert callable(main)


# ---------------------------------------------------------------------------
# Forest plot
# ---------------------------------------------------------------------------


class TestBuildForestPlot:
    def test_returns_figure(self, sample_studies, analysis_result):
        fig = build_forest_plot(sample_studies, analysis_result)
        assert isinstance(fig, go.Figure)

    def test_has_traces(self, sample_studies, analysis_result):
        fig = build_forest_plot(sample_studies, analysis_result)
        assert len(fig.data) > 0

    def test_empty_studies_returns_figure(self, analysis_result):
        fig = build_forest_plot([], analysis_result)
        assert isinstance(fig, go.Figure)

    def test_no_extractable_data(self, analysis_result):
        studies = [make_study(pmid="1"), make_study(pmid="2")]
        fig = build_forest_plot(studies, analysis_result)
        assert isinstance(fig, go.Figure)

    def test_title_is_set(self, sample_studies, analysis_result):
        fig = build_forest_plot(sample_studies, analysis_result, title="My Title")
        assert fig.layout.title.text == "My Title"


# ---------------------------------------------------------------------------
# Funnel plot
# ---------------------------------------------------------------------------


class TestBuildFunnelPlot:
    def test_returns_figure(self, sample_studies, analysis_result):
        fig = build_funnel_plot(sample_studies, analysis_result)
        assert isinstance(fig, go.Figure)

    def test_has_study_markers(self, sample_studies, analysis_result):
        fig = build_funnel_plot(sample_studies, analysis_result)
        assert len(fig.data) >= 1

    def test_empty_studies(self, analysis_result):
        fig = build_funnel_plot([], analysis_result)
        assert isinstance(fig, go.Figure)

    def test_egger_annotation(self, sample_studies, analysis_result):
        """When egger_p_value is present, annotation should appear."""
        fig = build_funnel_plot(sample_studies, analysis_result)
        # The egger annotation is in layout.annotations
        annotations = fig.layout.annotations
        assert any("Egger" in a.text for a in annotations if a.text)


# ---------------------------------------------------------------------------
# Evidence timeline
# ---------------------------------------------------------------------------


class TestBuildEvidenceTimeline:
    def test_returns_figure(self, sample_studies, analysis_result):
        fig = build_evidence_timeline(sample_studies, analysis_result)
        assert isinstance(fig, go.Figure)

    def test_empty_studies(self, analysis_result):
        fig = build_evidence_timeline([], analysis_result)
        assert isinstance(fig, go.Figure)

    def test_has_trace(self, sample_studies, analysis_result):
        fig = build_evidence_timeline(sample_studies, analysis_result)
        assert len(fig.data) > 0


# ---------------------------------------------------------------------------
# Risk of bias heatmap
# ---------------------------------------------------------------------------


class TestBuildRobHeatmap:
    def test_returns_figure(self, studies_with_rob):
        fig = build_rob_heatmap(studies_with_rob)
        assert isinstance(fig, go.Figure)

    def test_has_heatmap_trace(self, studies_with_rob):
        fig = build_rob_heatmap(studies_with_rob)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Heatmap)

    def test_no_rob_data(self):
        studies = [make_study(pmid="1"), make_study(pmid="2")]
        fig = build_rob_heatmap(studies)
        assert isinstance(fig, go.Figure)
        # Should have annotation about no data
        assert len(fig.layout.annotations) > 0

    def test_heatmap_dimensions(self, studies_with_rob):
        fig = build_rob_heatmap(studies_with_rob)
        heatmap = fig.data[0]
        # 2 studies x 6 domains
        assert len(heatmap.z) == 2
        assert len(heatmap.z[0]) == 6


# ---------------------------------------------------------------------------
# Study table
# ---------------------------------------------------------------------------


class TestBuildStudyTable:
    def test_returns_list(self, sample_studies, analysis_result):
        table = build_study_table(sample_studies, analysis_result)
        assert isinstance(table, list)
        assert len(table) > 0

    def test_table_has_expected_keys(self, sample_studies, analysis_result):
        table = build_study_table(sample_studies, analysis_result)
        row = table[0]
        assert "PMID" in row
        assert "Effect Size" in row
        assert "CI Lower" in row
        assert "CI Upper" in row
        assert "Weight (%)" in row

    def test_empty_studies(self, analysis_result):
        table = build_study_table([], analysis_result)
        assert table == []


# ---------------------------------------------------------------------------
# Topic discovery
# ---------------------------------------------------------------------------


class TestDiscoverTopics:
    def test_finds_topics(self, tmp_path):
        # Create a mock datasets structure
        topic_dir = tmp_path / "datasets" / "test-topic"
        topic_dir.mkdir(parents=True)
        (topic_dir / "config.yaml").write_text(
            "topic_id: test-topic\n"
            "topic_name: Test\n"
            "search_query: test\n"
            "primary_outcome: test\n"
        )

        topics = discover_topics(tmp_path)
        assert topics == ["test-topic"]

    def test_no_datasets_dir(self, tmp_path):
        topics = discover_topics(tmp_path)
        assert topics == []

    def test_skips_dirs_without_config(self, tmp_path):
        datasets = tmp_path / "datasets"
        (datasets / "no-config").mkdir(parents=True)
        (datasets / "has-config").mkdir(parents=True)
        (datasets / "has-config" / "config.yaml").write_text(
            "topic_id: has-config\n"
            "topic_name: Has Config\n"
            "search_query: test\n"
            "primary_outcome: test\n"
        )
        topics = discover_topics(tmp_path)
        assert topics == ["has-config"]


# ---------------------------------------------------------------------------
# CLI: status command
# ---------------------------------------------------------------------------


class TestStatusCommand:
    def test_status_topic_not_found(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["--base-dir", str(tmp_path), "status", "nonexistent"])
        assert result.exit_code == 0
        assert "not found" in result.output

    def test_status_shows_topic_info(self, tmp_path):
        # Set up a topic with config
        topic_dir = tmp_path / "datasets" / "test-topic"
        topic_dir.mkdir(parents=True)
        (topic_dir / "config.yaml").write_text(
            "topic_id: test-topic\n"
            "topic_name: Test Topic\n"
            "search_query: test\n"
            "effect_measure: odds_ratio\n"
            "primary_outcome: response\n"
        )
        (topic_dir / "studies").mkdir()
        (topic_dir / "analysis").mkdir()

        runner = CliRunner()
        result = runner.invoke(cli, ["--base-dir", str(tmp_path), "status", "test-topic"])
        assert result.exit_code == 0
        assert "Test Topic" in result.output
        assert "0 total" in result.output
        assert "No analysis available" in result.output


# ---------------------------------------------------------------------------
# CLI: list command
# ---------------------------------------------------------------------------


class TestListCommand:
    def test_list_no_datasets(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["--base-dir", str(tmp_path), "list"])
        assert result.exit_code == 0
        assert "No datasets directory" in result.output

    def test_list_shows_topics(self, tmp_path):
        topic_dir = tmp_path / "datasets" / "my-topic"
        topic_dir.mkdir(parents=True)
        (topic_dir / "config.yaml").write_text(
            "topic_id: my-topic\n"
            "topic_name: My Topic\n"
            "search_query: test\n"
            "effect_measure: odds_ratio\n"
            "primary_outcome: response\n"
        )
        (topic_dir / "studies").mkdir()
        (topic_dir / "analysis").mkdir()

        runner = CliRunner()
        result = runner.invoke(cli, ["--base-dir", str(tmp_path), "list"])
        assert result.exit_code == 0
        assert "my-topic" in result.output
        assert "My Topic" in result.output
        assert "no analysis" in result.output

    def test_list_empty_datasets(self, tmp_path):
        (tmp_path / "datasets").mkdir()
        runner = CliRunner()
        result = runner.invoke(cli, ["--base-dir", str(tmp_path), "list"])
        assert result.exit_code == 0
        assert "No topics configured" in result.output


# ---------------------------------------------------------------------------
# CLI: version flag
# ---------------------------------------------------------------------------


class TestVersionFlag:
    def test_version_option(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "evidence-sync" in result.output
        assert "0.1.0" in result.output
