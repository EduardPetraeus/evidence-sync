"""Dedicated security tests — path containment, input validation, API key handling."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from evidence_sync.config import validate_topic_id
from evidence_sync.export import _write_output, export_revman_xml
from evidence_sync.models import EffectMeasure, ReviewConfig, Study


def _make_study(pmid: str = "12345", effect_size: float = 0.5) -> Study:
    return Study(
        pmid=pmid,
        title=f"Study {pmid}",
        authors=["Smith J"],
        journal="Test Journal",
        publication_date=date(2024, 1, 1),
        abstract="Test abstract.",
        effect_size=effect_size,
        ci_lower=0.2,
        ci_upper=0.8,
        effect_measure=EffectMeasure.ODDS_RATIO,
        sample_size_treatment=50,
        sample_size_control=50,
    )


def _make_config() -> ReviewConfig:
    return ReviewConfig(
        topic_id="test-topic",
        topic_name="Test",
        search_query="test",
        effect_measure=EffectMeasure.ODDS_RATIO,
        primary_outcome="outcome",
    )


# ---------------------------------------------------------------------------
# Path Containment
# ---------------------------------------------------------------------------


class TestPathContainment:
    def test_traversal_rejected(self, tmp_path: Path):
        base_dir = tmp_path / "safe"
        base_dir.mkdir()
        malicious_path = tmp_path / "safe" / ".." / "evil.txt"

        with pytest.raises(ValueError, match="escapes base directory"):
            _write_output("content", malicious_path, base_dir=base_dir)

    def test_subdir_allowed(self, tmp_path: Path):
        base_dir = tmp_path / "safe"
        base_dir.mkdir()
        valid_path = base_dir / "sub" / "output.csv"

        result = _write_output("content", valid_path, base_dir=base_dir)
        assert result == "content"
        assert valid_path.exists()

    def test_no_base_dir_allows_any_path(self, tmp_path: Path):
        path = tmp_path / "anywhere" / "file.txt"
        result = _write_output("content", path, base_dir=None)
        assert result == "content"
        assert path.exists()

    def test_none_output_path_returns_content(self):
        result = _write_output("content", None, base_dir=Path("/some/dir"))
        assert result == "content"


# ---------------------------------------------------------------------------
# Input Validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_topic_id_rejects_path_traversal(self):
        with pytest.raises(ValueError):
            validate_topic_id("../etc/passwd")

    def test_topic_id_rejects_slashes(self):
        with pytest.raises(ValueError):
            validate_topic_id("foo/bar")

    def test_topic_id_rejects_spaces(self):
        with pytest.raises(ValueError):
            validate_topic_id("foo bar")

    def test_topic_id_accepts_valid(self):
        assert validate_topic_id("ssri-depression") == "ssri-depression"
        assert validate_topic_id("topic123") == "topic123"

    def test_revman_xml_skips_non_numeric_pmid(self):
        studies = [
            _make_study("12345"),
            _make_study("not-a-number"),
            _make_study("67890"),
        ]
        xml_str = export_revman_xml(studies, _make_config())
        root = ET.fromstring(xml_str)
        study_elems = root.find("included_studies").findall("study")

        pmids = [s.get("id") for s in study_elems]
        assert "12345" in pmids
        assert "67890" in pmids
        assert "not-a-number" not in pmids

    def test_csv_formula_injection_sanitized(self):
        from evidence_sync.export import _sanitize_csv_cell

        assert _sanitize_csv_cell("=cmd()").startswith("'")
        assert _sanitize_csv_cell("+cmd()").startswith("'")
        assert _sanitize_csv_cell("-cmd()").startswith("'")
        assert _sanitize_csv_cell("@cmd()").startswith("'")
        assert _sanitize_csv_cell("normal") == "normal"


# ---------------------------------------------------------------------------
# API Key Security
# ---------------------------------------------------------------------------


class TestAPIKeySecurity:
    def test_gemini_key_in_header_not_url(self):
        """Verify extract_study_data_gemini sends API key via header, not URL params."""
        from evidence_sync.extractor import extract_study_data_gemini

        study = _make_study()
        study.abstract = "A randomized trial comparing X vs Y."

        with patch("evidence_sync.extractor.httpx.post") as mock_post:
            mock_post.side_effect = Exception("test abort")
            extract_study_data_gemini(study, "test-secret-key")

            assert mock_post.called, "httpx.post was not called"
            call_kwargs = mock_post.call_args
            # Key must be in headers
            headers = call_kwargs.kwargs.get("headers", {})
            assert headers.get("x-goog-api-key") == "test-secret-key"
            # Key must NOT be in URL params
            params = call_kwargs.kwargs.get("params", {})
            assert "key" not in params

    def test_error_message_does_not_leak_key(self, caplog):
        """Verify that Gemini errors don't include the API key in logs."""
        import logging

        from evidence_sync.extractor import extract_study_data_gemini

        study = _make_study()
        study.abstract = "Abstract text."
        secret = "super-secret-api-key-12345"

        with (
            patch("evidence_sync.extractor.httpx.post") as mock_post,
            caplog.at_level(logging.ERROR),
        ):
            mock_post.side_effect = Exception("connection failed")
            extract_study_data_gemini(study, secret)

        for record in caplog.records:
            assert secret not in record.getMessage()


# ---------------------------------------------------------------------------
# Max Results Bound
# ---------------------------------------------------------------------------


class TestMaxResultsBound:
    def test_max_results_rejects_over_10000(self):
        from click.testing import CliRunner

        from evidence_sync.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["search", "test-topic", "--max-results", "20000"])
        assert result.exit_code != 0

    def test_max_results_rejects_zero(self):
        from click.testing import CliRunner

        from evidence_sync.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["search", "test-topic", "--max-results", "0"])
        assert result.exit_code != 0

    def test_max_results_accepts_valid(self):
        from click.testing import CliRunner

        from evidence_sync.cli import cli

        runner = CliRunner()
        # Will fail on missing topic, but should pass validation
        result = runner.invoke(
            cli, ["search", "test-topic", "--max-results", "500", "--base-dir", "/tmp"]
        )
        # Should not fail on max-results validation
        assert "Invalid value" not in (result.output or "")
