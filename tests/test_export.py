"""Tests for export module — CSV, RevMan XML, R-compatible formats."""

from __future__ import annotations

import csv
import io
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

import pytest

from evidence_sync.export import export_csv, export_r_dataframe, export_revman_xml
from evidence_sync.models import (
    AnalysisResult,
    EffectMeasure,
    ReviewConfig,
    Study,
)


def _make_study(
    pmid: str = "12345",
    effect_size: float = 0.5,
    ci_lower: float = 0.2,
    ci_upper: float = 0.8,
) -> Study:
    return Study(
        pmid=pmid,
        title=f"Study {pmid}",
        authors=["Smith J", "Doe A"],
        journal="Test Journal",
        publication_date=date(2024, 1, 1),
        abstract="Test abstract.",
        effect_size=effect_size,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        effect_measure=EffectMeasure.STANDARDIZED_MEAN_DIFFERENCE,
        sample_size_treatment=50,
        sample_size_control=50,
    )


def _make_config() -> ReviewConfig:
    return ReviewConfig(
        topic_id="ssri-depression",
        topic_name="SSRI vs Placebo for Depression",
        search_query="SSRI AND depression AND RCT",
        effect_measure=EffectMeasure.STANDARDIZED_MEAN_DIFFERENCE,
        primary_outcome="Depression severity (HAM-D)",
    )


def _make_result() -> AnalysisResult:
    return AnalysisResult(
        topic="ssri-depression",
        n_studies=2,
        pooled_effect=-0.31,
        pooled_ci_lower=-0.45,
        pooled_ci_upper=-0.17,
        pooled_p_value=0.001,
        effect_measure=EffectMeasure.STANDARDIZED_MEAN_DIFFERENCE,
        i_squared=42.0,
        q_statistic=6.9,
        q_p_value=0.14,
        tau_squared=0.02,
    )


# ---------------------------------------------------------------------------
# CSV Export
# ---------------------------------------------------------------------------


class TestCSVExport:
    def test_basic_csv(self):
        studies = [_make_study("1"), _make_study("2")]
        csv_str = export_csv(studies)

        reader = csv.DictReader(io.StringIO(csv_str))
        rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["pmid"] == "1"
        assert float(rows[0]["effect_size"]) == pytest.approx(0.5)

    def test_csv_headers(self):
        csv_str = export_csv([_make_study()])
        reader = csv.DictReader(io.StringIO(csv_str))
        headers = reader.fieldnames

        assert "pmid" in headers
        assert "effect_size" in headers
        assert "ci_lower" in headers
        assert "ci_upper" in headers
        assert "se" in headers
        assert "population" in headers

    def test_csv_empty_studies(self):
        csv_str = export_csv([])
        lines = csv_str.strip().split("\n")
        assert len(lines) == 1  # header only

    def test_csv_filters_no_data(self):
        study_no_data = Study(
            pmid="999",
            title="No data study",
            authors=[],
            journal="J",
            publication_date=date(2024, 1, 1),
            abstract="abs",
        )
        csv_str = export_csv([study_no_data, _make_study()])
        reader = csv.DictReader(io.StringIO(csv_str))
        rows = list(reader)

        assert len(rows) == 1  # only study with data

    def test_csv_write_to_file(self, tmp_path: Path):
        output = tmp_path / "export.csv"
        csv_str = export_csv([_make_study()], output_path=output)

        assert output.exists()
        # Normalize line endings for cross-platform comparison
        written = output.read_text().replace("\r\n", "\n")
        assert written == csv_str.replace("\r\n", "\n")

    def test_csv_se_calculation(self):
        study = _make_study(ci_lower=0.2, ci_upper=0.8)
        csv_str = export_csv([study])
        reader = csv.DictReader(io.StringIO(csv_str))
        row = next(reader)

        se = float(row["se"])
        expected_se = (0.8 - 0.2) / (2 * 1.96)
        assert se == pytest.approx(expected_se, abs=0.001)


# ---------------------------------------------------------------------------
# RevMan XML Export
# ---------------------------------------------------------------------------


class TestRevManXMLExport:
    def test_basic_xml(self):
        studies = [_make_study("1"), _make_study("2")]
        xml_str = export_revman_xml(studies, _make_config())

        root = ET.fromstring(xml_str)
        assert root.tag == "cochrane_review"
        assert root.get("id") == "ssri-depression"

    def test_xml_has_studies(self):
        xml_str = export_revman_xml([_make_study("1")], _make_config())
        root = ET.fromstring(xml_str)

        studies_elem = root.find("included_studies")
        assert studies_elem is not None
        study_elems = studies_elem.findall("study")
        assert len(study_elems) == 1
        assert study_elems[0].get("id") == "1"

    def test_xml_with_result(self):
        xml_str = export_revman_xml(
            [_make_study()],
            _make_config(),
            result=_make_result(),
        )
        root = ET.fromstring(xml_str)

        analysis = root.find("analysis")
        assert analysis is not None
        pooled = analysis.find("pooled_effect")
        assert pooled is not None
        assert float(pooled.text) == pytest.approx(-0.31)

    def test_xml_well_formed(self):
        xml_str = export_revman_xml(
            [_make_study("1"), _make_study("2")],
            _make_config(),
            result=_make_result(),
        )
        # Should not raise
        ET.fromstring(xml_str)

    def test_xml_empty_studies(self):
        xml_str = export_revman_xml([], _make_config())
        root = ET.fromstring(xml_str)

        studies_elem = root.find("included_studies")
        assert len(studies_elem.findall("study")) == 0

    def test_xml_write_to_file(self, tmp_path: Path):
        output = tmp_path / "export.xml"
        xml_str = export_revman_xml(
            [_make_study()],
            _make_config(),
            output_path=output,
        )

        assert output.exists()
        assert output.read_text() == xml_str


# ---------------------------------------------------------------------------
# R Dataframe Export
# ---------------------------------------------------------------------------


class TestRDataframeExport:
    def test_basic_r_export(self):
        studies = [_make_study("1"), _make_study("2")]
        csv_str = export_r_dataframe(studies)

        reader = csv.DictReader(io.StringIO(csv_str))
        rows = list(reader)

        assert len(rows) == 2

    def test_r_column_names(self):
        csv_str = export_r_dataframe([_make_study()])
        reader = csv.DictReader(io.StringIO(csv_str))
        headers = reader.fieldnames

        # metafor convention
        assert "yi" in headers
        assert "vi" in headers
        assert "sei" in headers
        assert "ci.lb" in headers
        assert "ci.ub" in headers
        assert "ni" in headers
        assert "n1i" in headers
        assert "n2i" in headers

    def test_r_variance_calculation(self):
        study = _make_study(ci_lower=0.2, ci_upper=0.8)
        csv_str = export_r_dataframe([study])
        reader = csv.DictReader(io.StringIO(csv_str))
        row = next(reader)

        se = (0.8 - 0.2) / (2 * 1.96)
        vi = se**2

        assert float(row["sei"]) == pytest.approx(se, abs=0.001)
        assert float(row["vi"]) == pytest.approx(vi, abs=0.001)

    def test_r_empty_studies(self):
        csv_str = export_r_dataframe([])
        lines = csv_str.strip().split("\n")
        assert len(lines) == 1  # header only

    def test_r_write_to_file(self, tmp_path: Path):
        output = tmp_path / "r_data.csv"
        csv_str = export_r_dataframe([_make_study()], output_path=output)

        assert output.exists()
        written = output.read_text().replace("\r\n", "\n")
        assert written == csv_str.replace("\r\n", "\n")


# ---------------------------------------------------------------------------
# Security: PMID Validation + Path Traversal
# ---------------------------------------------------------------------------


class TestExportSecurity:
    def test_revman_xml_skips_invalid_pmid(self):
        studies = [_make_study("12345"), _make_study("abc-invalid")]
        xml_str = export_revman_xml(studies, _make_config())
        root = ET.fromstring(xml_str)
        study_elems = root.find("included_studies").findall("study")

        assert len(study_elems) == 1
        assert study_elems[0].get("id") == "12345"

    def test_write_output_path_traversal(self, tmp_path: Path):
        from evidence_sync.export import _write_output

        base = tmp_path / "safe"
        base.mkdir()
        evil = base / ".." / "evil.txt"

        with pytest.raises(ValueError, match="escapes"):
            _write_output("data", evil, base_dir=base)
