"""Tests for extraction accuracy validation."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from evidence_sync.accuracy import (
    compare_extraction,
    compute_accuracy_report,
    format_accuracy_report,
    load_ground_truth,
)
from evidence_sync.cli import cli
from evidence_sync.models import EffectMeasure, Study, StudyDesign


@pytest.fixture
def ground_truth_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with ground truth YAML files."""
    gt_dir = tmp_path / "ground_truth"
    gt_dir.mkdir()

    # Study 1: complete data
    data1 = {
        "pmid": "10000001",
        "title": "Test Study One",
        "authors": ["Smith J"],
        "journal": "Test Journal",
        "publication_date": "2020-01-01",
        "abstract": "A test abstract.",
        "ground_truth": {
            "sample_size_treatment": 100,
            "sample_size_control": 100,
            "effect_size": 1.50,
            "effect_measure": "odds_ratio",
            "ci_lower": 1.10,
            "ci_upper": 2.10,
            "p_value": 0.01,
            "study_design": "rct",
            "primary_outcome": "Response rate (50% HAM-D reduction)",
        },
    }

    # Study 2: partial data (some nulls)
    data2 = {
        "pmid": "10000002",
        "title": "Test Study Two",
        "authors": ["Doe A"],
        "journal": "Another Journal",
        "publication_date": "2021-06-15",
        "abstract": "Another test abstract.",
        "ground_truth": {
            "sample_size_treatment": 80,
            "sample_size_control": 80,
            "effect_size": None,
            "effect_measure": None,
            "ci_lower": None,
            "ci_upper": None,
            "p_value": 0.35,
            "study_design": "rct",
            "primary_outcome": "Change in depression score",
        },
    }

    with open(gt_dir / "10000001.yaml", "w") as f:
        yaml.dump(data1, f, default_flow_style=False)
    with open(gt_dir / "10000002.yaml", "w") as f:
        yaml.dump(data2, f, default_flow_style=False)

    return gt_dir


def _make_study(
    pmid: str = "10000001",
    sample_size_treatment: int | None = 100,
    sample_size_control: int | None = 100,
    effect_size: float | None = 1.50,
    effect_measure: EffectMeasure | None = EffectMeasure.ODDS_RATIO,
    ci_lower: float | None = 1.10,
    ci_upper: float | None = 2.10,
    p_value: float | None = 0.01,
    study_design: StudyDesign = StudyDesign.RCT,
    primary_outcome: str | None = "Response rate (50% HAM-D reduction)",
) -> Study:
    """Helper to create a study with given extracted data."""
    return Study(
        pmid=pmid,
        title=f"Test Study {pmid}",
        authors=["Smith J"],
        journal="Test Journal",
        publication_date=date(2020, 1, 1),
        abstract="A test abstract.",
        sample_size_treatment=sample_size_treatment,
        sample_size_control=sample_size_control,
        effect_size=effect_size,
        effect_measure=effect_measure,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        study_design=study_design,
        primary_outcome=primary_outcome,
    )


class TestLoadGroundTruth:
    def test_load_ground_truth(self, ground_truth_dir: Path) -> None:
        entries = load_ground_truth(ground_truth_dir)
        assert len(entries) == 2
        assert entries[0]["pmid"] == "10000001"
        assert entries[1]["pmid"] == "10000002"
        assert "ground_truth" in entries[0]
        assert entries[0]["ground_truth"]["effect_size"] == 1.50

    def test_load_empty_directory(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        entries = load_ground_truth(empty_dir)
        assert entries == []

    def test_load_nonexistent_directory(self, tmp_path: Path) -> None:
        entries = load_ground_truth(tmp_path / "nonexistent")
        assert entries == []

    def test_load_skips_files_without_ground_truth(self, tmp_path: Path) -> None:
        gt_dir = tmp_path / "gt"
        gt_dir.mkdir()
        # File without ground_truth key
        with open(gt_dir / "bad.yaml", "w") as f:
            yaml.dump({"pmid": "99999999", "title": "No GT"}, f)
        entries = load_ground_truth(gt_dir)
        assert entries == []


class TestCompareExtractionPerfectMatch:
    def test_perfect_match(self) -> None:
        """All fields match exactly — 100% accuracy."""
        study = _make_study()
        gt = {
            "sample_size_treatment": 100,
            "sample_size_control": 100,
            "effect_size": 1.50,
            "effect_measure": "odds_ratio",
            "ci_lower": 1.10,
            "ci_upper": 2.10,
            "p_value": 0.01,
            "study_design": "rct",
            "primary_outcome": "Response rate (50% HAM-D reduction)",
        }
        result = compare_extraction(study, gt)
        assert result["accuracy"] == 1.0
        assert result["n_correct"] == result["n_total"]
        assert result["pmid"] == "10000001"
        for fr in result["field_results"].values():
            assert fr["correct"] is True


class TestCompareExtractionPartialMatch:
    def test_partial_match(self) -> None:
        """Some fields match, some don't."""
        study = _make_study(
            effect_size=2.00,  # wrong
            p_value=0.05,  # wrong (outside tolerance)
            primary_outcome="Something completely different",
        )
        gt = {
            "sample_size_treatment": 100,
            "sample_size_control": 100,
            "effect_size": 1.50,
            "effect_measure": "odds_ratio",
            "ci_lower": 1.10,
            "ci_upper": 2.10,
            "p_value": 0.01,
            "study_design": "rct",
            "primary_outcome": "Response rate (50% HAM-D reduction)",
        }
        result = compare_extraction(study, gt)
        assert result["accuracy"] < 1.0
        assert result["n_correct"] < result["n_total"]
        # effect_size should be wrong (2.00 vs 1.50 = 33% diff)
        assert result["field_results"]["effect_size"]["correct"] is False
        # sample sizes should be correct
        assert result["field_results"]["sample_size_treatment"]["correct"] is True
        assert result["field_results"]["sample_size_control"]["correct"] is True


class TestCompareExtractionNumericTolerance:
    def test_within_relative_tolerance(self) -> None:
        """Values within 5% relative tolerance pass."""
        study = _make_study(effect_size=1.53)  # 2% off from 1.50
        gt = {
            "sample_size_treatment": 100,
            "sample_size_control": 100,
            "effect_size": 1.50,
            "effect_measure": "odds_ratio",
            "ci_lower": 1.10,
            "ci_upper": 2.10,
            "p_value": 0.01,
            "study_design": "rct",
            "primary_outcome": "Response rate (50% HAM-D reduction)",
        }
        result = compare_extraction(study, gt)
        assert result["field_results"]["effect_size"]["correct"] is True

    def test_within_absolute_tolerance(self) -> None:
        """Values within 0.05 absolute tolerance pass."""
        study = _make_study(p_value=0.013)  # 0.003 off from 0.01
        gt = {
            "sample_size_treatment": 100,
            "sample_size_control": 100,
            "effect_size": 1.50,
            "effect_measure": "odds_ratio",
            "ci_lower": 1.10,
            "ci_upper": 2.10,
            "p_value": 0.01,
            "study_design": "rct",
            "primary_outcome": "Response rate (50% HAM-D reduction)",
        }
        result = compare_extraction(study, gt)
        assert result["field_results"]["p_value"]["correct"] is True

    def test_outside_tolerance(self) -> None:
        """Values outside tolerance fail."""
        study = _make_study(effect_size=2.00)  # 33% off from 1.50
        gt = {
            "sample_size_treatment": 100,
            "sample_size_control": 100,
            "effect_size": 1.50,
            "effect_measure": "odds_ratio",
            "ci_lower": 1.10,
            "ci_upper": 2.10,
            "p_value": 0.01,
            "study_design": "rct",
            "primary_outcome": "Response rate (50% HAM-D reduction)",
        }
        result = compare_extraction(study, gt)
        assert result["field_results"]["effect_size"]["correct"] is False

    def test_integer_exact_match_required(self) -> None:
        """Integer fields require exact match, not tolerance."""
        study = _make_study(sample_size_treatment=101)  # Off by 1
        gt = {
            "sample_size_treatment": 100,
            "sample_size_control": 100,
            "effect_size": 1.50,
            "effect_measure": "odds_ratio",
            "ci_lower": 1.10,
            "ci_upper": 2.10,
            "p_value": 0.01,
            "study_design": "rct",
            "primary_outcome": "Response rate (50% HAM-D reduction)",
        }
        result = compare_extraction(study, gt)
        assert result["field_results"]["sample_size_treatment"]["correct"] is False


class TestCompareExtractionNullHandling:
    def test_both_null_is_correct(self) -> None:
        """Null in both expected and actual = correct."""
        study = _make_study(effect_size=None, ci_lower=None, ci_upper=None)
        gt = {
            "sample_size_treatment": 100,
            "sample_size_control": 100,
            "effect_size": None,
            "effect_measure": "odds_ratio",
            "ci_lower": None,
            "ci_upper": None,
            "p_value": 0.01,
            "study_design": "rct",
            "primary_outcome": "Response rate (50% HAM-D reduction)",
        }
        result = compare_extraction(study, gt)
        assert result["field_results"]["effect_size"]["correct"] is True
        assert result["field_results"]["ci_lower"]["correct"] is True
        assert result["field_results"]["ci_upper"]["correct"] is True

    def test_null_mismatch_is_wrong(self) -> None:
        """One null and one non-null = wrong."""
        study = _make_study(effect_size=1.50)
        gt = {
            "sample_size_treatment": 100,
            "sample_size_control": 100,
            "effect_size": None,
            "effect_measure": "odds_ratio",
            "ci_lower": 1.10,
            "ci_upper": 2.10,
            "p_value": 0.01,
            "study_design": "rct",
            "primary_outcome": "Response rate (50% HAM-D reduction)",
        }
        result = compare_extraction(study, gt)
        assert result["field_results"]["effect_size"]["correct"] is False

    def test_expected_value_actual_null(self) -> None:
        """Expected has value but actual is null = wrong."""
        study = _make_study(effect_size=None)
        gt = {
            "sample_size_treatment": 100,
            "sample_size_control": 100,
            "effect_size": 1.50,
            "effect_measure": "odds_ratio",
            "ci_lower": 1.10,
            "ci_upper": 2.10,
            "p_value": 0.01,
            "study_design": "rct",
            "primary_outcome": "Response rate (50% HAM-D reduction)",
        }
        result = compare_extraction(study, gt)
        assert result["field_results"]["effect_size"]["correct"] is False


class TestComputeAccuracyReport:
    def test_aggregate_accuracy(self) -> None:
        """Aggregates correctly across multiple studies."""
        study1 = _make_study(pmid="10000001")
        study2 = _make_study(
            pmid="10000002",
            effect_size=2.00,  # wrong
            primary_outcome="Something else entirely",  # wrong
        )
        gt = {
            "sample_size_treatment": 100,
            "sample_size_control": 100,
            "effect_size": 1.50,
            "effect_measure": "odds_ratio",
            "ci_lower": 1.10,
            "ci_upper": 2.10,
            "p_value": 0.01,
            "study_design": "rct",
            "primary_outcome": "Response rate (50% HAM-D reduction)",
        }
        r1 = compare_extraction(study1, gt)
        r2 = compare_extraction(study2, gt)
        report = compute_accuracy_report([r1, r2])

        # study1 is 9/9, study2 has at least 2 wrong
        assert 0.0 < report["overall_accuracy"] < 1.0
        assert "per_field_accuracy" in report
        assert "numeric_error_rates" in report
        assert "problem_studies" in report
        assert "summary_text" in report

        # sample_size_treatment should be 100% (both correct)
        assert report["per_field_accuracy"]["sample_size_treatment"] == 1.0

    def test_empty_results(self) -> None:
        """Empty results produces a valid report."""
        report = compute_accuracy_report([])
        assert report["overall_accuracy"] == 0.0
        assert report["per_field_accuracy"] == {}
        assert report["problem_studies"] == []

    def test_problem_studies_detected(self) -> None:
        """Studies with < 50% accuracy appear in problem_studies."""
        # Create a study where almost everything is wrong
        study = _make_study(
            pmid="99999999",
            sample_size_treatment=999,
            sample_size_control=999,
            effect_size=99.9,
            effect_measure=EffectMeasure.RISK_RATIO,
            ci_lower=88.0,
            ci_upper=99.0,
            p_value=0.99,
            study_design=StudyDesign.CROSSOVER,
            primary_outcome="Something completely unrelated",
        )
        gt = {
            "sample_size_treatment": 100,
            "sample_size_control": 100,
            "effect_size": 1.50,
            "effect_measure": "odds_ratio",
            "ci_lower": 1.10,
            "ci_upper": 2.10,
            "p_value": 0.01,
            "study_design": "rct",
            "primary_outcome": "Response rate (50% HAM-D reduction)",
        }
        result = compare_extraction(study, gt)
        report = compute_accuracy_report([result])
        assert "99999999" in report["problem_studies"]


class TestFormatAccuracyReport:
    def test_produces_readable_markdown(self) -> None:
        """Produces readable markdown output."""
        report = {
            "overall_accuracy": 0.85,
            "per_field_accuracy": {
                "sample_size_treatment": 1.0,
                "effect_size": 0.7,
            },
            "numeric_error_rates": {
                "effect_size": 0.05,
            },
            "problem_studies": ["30012345"],
            "summary_text": "Test summary",
        }
        output = format_accuracy_report(report)
        assert "# Extraction Accuracy Report" in output
        assert "85.0%" in output
        assert "sample_size_treatment" in output
        assert "effect_size" in output
        assert "30012345" in output
        assert "## Per-Field Accuracy" in output
        assert "## Numeric Error Rates" in output
        assert "## Problem Studies" in output

    def test_no_problem_studies(self) -> None:
        """No problem studies section when none exist."""
        report = {
            "overall_accuracy": 1.0,
            "per_field_accuracy": {"effect_size": 1.0},
            "numeric_error_rates": {},
            "problem_studies": [],
            "summary_text": "",
        }
        output = format_accuracy_report(report)
        assert "Problem Studies" not in output

    def test_no_numeric_errors_section_when_empty(self) -> None:
        """No numeric errors section when dict is empty."""
        report = {
            "overall_accuracy": 1.0,
            "per_field_accuracy": {"study_design": 1.0},
            "numeric_error_rates": {},
            "problem_studies": [],
            "summary_text": "",
        }
        output = format_accuracy_report(report)
        assert "Numeric Error Rates" not in output


class TestValidateCliCommand:
    def test_validate_command_with_ground_truth(self, tmp_path: Path) -> None:
        """CLI validate command works with ground truth directory."""
        from evidence_sync.config import save_review_config
        from evidence_sync.models import ReviewConfig

        # Set up a topic
        base_dir = tmp_path / "project"
        datasets_dir = base_dir / "datasets" / "ssri-test"
        (datasets_dir / "studies").mkdir(parents=True)
        (datasets_dir / "analysis").mkdir(parents=True)

        config = ReviewConfig(
            topic_id="ssri-test",
            topic_name="SSRI Test",
            search_query="ssri depression",
            effect_measure=EffectMeasure.ODDS_RATIO,
            primary_outcome="Response rate",
        )
        save_review_config(config, datasets_dir / "config.yaml")

        # Create a small ground truth dir
        gt_dir = tmp_path / "gt"
        gt_dir.mkdir()
        gt_data = {
            "pmid": "10000001",
            "title": "Test Study",
            "authors": ["Smith J"],
            "journal": "Test Journal",
            "publication_date": "2020-01-01",
            "abstract": "A test abstract.",
            "ground_truth": {
                "sample_size_treatment": 100,
                "sample_size_control": 100,
                "effect_size": 1.50,
                "effect_measure": "odds_ratio",
                "ci_lower": 1.10,
                "ci_upper": 2.10,
                "p_value": 0.01,
                "study_design": "rct",
                "primary_outcome": "Response rate (50% HAM-D reduction)",
            },
        }
        with open(gt_dir / "10000001.yaml", "w") as f:
            yaml.dump(gt_data, f, default_flow_style=False)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--base-dir", str(base_dir),
                "validate", "ssri-test",
                "--ground-truth-dir", str(gt_dir),
            ],
        )
        assert result.exit_code == 0
        assert "10000001" in result.output
        assert "Extraction Accuracy Report" in result.output

    def test_validate_command_no_ground_truth(self, tmp_path: Path) -> None:
        """CLI validate command handles missing ground truth gracefully."""
        from evidence_sync.config import save_review_config
        from evidence_sync.models import ReviewConfig

        base_dir = tmp_path / "project"
        datasets_dir = base_dir / "datasets" / "ssri-test"
        (datasets_dir / "studies").mkdir(parents=True)
        (datasets_dir / "analysis").mkdir(parents=True)

        config = ReviewConfig(
            topic_id="ssri-test",
            topic_name="SSRI Test",
            search_query="ssri depression",
            effect_measure=EffectMeasure.ODDS_RATIO,
            primary_outcome="Response rate",
        )
        save_review_config(config, datasets_dir / "config.yaml")

        gt_dir = tmp_path / "empty_gt"
        gt_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--base-dir", str(base_dir),
                "validate", "ssri-test",
                "--ground-truth-dir", str(gt_dir),
            ],
        )
        assert result.exit_code == 0
        assert "No ground truth entries found" in result.output


class TestFuzzyStringComparison:
    def test_fuzzy_match_overlapping_terms(self) -> None:
        """Fuzzy match works when key terms overlap."""
        study = _make_study(primary_outcome="HAM-D response rate (50% reduction)")
        gt = {
            "sample_size_treatment": 100,
            "sample_size_control": 100,
            "effect_size": 1.50,
            "effect_measure": "odds_ratio",
            "ci_lower": 1.10,
            "ci_upper": 2.10,
            "p_value": 0.01,
            "study_design": "rct",
            "primary_outcome": "Response rate (50% HAM-D reduction)",
        }
        result = compare_extraction(study, gt)
        assert result["field_results"]["primary_outcome"]["correct"] is True

    def test_fuzzy_match_fails_on_unrelated(self) -> None:
        """Fuzzy match fails when terms are completely different."""
        study = _make_study(primary_outcome="Blood pressure measurement at 6 months")
        gt = {
            "sample_size_treatment": 100,
            "sample_size_control": 100,
            "effect_size": 1.50,
            "effect_measure": "odds_ratio",
            "ci_lower": 1.10,
            "ci_upper": 2.10,
            "p_value": 0.01,
            "study_design": "rct",
            "primary_outcome": "Response rate (50% HAM-D reduction)",
        }
        result = compare_extraction(study, gt)
        assert result["field_results"]["primary_outcome"]["correct"] is False


class TestRealGroundTruth:
    """Integration test: load the actual ground truth files from tests/ground_truth/."""

    def test_load_real_ground_truth_files(self) -> None:
        gt_dir = Path(__file__).parent / "ground_truth"
        if not gt_dir.exists():
            pytest.skip("Ground truth directory not found")
        entries = load_ground_truth(gt_dir)
        assert len(entries) == 25

    def test_self_validation_perfect_accuracy(self) -> None:
        """When ground truth is used as extracted data, accuracy should be 100%."""
        from evidence_sync.extractor import extract_study_data_from_dict

        gt_dir = Path(__file__).parent / "ground_truth"
        if not gt_dir.exists():
            pytest.skip("Ground truth directory not found")

        entries = load_ground_truth(gt_dir)
        results = []
        for entry in entries:
            gt_data = entry["ground_truth"]
            pub_date = entry.get("publication_date", "2020-01-01")
            if isinstance(pub_date, str):
                pub_date = date.fromisoformat(pub_date)

            study = Study(
                pmid=str(entry["pmid"]),
                title=entry.get("title", ""),
                authors=entry.get("authors", []),
                journal=entry.get("journal", ""),
                publication_date=pub_date,
                abstract=entry.get("abstract", ""),
            )
            # Apply ground truth as extracted data
            extracted = extract_study_data_from_dict(study, gt_data, model="test")
            result = compare_extraction(extracted, gt_data)
            results.append(result)

        report = compute_accuracy_report(results)
        assert report["overall_accuracy"] == 1.0
        assert len(report["problem_studies"]) == 0
