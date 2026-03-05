"""Tests for YAML versioning and serialization."""

from __future__ import annotations

from datetime import date

from evidence_sync.models import (
    AnalysisResult,
    BiasRisk,
    EffectMeasure,
    RiskOfBias,
)
from evidence_sync.versioning import (
    load_all_studies,
    load_analysis,
    load_study,
    save_analysis,
    save_study,
)
from tests.conftest import make_study


class TestStudySerialization:
    def test_save_and_load_basic(self, tmp_path):
        study = make_study(
            pmid="12345",
            effect_size=1.5,
            ci_lower=1.1,
            ci_upper=2.0,
            effect_measure=EffectMeasure.ODDS_RATIO,
            sample_size_treatment=100,
            sample_size_control=100,
        )

        save_study(study, tmp_path)
        loaded = load_study(tmp_path / "12345.yaml")

        assert loaded.pmid == "12345"
        assert loaded.effect_size == 1.5
        assert loaded.ci_lower == 1.1
        assert loaded.ci_upper == 2.0
        assert loaded.effect_measure == EffectMeasure.ODDS_RATIO
        assert loaded.sample_size_treatment == 100

    def test_save_and_load_with_risk_of_bias(self, tmp_path):
        rob = RiskOfBias(
            random_sequence_generation=BiasRisk.LOW,
            allocation_concealment=BiasRisk.HIGH,
            blinding_participants=BiasRisk.LOW,
            blinding_outcome=BiasRisk.UNCLEAR,
            incomplete_outcome=BiasRisk.LOW,
            selective_reporting=BiasRisk.LOW,
        )
        study = make_study(pmid="99999", risk_of_bias=rob)

        save_study(study, tmp_path)
        loaded = load_study(tmp_path / "99999.yaml")

        assert loaded.risk_of_bias is not None
        assert loaded.risk_of_bias.random_sequence_generation == BiasRisk.LOW
        assert loaded.risk_of_bias.allocation_concealment == BiasRisk.HIGH

    def test_save_and_load_with_none_fields(self, tmp_path):
        study = make_study(pmid="00001")

        save_study(study, tmp_path)
        loaded = load_study(tmp_path / "00001.yaml")

        assert loaded.effect_size is None
        assert loaded.ci_lower is None
        assert loaded.risk_of_bias is None

    def test_roundtrip_preserves_dates(self, tmp_path):
        study = make_study(
            pmid="55555",
            extraction_date=date(2025, 6, 15),
            publication_date=date(2020, 3, 1),
        )

        save_study(study, tmp_path)
        loaded = load_study(tmp_path / "55555.yaml")

        assert loaded.publication_date == date(2020, 3, 1)
        assert loaded.extraction_date == date(2025, 6, 15)


class TestLoadAllStudies:
    def test_load_multiple(self, tmp_path):
        for i in range(5):
            save_study(make_study(pmid=str(i)), tmp_path)

        studies = load_all_studies(tmp_path)
        assert len(studies) == 5

    def test_load_empty_dir(self, tmp_path):
        studies = load_all_studies(tmp_path)
        assert studies == []

    def test_load_nonexistent_dir(self, tmp_path):
        studies = load_all_studies(tmp_path / "nonexistent")
        assert studies == []


class TestAnalysisSerialization:
    def test_save_and_load_analysis(self, tmp_path):
        result = AnalysisResult(
            topic="test",
            n_studies=5,
            pooled_effect=1.65,
            pooled_ci_lower=1.35,
            pooled_ci_upper=2.01,
            pooled_p_value=0.0001,
            effect_measure=EffectMeasure.ODDS_RATIO,
            i_squared=25.3,
            q_statistic=5.34,
            q_p_value=0.254,
            tau_squared=0.012,
            egger_intercept=0.45,
            egger_p_value=0.32,
            analysis_date=date(2025, 7, 1),
            studies_included=["10001", "10002", "10003"],
        )

        save_analysis(result, tmp_path)
        loaded = load_analysis(tmp_path)

        assert loaded is not None
        assert loaded.topic == "test"
        assert loaded.n_studies == 5
        assert abs(loaded.pooled_effect - 1.65) < 0.001
        assert loaded.effect_measure == EffectMeasure.ODDS_RATIO
        assert abs(loaded.i_squared - 25.3) < 0.1
        assert loaded.studies_included == ["10001", "10002", "10003"]

    def test_load_nonexistent_analysis(self, tmp_path):
        result = load_analysis(tmp_path)
        assert result is None
