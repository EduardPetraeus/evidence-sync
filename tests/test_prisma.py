"""Tests for PRISMA 2020 compliance module."""

from __future__ import annotations

from datetime import date

from evidence_sync.models import (
    AnalysisResult,
    BiasRisk,
    EffectMeasure,
    ReviewConfig,
    ReviewStatus,
    RiskOfBias,
    ScreeningResult,
    Study,
)
from evidence_sync.prisma import (
    DecisionLogEntry,
    GRADEAssessment,
    PRISMAFlowData,
    build_decision_log,
    format_checklist_text,
    format_decision_log_text,
    format_prisma_flow_dict,
    format_prisma_flow_text,
    format_sof_text,
    generate_methods_section,
    generate_prisma_checklist,
    generate_prisma_flow,
    generate_summary_of_findings,
    get_exclusion_reasons_summary,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_study(
    pmid: str = "12345",
    effect_size: float | None = 0.5,
    ci_lower: float | None = 0.2,
    ci_upper: float | None = 0.8,
    review_status: ReviewStatus = ReviewStatus.PENDING,
    reviewer: str | None = None,
    review_notes: str | None = None,
    population: str | None = None,
    nct_id: str | None = None,
    risk_of_bias: RiskOfBias | None = None,
    sample_size_treatment: int | None = 50,
    sample_size_control: int | None = 50,
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
        review_status=review_status,
        reviewer=reviewer,
        review_notes=review_notes,
        review_date=date(2024, 6, 1) if reviewer else None,
        population=population,
        nct_id=nct_id,
        risk_of_bias=risk_of_bias,
        sample_size_treatment=sample_size_treatment,
        sample_size_control=sample_size_control,
    )


def _make_screening(
    pmid: str = "12345",
    decision: str = "include",
    reasons: list[str] | None = None,
) -> ScreeningResult:
    return ScreeningResult(
        pmid=pmid,
        relevance_score=0.9 if decision == "include" else 0.1,
        decision=decision,
        reasons=reasons or [],
        screening_date=date(2024, 3, 1),
        screening_model="claude-test",
    )


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
        egger_intercept=-1.2,
        egger_p_value=0.18,
        analysis_date=date(2024, 6, 15),
        studies_included=["111", "222", "333", "444", "555"],
    )


# ---------------------------------------------------------------------------
# D1: PRISMA Flow Diagram
# ---------------------------------------------------------------------------


class TestPRISMAFlow:
    def test_flow_basic(self):
        studies = [_make_study("1"), _make_study("2"), _make_study("3")]
        screening = [
            _make_screening("1", "include"),
            _make_screening("2", "include"),
            _make_screening("3", "exclude", ["Wrong population"]),
        ]
        config = _make_config()

        flow = generate_prisma_flow(studies, screening, config)

        assert flow.records_screened == 3
        assert flow.records_excluded_screening == 1
        assert flow.screening_exclusion_reasons == {"Wrong population": 1}
        assert flow.studies_included > 0

    def test_flow_empty_inputs(self):
        flow = generate_prisma_flow([], [], _make_config())

        assert flow.records_identified == 0
        assert flow.records_screened == 0
        assert flow.records_excluded_screening == 0
        assert flow.studies_included == 0
        assert flow.studies_in_meta_analysis == 0

    def test_flow_with_duplicates(self):
        studies = [_make_study("1"), _make_study("2")]
        flow = generate_prisma_flow(
            studies,
            [],
            _make_config(),
            duplicates_removed=5,
        )

        assert flow.duplicates_removed == 5
        assert flow.records_identified == 7  # 2 studies + 5 dupes

    def test_flow_total_override(self):
        studies = [_make_study("1")]
        flow = generate_prisma_flow(
            studies,
            [],
            _make_config(),
            total_identified=100,
        )

        assert flow.records_identified == 100

    def test_flow_with_rejected_studies(self):
        studies = [
            _make_study("1", review_status=ReviewStatus.APPROVED, reviewer="dr"),
            _make_study(
                "2",
                review_status=ReviewStatus.REJECTED,
                reviewer="dr",
                review_notes="Low quality",
            ),
        ]
        flow = generate_prisma_flow(studies, [], _make_config())

        assert flow.studies_included == 1
        assert flow.reports_excluded_eligibility == 1
        assert "Low quality" in flow.eligibility_exclusion_reasons

    def test_flow_registers(self):
        studies = [
            _make_study("1", nct_id="NCT001"),
            _make_study("2"),
        ]
        flow = generate_prisma_flow(studies, [], _make_config())

        assert flow.records_from_registers == 1

    def test_format_text(self):
        flow = PRISMAFlowData(
            records_identified=100,
            records_from_databases=90,
            records_from_registers=10,
            duplicates_removed=5,
            records_screened=95,
            records_excluded_screening=20,
            screening_exclusion_reasons={"Wrong population": 15, "Not RCT": 5},
            reports_sought=75,
            reports_assessed=75,
            reports_excluded_eligibility=5,
            studies_included=70,
            studies_in_meta_analysis=65,
        )

        text = format_prisma_flow_text(flow)

        assert "PRISMA 2020 Flow Diagram" in text
        assert "Records identified from databases: 90" in text
        assert "Records excluded: 20" in text
        assert "Wrong population: 15" in text
        assert "Studies included in review: 70" in text

    def test_format_dict(self):
        flow = PRISMAFlowData(
            records_identified=50,
            records_screened=45,
            studies_included=30,
        )

        d = format_prisma_flow_dict(flow)

        assert d["identification"]["records_identified"] == 50
        assert d["screening"]["records_screened"] == 45
        assert d["included"]["studies_included"] == 30


# ---------------------------------------------------------------------------
# D3: Decision Log
# ---------------------------------------------------------------------------


class TestDecisionLog:
    def test_build_from_screening(self):
        screening = [
            _make_screening("1", "include", ["Meets criteria"]),
            _make_screening("2", "exclude", ["Wrong population"]),
        ]

        log = build_decision_log([], screening)

        assert len(log) == 2
        assert log[0].stage == "screening"
        assert log[0].decision == "include"

    def test_build_from_review(self):
        studies = [
            _make_study(
                "1",
                review_status=ReviewStatus.APPROVED,
                reviewer="dr.smith",
                review_notes="Good quality",
            ),
            _make_study(
                "2",
                review_status=ReviewStatus.REJECTED,
                reviewer="dr.smith",
                review_notes="Low sample size",
            ),
        ]

        log = build_decision_log(studies, [])

        assert len(log) == 2
        elig = [e for e in log if e.stage == "eligibility"]
        assert len(elig) == 2
        included = [e for e in elig if e.decision == "include"]
        excluded = [e for e in elig if e.decision == "exclude"]
        assert len(included) == 1
        assert len(excluded) == 1

    def test_build_combined(self):
        studies = [
            _make_study(
                "1",
                review_status=ReviewStatus.APPROVED,
                reviewer="dr",
            ),
        ]
        screening = [
            _make_screening("1", "include"),
            _make_screening("2", "exclude", ["Not relevant"]),
        ]

        log = build_decision_log(studies, screening)

        # 2 screening + 1 eligibility
        assert len(log) == 3
        stages = [e.stage for e in log]
        assert "screening" in stages
        assert "eligibility" in stages

    def test_pending_studies_excluded(self):
        studies = [_make_study("1")]  # default PENDING
        log = build_decision_log(studies, [])
        assert len(log) == 0

    def test_format_text(self):
        entries = [
            DecisionLogEntry(
                pmid="12345",
                stage="screening",
                decision="include",
                reasons=["Meets criteria"],
            ),
        ]

        text = format_decision_log_text(entries)

        assert "Decision Log" in text
        assert "12345" in text
        assert "screening" in text

    def test_format_empty(self):
        text = format_decision_log_text([])
        assert "No decisions recorded" in text

    def test_exclusion_summary(self):
        entries = [
            DecisionLogEntry(
                pmid="1",
                stage="screening",
                decision="exclude",
                reasons=["Wrong population"],
            ),
            DecisionLogEntry(
                pmid="2",
                stage="screening",
                decision="exclude",
                reasons=["Wrong population", "Not RCT"],
            ),
            DecisionLogEntry(
                pmid="3",
                stage="eligibility",
                decision="exclude",
                reasons=["Low quality"],
            ),
        ]

        summary = get_exclusion_reasons_summary(entries)

        assert "screening" in summary
        assert summary["screening"]["Wrong population"] == 2
        assert summary["screening"]["Not RCT"] == 1
        assert "eligibility" in summary
        assert summary["eligibility"]["Low quality"] == 1


# ---------------------------------------------------------------------------
# D2: Checklist
# ---------------------------------------------------------------------------


class TestPRISMAChecklist:
    def test_checklist_count(self):
        items = generate_prisma_checklist([], _make_config())
        assert len(items) == 27

    def test_checklist_basic_compliance(self):
        config = _make_config()
        items = generate_prisma_checklist(
            [_make_study("1")],
            config,
        )

        # Title should be compliant (config has topic_name)
        title_item = next(i for i in items if i.number == 1)
        assert title_item.compliant is True

        # Eligibility criteria should be compliant (config has criteria)
        elig_item = next(i for i in items if i.number == 5)
        assert elig_item.compliant is True

        # Abstract should NOT be compliant (must be written manually)
        abstract_item = next(i for i in items if i.number == 2)
        assert abstract_item.compliant is False

    def test_checklist_with_result(self):
        items = generate_prisma_checklist(
            [_make_study("1")],
            _make_config(),
            result=_make_result(),
        )

        # Results of syntheses (#20) should be compliant
        item20 = next(i for i in items if i.number == 20)
        assert item20.compliant is True
        assert "Pooled effect" in item20.evidence

    def test_checklist_with_flow(self):
        flow = PRISMAFlowData(studies_included=5)
        items = generate_prisma_checklist(
            [],
            _make_config(),
            flow=flow,
        )

        item16 = next(i for i in items if i.number == 16)
        assert item16.compliant is True

    def test_format_checklist(self):
        items = generate_prisma_checklist(
            [_make_study("1")],
            _make_config(),
        )

        text = format_checklist_text(items)

        assert "PRISMA 2020 Checklist" in text
        assert "YES" in text
        assert "NO" in text
        assert "Compliance:" in text

    def test_checklist_partial_data(self):
        config = ReviewConfig(
            topic_id="test",
            topic_name="",
            search_query="",
            effect_measure=EffectMeasure.ODDS_RATIO,
            primary_outcome="",
        )
        items = generate_prisma_checklist([], config)

        # Title not compliant without topic_name
        title = next(i for i in items if i.number == 1)
        assert title.compliant is False


# ---------------------------------------------------------------------------
# D5: Methods Section
# ---------------------------------------------------------------------------


class TestMethodsSection:
    def test_basic_generation(self):
        text = generate_methods_section(_make_config())

        assert "systematic search" in text.lower()
        assert "SSRI AND depression AND RCT" in text
        assert "DerSimonian-Laird" in text
        assert "random-effects" in text

    def test_with_criteria(self):
        config = _make_config()
        text = generate_methods_section(config)

        assert "Adults with MDD" in text
        assert "Animal studies" in text

    def test_with_result(self):
        text = generate_methods_section(
            _make_config(),
            result=_make_result(),
        )

        assert "-0.31" in text or "-0.310" in text

    def test_with_flow(self):
        flow = PRISMAFlowData(
            records_identified=100,
            records_screened=90,
            records_excluded_screening=20,
            duplicates_removed=10,
            reports_assessed=70,
            reports_excluded_eligibility=5,
            studies_included=65,
            studies_in_meta_analysis=60,
        )

        text = generate_methods_section(_make_config(), flow=flow)

        assert "100 records" in text
        assert "10 duplicates" in text

    def test_with_date_range(self):
        config = _make_config()
        config.min_date = "2020-01-01"
        config.max_date = "2024-12-31"

        text = generate_methods_section(config)

        assert "2020-01-01" in text
        assert "2024-12-31" in text


# ---------------------------------------------------------------------------
# D6: Summary of Findings
# ---------------------------------------------------------------------------


class TestSummaryOfFindings:
    def test_basic_generation(self):
        studies = [
            _make_study("1"),
            _make_study("2"),
        ]
        result = _make_result()

        sof = generate_summary_of_findings(studies, result, _make_config())

        assert sof.outcome == "Depression severity (HAM-D)"
        assert sof.n_studies == 5
        assert sof.certainty in ("high", "moderate", "low", "very_low")

    def test_certainty_high_when_no_downgrades(self):
        studies = [_make_study("1"), _make_study("2")]
        result = _make_result()
        result.i_squared = 10.0  # low heterogeneity
        result.egger_p_value = 0.5  # no pub bias

        sof = generate_summary_of_findings(studies, result, _make_config())

        # Might still be downgraded for imprecision
        assert sof.inconsistency == "not_serious"
        assert sof.publication_bias == "undetected"

    def test_certainty_downgrade_heterogeneity(self):
        studies = [_make_study("1")]
        result = _make_result()
        result.i_squared = 85.0  # high heterogeneity

        sof = generate_summary_of_findings(studies, result, _make_config())

        assert sof.inconsistency == "serious"

    def test_certainty_downgrade_pub_bias(self):
        studies = [_make_study("1")]
        result = _make_result()
        result.egger_p_value = 0.01  # significant bias

        sof = generate_summary_of_findings(studies, result, _make_config())

        assert sof.publication_bias == "detected"

    def test_rob_assessment(self):
        rob_high = RiskOfBias(
            random_sequence_generation=BiasRisk.HIGH,
            allocation_concealment=BiasRisk.HIGH,
        )
        studies = [
            _make_study("1", risk_of_bias=rob_high),
            _make_study("2", risk_of_bias=rob_high),
        ]
        result = _make_result()

        sof = generate_summary_of_findings(studies, result, _make_config())

        assert sof.risk_of_bias == "serious"

    def test_format_sof(self):
        sof = GRADEAssessment(
            outcome="Depression severity",
            n_studies=5,
            n_participants=500,
            effect_estimate="-0.31 (95% CI: -0.45 to -0.17)",
            certainty="moderate",
        )

        text = format_sof_text(sof)

        assert "Summary of Findings" in text
        assert "Depression severity" in text
        assert "MODERATE" in text
