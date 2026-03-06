"""Core data models for Evidence Sync."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Optional


class StudyDesign(Enum):
    RCT = "rct"
    CROSSOVER = "crossover"
    CLUSTER_RCT = "cluster_rct"
    QUASI_EXPERIMENTAL = "quasi_experimental"
    UNKNOWN = "unknown"


class EffectMeasure(Enum):
    MEAN_DIFFERENCE = "mean_difference"
    STANDARDIZED_MEAN_DIFFERENCE = "standardized_mean_difference"
    ODDS_RATIO = "odds_ratio"
    RISK_RATIO = "risk_ratio"
    HAZARD_RATIO = "hazard_ratio"


class ReviewStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CORRECTED = "corrected"


class BiasRisk(Enum):
    LOW = "low"
    UNCLEAR = "unclear"
    HIGH = "high"


@dataclass
class RiskOfBias:
    """Cochrane Risk of Bias assessment domains."""

    random_sequence_generation: BiasRisk = BiasRisk.UNCLEAR
    allocation_concealment: BiasRisk = BiasRisk.UNCLEAR
    blinding_participants: BiasRisk = BiasRisk.UNCLEAR
    blinding_outcome: BiasRisk = BiasRisk.UNCLEAR
    incomplete_outcome: BiasRisk = BiasRisk.UNCLEAR
    selective_reporting: BiasRisk = BiasRisk.UNCLEAR

    @property
    def overall(self) -> BiasRisk:
        """Overall risk: HIGH if any domain is high, LOW if all low, else UNCLEAR."""
        domains = [
            self.random_sequence_generation,
            self.allocation_concealment,
            self.blinding_participants,
            self.blinding_outcome,
            self.incomplete_outcome,
            self.selective_reporting,
        ]
        if any(d == BiasRisk.HIGH for d in domains):
            return BiasRisk.HIGH
        if all(d == BiasRisk.LOW for d in domains):
            return BiasRisk.LOW
        return BiasRisk.UNCLEAR


@dataclass
class Study:
    """A single study extracted from PubMed."""

    pmid: str
    title: str
    authors: list[str]
    journal: str
    publication_date: date
    abstract: str

    # Extracted data (populated by extractor)
    sample_size_treatment: Optional[int] = None
    sample_size_control: Optional[int] = None
    effect_size: Optional[float] = None
    effect_measure: Optional[EffectMeasure] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    p_value: Optional[float] = None
    study_design: StudyDesign = StudyDesign.UNKNOWN
    primary_outcome: Optional[str] = None
    risk_of_bias: Optional[RiskOfBias] = None

    # Metadata
    extraction_date: Optional[date] = None
    extraction_model: Optional[str] = None
    extraction_confidence: Optional[float] = None
    full_text_available: bool = False

    # Review workflow
    review_status: ReviewStatus = ReviewStatus.PENDING
    reviewer: Optional[str] = None
    review_date: Optional[date] = None
    review_notes: Optional[str] = None

    # PICO extraction
    population: Optional[str] = None
    intervention: Optional[str] = None
    comparator: Optional[str] = None
    outcome: Optional[str] = None

    # Data provenance
    data_source: str = "abstract"  # "abstract", "full_text", "registry", "manual"
    pmc_id: Optional[str] = None
    nct_id: Optional[str] = None

    # Original values before correction (for audit trail)
    original_effect_size: Optional[float] = None
    original_ci_lower: Optional[float] = None
    original_ci_upper: Optional[float] = None

    @property
    def sample_size_total(self) -> Optional[int]:
        if self.sample_size_treatment is not None and self.sample_size_control is not None:
            return self.sample_size_treatment + self.sample_size_control
        return None

    @property
    def has_extractable_data(self) -> bool:
        return (
            self.effect_size is not None
            and self.ci_lower is not None
            and self.ci_upper is not None
        )

    @property
    def se_from_ci(self) -> Optional[float]:
        """Estimate standard error from 95% confidence interval."""
        if self.ci_lower is not None and self.ci_upper is not None:
            return (self.ci_upper - self.ci_lower) / (2 * 1.96)
        return None

    @property
    def weight_inverse_variance(self) -> Optional[float]:
        """Inverse-variance weight for meta-analysis."""
        se = self.se_from_ci
        if se is not None and se > 0:
            return 1.0 / (se**2)
        return None


@dataclass
class AnalysisResult:
    """Result of a meta-analysis run."""

    topic: str
    n_studies: int
    pooled_effect: float
    pooled_ci_lower: float
    pooled_ci_upper: float
    pooled_p_value: float
    effect_measure: EffectMeasure

    # Heterogeneity
    i_squared: float
    q_statistic: float
    q_p_value: float
    tau_squared: float

    # Publication bias
    egger_intercept: Optional[float] = None
    egger_p_value: Optional[float] = None

    # Metadata
    analysis_date: Optional[date] = None
    studies_included: list[str] = field(default_factory=list)

    @property
    def significant(self) -> bool:
        return self.pooled_p_value < 0.05

    @property
    def high_heterogeneity(self) -> bool:
        return self.i_squared > 75.0


@dataclass
class DriftResult:
    """Result of comparing two analysis snapshots."""

    topic: str
    previous_effect: float
    current_effect: float
    effect_change_pct: float
    significance_flipped: bool
    heterogeneity_change: float
    alert_triggered: bool
    alert_reasons: list[str] = field(default_factory=list)


@dataclass
class ScreeningResult:
    """Result of automated study screening."""

    pmid: str
    relevance_score: float  # 0.0-1.0
    decision: str  # "include", "exclude", "uncertain"
    reasons: list[str] = field(default_factory=list)
    screening_date: Optional[date] = None
    screening_model: Optional[str] = None


@dataclass
class ReviewConfig:
    """Configuration for a living review topic."""

    topic_id: str
    topic_name: str
    search_query: str
    effect_measure: EffectMeasure
    primary_outcome: str
    inclusion_criteria: list[str] = field(default_factory=list)
    exclusion_criteria: list[str] = field(default_factory=list)
    publication_types: list[str] = field(default_factory=lambda: ["Randomized Controlled Trial"])
    min_date: Optional[str] = None
    max_date: Optional[str] = None

    # Drift thresholds
    effect_change_threshold_pct: float = 10.0
    heterogeneity_change_threshold: float = 15.0

    # Schedule
    schedule: str = "daily"

    # Alert
    alert_webhook: Optional[str] = None
    alert_email: Optional[str] = None
