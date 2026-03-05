"""Git-based dataset versioning — YAML serialization and audit trail."""

from __future__ import annotations

import logging
import subprocess
from datetime import date
from pathlib import Path
from typing import Optional

import yaml

from evidence_sync.models import (
    AnalysisResult,
    BiasRisk,
    EffectMeasure,
    RiskOfBias,
    Study,
    StudyDesign,
)

logger = logging.getLogger(__name__)


def _validate_pmid(pmid: str) -> str:
    """Validate PMID is numeric to prevent path traversal."""
    if not pmid.isdigit():
        raise ValueError(f"Invalid PMID '{pmid}' — must be numeric")
    return pmid


def save_study(study: Study, studies_dir: Path) -> Path:
    """Save a study to a YAML file in the studies directory."""
    _validate_pmid(study.pmid)
    studies_dir.mkdir(parents=True, exist_ok=True)
    filepath = studies_dir / f"{study.pmid}.yaml"

    data = _study_to_dict(study)
    with open(filepath, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    logger.info(f"Saved study {study.pmid} to {filepath}")
    return filepath


def load_study(filepath: Path) -> Study:
    """Load a study from a YAML file."""
    with open(filepath) as f:
        data = yaml.safe_load(f)

    return _dict_to_study(data)


def load_all_studies(studies_dir: Path) -> list[Study]:
    """Load all studies from a directory."""
    studies = []
    if not studies_dir.exists():
        return studies

    for filepath in sorted(studies_dir.glob("*.yaml")):
        try:
            study = load_study(filepath)
            studies.append(study)
        except Exception:
            logger.warning(f"Failed to load study from {filepath}", exc_info=True)

    logger.info(f"Loaded {len(studies)} studies from {studies_dir}")
    return studies


def save_analysis(result: AnalysisResult, analysis_dir: Path) -> Path:
    """Save analysis result to YAML."""
    analysis_dir.mkdir(parents=True, exist_ok=True)
    filepath = analysis_dir / "summary.yaml"

    data = {
        "topic": result.topic,
        "n_studies": result.n_studies,
        "pooled_effect": result.pooled_effect,
        "pooled_ci_lower": result.pooled_ci_lower,
        "pooled_ci_upper": result.pooled_ci_upper,
        "pooled_p_value": result.pooled_p_value,
        "effect_measure": result.effect_measure.value,
        "i_squared": result.i_squared,
        "q_statistic": result.q_statistic,
        "q_p_value": result.q_p_value,
        "tau_squared": result.tau_squared,
        "egger_intercept": result.egger_intercept,
        "egger_p_value": result.egger_p_value,
        "analysis_date": result.analysis_date.isoformat() if result.analysis_date else None,
        "studies_included": result.studies_included,
    }

    with open(filepath, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved analysis to {filepath}")
    return filepath


def load_analysis(analysis_dir: Path) -> Optional[AnalysisResult]:
    """Load the most recent analysis result."""
    filepath = analysis_dir / "summary.yaml"
    if not filepath.exists():
        return None

    with open(filepath) as f:
        data = yaml.safe_load(f)

    # Convert dates back from strings
    analysis_date = data.get("analysis_date")
    if isinstance(analysis_date, str):
        analysis_date = date.fromisoformat(analysis_date)
    elif isinstance(analysis_date, date):
        pass
    else:
        analysis_date = None

    return AnalysisResult(
        topic=data["topic"],
        n_studies=data["n_studies"],
        pooled_effect=data["pooled_effect"],
        pooled_ci_lower=data["pooled_ci_lower"],
        pooled_ci_upper=data["pooled_ci_upper"],
        pooled_p_value=data["pooled_p_value"],
        effect_measure=EffectMeasure(data["effect_measure"]),
        i_squared=data["i_squared"],
        q_statistic=data["q_statistic"],
        q_p_value=data["q_p_value"],
        tau_squared=data["tau_squared"],
        egger_intercept=data.get("egger_intercept"),
        egger_p_value=data.get("egger_p_value"),
        analysis_date=analysis_date,
        studies_included=data.get("studies_included", []),
    )


def _study_to_dict(study: Study) -> dict:
    """Convert Study dataclass to serializable dict."""
    data: dict = {
        "pmid": study.pmid,
        "title": study.title,
        "authors": study.authors,
        "journal": study.journal,
        "publication_date": study.publication_date.isoformat(),
        "abstract": study.abstract,
        "sample_size_treatment": study.sample_size_treatment,
        "sample_size_control": study.sample_size_control,
        "effect_size": study.effect_size,
        "effect_measure": study.effect_measure.value if study.effect_measure else None,
        "ci_lower": study.ci_lower,
        "ci_upper": study.ci_upper,
        "p_value": study.p_value,
        "study_design": study.study_design.value,
        "primary_outcome": study.primary_outcome,
        "extraction_date": study.extraction_date.isoformat() if study.extraction_date else None,
        "extraction_model": study.extraction_model,
        "extraction_confidence": study.extraction_confidence,
        "full_text_available": study.full_text_available,
    }

    if study.risk_of_bias:
        data["risk_of_bias"] = {
            "random_sequence_generation": study.risk_of_bias.random_sequence_generation.value,
            "allocation_concealment": study.risk_of_bias.allocation_concealment.value,
            "blinding_participants": study.risk_of_bias.blinding_participants.value,
            "blinding_outcome": study.risk_of_bias.blinding_outcome.value,
            "incomplete_outcome": study.risk_of_bias.incomplete_outcome.value,
            "selective_reporting": study.risk_of_bias.selective_reporting.value,
        }
    else:
        data["risk_of_bias"] = None

    return data


def _dict_to_study(data: dict) -> Study:
    """Convert dict back to Study dataclass."""
    # Handle date fields
    pub_date = data.get("publication_date", "1900-01-01")
    if isinstance(pub_date, str):
        pub_date = date.fromisoformat(pub_date)

    ext_date = data.get("extraction_date")
    if isinstance(ext_date, str):
        ext_date = date.fromisoformat(ext_date)
    elif not isinstance(ext_date, date):
        ext_date = None

    # Effect measure
    em_str = data.get("effect_measure")
    effect_measure = EffectMeasure(em_str) if em_str else None

    # Study design
    sd_str = data.get("study_design", "unknown")
    try:
        study_design = StudyDesign(sd_str)
    except ValueError:
        study_design = StudyDesign.UNKNOWN

    # Risk of bias
    rob_data = data.get("risk_of_bias")
    risk_of_bias = None
    if rob_data and isinstance(rob_data, dict):
        risk_of_bias = RiskOfBias(
            random_sequence_generation=BiasRisk(
                rob_data.get("random_sequence_generation", "unclear")
            ),
            allocation_concealment=BiasRisk(rob_data.get("allocation_concealment", "unclear")),
            blinding_participants=BiasRisk(rob_data.get("blinding_participants", "unclear")),
            blinding_outcome=BiasRisk(rob_data.get("blinding_outcome", "unclear")),
            incomplete_outcome=BiasRisk(rob_data.get("incomplete_outcome", "unclear")),
            selective_reporting=BiasRisk(rob_data.get("selective_reporting", "unclear")),
        )

    return Study(
        pmid=data["pmid"],
        title=data.get("title", ""),
        authors=data.get("authors", []),
        journal=data.get("journal", ""),
        publication_date=pub_date,
        abstract=data.get("abstract", ""),
        sample_size_treatment=data.get("sample_size_treatment"),
        sample_size_control=data.get("sample_size_control"),
        effect_size=data.get("effect_size"),
        effect_measure=effect_measure,
        ci_lower=data.get("ci_lower"),
        ci_upper=data.get("ci_upper"),
        p_value=data.get("p_value"),
        study_design=study_design,
        primary_outcome=data.get("primary_outcome"),
        risk_of_bias=risk_of_bias,
        extraction_date=ext_date,
        extraction_model=data.get("extraction_model"),
        extraction_confidence=data.get("extraction_confidence"),
        full_text_available=data.get("full_text_available", False),
    )


def commit_dataset_changes(base_dir: Path, topic_id: str) -> bool:
    """Commit dataset changes for a topic to git.

    Stages all files under datasets/<topic_id>/ and commits with a
    conventional commit message. Only commits if there are actual changes.
    Does NOT auto-push — that is a destructive action the user controls.

    Args:
        base_dir: The repository root directory.
        topic_id: The topic identifier.

    Returns:
        True if a commit was created, False if no changes to commit.
    """
    dataset_path = f"datasets/{topic_id}/"

    try:
        # Check if there are any changes to commit
        status_result = subprocess.run(
            ["git", "status", "--porcelain", dataset_path],
            cwd=str(base_dir),
            capture_output=True,
            text=True,
            check=True,
        )

        if not status_result.stdout.strip():
            logger.info(f"No changes to commit for {topic_id}")
            return False

        # Count studies for commit message
        studies_dir = base_dir / "datasets" / topic_id / "studies"
        n_studies = len(list(studies_dir.glob("*.yaml"))) if studies_dir.exists() else 0

        # Stage changes
        subprocess.run(
            ["git", "add", dataset_path],
            cwd=str(base_dir),
            capture_output=True,
            text=True,
            check=True,
        )

        # Commit
        commit_msg = f"data: update {topic_id} — {n_studies} studies"
        subprocess.run(
            ["git", "commit", "-m", commit_msg],
            cwd=str(base_dir),
            capture_output=True,
            text=True,
            check=True,
        )

        logger.info(f"Committed dataset changes for {topic_id}: {commit_msg}")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Git operation failed for {topic_id}: {e.stderr}")
        return False
