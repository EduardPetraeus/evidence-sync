"""Configuration loading and management."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import yaml

from evidence_sync.models import EffectMeasure, ReviewConfig

TOPIC_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9\-]{0,63}\Z")


def validate_topic_id(topic_id: str) -> str:
    """Validate topic_id to prevent path traversal. Returns the topic_id if valid."""
    if not TOPIC_ID_PATTERN.match(topic_id):
        raise ValueError(
            f"Invalid topic_id '{topic_id}'. "
            "Must be lowercase alphanumeric with hyphens, 1-64 chars."
        )
    return topic_id


def _convert_dates(data: dict) -> dict:
    """Convert PyYAML auto-parsed date objects back to strings."""
    for key, value in data.items():
        if isinstance(value, date) and not isinstance(value, str):
            data[key] = value.isoformat()
        elif isinstance(value, dict):
            _convert_dates(value)
    return data


def load_review_config(config_path: Path) -> ReviewConfig:
    """Load a review configuration from YAML file."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    raw = _convert_dates(raw)

    effect_measure = EffectMeasure(raw.get("effect_measure", "odds_ratio"))

    return ReviewConfig(
        topic_id=raw["topic_id"],
        topic_name=raw["topic_name"],
        search_query=raw["search_query"],
        effect_measure=effect_measure,
        primary_outcome=raw.get("primary_outcome", ""),
        inclusion_criteria=raw.get("inclusion_criteria", []),
        exclusion_criteria=raw.get("exclusion_criteria", []),
        publication_types=raw.get("publication_types", ["Randomized Controlled Trial"]),
        min_date=raw.get("min_date"),
        max_date=raw.get("max_date"),
        effect_change_threshold_pct=raw.get("effect_change_threshold_pct", 10.0),
        heterogeneity_change_threshold=raw.get("heterogeneity_change_threshold", 15.0),
        schedule=raw.get("schedule", "daily"),
        alert_webhook=raw.get("alert_webhook"),
        alert_email=raw.get("alert_email"),
    )


def save_review_config(config: ReviewConfig, config_path: Path) -> None:
    """Save a review configuration to YAML file."""
    data = {
        "topic_id": config.topic_id,
        "topic_name": config.topic_name,
        "search_query": config.search_query,
        "effect_measure": config.effect_measure.value,
        "primary_outcome": config.primary_outcome,
        "inclusion_criteria": config.inclusion_criteria,
        "exclusion_criteria": config.exclusion_criteria,
        "publication_types": config.publication_types,
        "min_date": config.min_date,
        "max_date": config.max_date,
        "effect_change_threshold_pct": config.effect_change_threshold_pct,
        "heterogeneity_change_threshold": config.heterogeneity_change_threshold,
        "schedule": config.schedule,
        "alert_webhook": config.alert_webhook,
        "alert_email": config.alert_email,
    }

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def get_dataset_path(base_dir: Path, topic_id: str) -> Path:
    """Get the dataset directory for a topic."""
    validate_topic_id(topic_id)
    return base_dir / "datasets" / topic_id


def get_studies_dir(base_dir: Path, topic_id: str) -> Path:
    """Get the studies directory for a topic."""
    return get_dataset_path(base_dir, topic_id) / "studies"


def get_analysis_dir(base_dir: Path, topic_id: str) -> Path:
    """Get the analysis directory for a topic."""
    return get_dataset_path(base_dir, topic_id) / "analysis"
