"""Tests for configuration loading and saving."""

from __future__ import annotations

from evidence_sync.config import load_review_config, save_review_config
from evidence_sync.models import EffectMeasure, ReviewConfig


class TestConfigRoundtrip:
    def test_save_and_load(self, tmp_path):
        config = ReviewConfig(
            topic_id="test-topic",
            topic_name="Test Topic",
            search_query="fluoxetine AND depression",
            effect_measure=EffectMeasure.ODDS_RATIO,
            primary_outcome="Response rate",
            inclusion_criteria=["RCT", "Adults"],
            exclusion_criteria=["Pediatric"],
            publication_types=["Randomized Controlled Trial"],
            min_date="2020",
            effect_change_threshold_pct=15.0,
        )

        config_path = tmp_path / "config.yaml"
        save_review_config(config, config_path)
        loaded = load_review_config(config_path)

        assert loaded.topic_id == "test-topic"
        assert loaded.topic_name == "Test Topic"
        assert loaded.effect_measure == EffectMeasure.ODDS_RATIO
        assert loaded.primary_outcome == "Response rate"
        assert loaded.inclusion_criteria == ["RCT", "Adults"]
        assert loaded.min_date == "2020"
        assert loaded.effect_change_threshold_pct == 15.0

    def test_load_with_defaults(self, tmp_path):
        config_path = tmp_path / "minimal.yaml"
        config_path.write_text(
            "topic_id: minimal\n"
            "topic_name: Minimal\n"
            "search_query: test\n"
            "primary_outcome: test\n"
        )
        loaded = load_review_config(config_path)
        assert loaded.topic_id == "minimal"
        assert loaded.effect_measure == EffectMeasure.ODDS_RATIO  # default
        assert loaded.schedule == "daily"
