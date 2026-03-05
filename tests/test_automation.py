"""Tests for Phase 3 automation: webhook alerts and git auto-commit."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from evidence_sync.drift import send_alert
from evidence_sync.models import DriftResult, EffectMeasure, ReviewConfig
from evidence_sync.versioning import commit_dataset_changes


@pytest.fixture
def drift_result() -> DriftResult:
    """A sample drift result with triggered alerts."""
    return DriftResult(
        topic="ssri-depression",
        previous_effect=1.50,
        current_effect=1.80,
        effect_change_pct=20.0,
        significance_flipped=False,
        heterogeneity_change=5.0,
        alert_triggered=True,
        alert_reasons=["Effect size changed by 20.0% (threshold: 10.0%)"],
    )


@pytest.fixture
def config_with_webhook() -> ReviewConfig:
    """Review config with an https webhook URL."""
    return ReviewConfig(
        topic_id="ssri-depression",
        topic_name="SSRIs vs. Placebo for MDD",
        search_query="SSRI depression",
        effect_measure=EffectMeasure.ODDS_RATIO,
        primary_outcome="Response rate",
        alert_webhook="https://hooks.example.com/alert",
    )


@pytest.fixture
def config_with_http_webhook() -> ReviewConfig:
    """Review config with an insecure http webhook URL."""
    return ReviewConfig(
        topic_id="ssri-depression",
        topic_name="SSRIs vs. Placebo for MDD",
        search_query="SSRI depression",
        effect_measure=EffectMeasure.ODDS_RATIO,
        primary_outcome="Response rate",
        alert_webhook="http://internal-service.local/webhook",
    )


class TestSendAlert:
    def test_webhook_sends_correct_payload(self, drift_result, config_with_webhook):
        """Verify the webhook POST sends the expected JSON payload structure."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx, "post", return_value=mock_response) as mock_post:
            result = send_alert(drift_result, config_with_webhook)

        assert result is True
        mock_post.assert_called_once()

        call_kwargs = mock_post.call_args
        assert call_kwargs[0][0] == "https://hooks.example.com/alert"

        payload = call_kwargs[1]["json"]
        assert payload["topic"] == "ssri-depression"
        assert payload["previous_effect"] == 1.50
        assert payload["current_effect"] == 1.80
        assert payload["change_pct"] == 20.0
        assert payload["reasons"] == ["Effect size changed by 20.0% (threshold: 10.0%)"]
        assert "timestamp" in payload

    def test_webhook_rejects_http_url(self, drift_result, config_with_http_webhook):
        """Webhook must reject non-https URLs to prevent SSRF."""
        with patch.object(httpx, "post") as mock_post:
            result = send_alert(drift_result, config_with_http_webhook)

        assert result is False
        mock_post.assert_not_called()

    def test_webhook_handles_network_error(self, drift_result, config_with_webhook):
        """Network errors should be logged, not raised."""
        with patch.object(httpx, "post", side_effect=httpx.ConnectError("timeout")):
            result = send_alert(drift_result, config_with_webhook)

        assert result is False

    def test_no_webhook_configured(self, drift_result):
        """No alert sent when webhook is not configured."""
        config = ReviewConfig(
            topic_id="test",
            topic_name="Test",
            search_query="test",
            effect_measure=EffectMeasure.ODDS_RATIO,
            primary_outcome="test",
        )

        with patch.object(httpx, "post") as mock_post:
            result = send_alert(drift_result, config)

        assert result is False
        mock_post.assert_not_called()

    def test_email_alert_logs_warning(self, drift_result):
        """Email alert should log a warning (not implemented)."""
        config = ReviewConfig(
            topic_id="test",
            topic_name="Test",
            search_query="test",
            effect_measure=EffectMeasure.ODDS_RATIO,
            primary_outcome="test",
            alert_email="user@example.com",
        )

        with patch("evidence_sync.drift.logger") as mock_logger:
            send_alert(drift_result, config)

        mock_logger.warning.assert_called_once()
        assert "user@example.com" in mock_logger.warning.call_args[0][0]


def _init_git_repo(path: Path) -> None:
    """Initialize a bare git repo at the given path."""
    subprocess.run(["git", "init"], cwd=str(path), capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=str(path),
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=str(path),
        capture_output=True,
        check=True,
    )
    # Create initial commit so HEAD exists
    (path / ".gitkeep").touch()
    subprocess.run(["git", "add", "."], cwd=str(path), capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=str(path),
        capture_output=True,
        check=True,
    )


class TestCommitDatasetChanges:
    def test_commits_when_changes_exist(self, tmp_path):
        """Commit should be created when dataset files are modified."""
        _init_git_repo(tmp_path)

        # Create a study file
        studies_dir = tmp_path / "datasets" / "test-topic" / "studies"
        studies_dir.mkdir(parents=True)
        (studies_dir / "12345.yaml").write_text("pmid: '12345'\ntitle: Test\n")

        result = commit_dataset_changes(tmp_path, "test-topic")

        assert result is True

        # Verify commit was actually created
        log_result = subprocess.run(
            ["git", "log", "--oneline", "-1"],
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
        )
        assert "data: update test-topic" in log_result.stdout

    def test_no_commit_when_no_changes(self, tmp_path):
        """No commit should be created when there are no changes."""
        _init_git_repo(tmp_path)

        # Create and commit dataset directory first
        studies_dir = tmp_path / "datasets" / "test-topic" / "studies"
        studies_dir.mkdir(parents=True)
        (studies_dir / "12345.yaml").write_text("pmid: '12345'\ntitle: Test\n")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "add data"],
            cwd=str(tmp_path),
            capture_output=True,
            check=True,
        )

        # Now try to commit again with no changes
        result = commit_dataset_changes(tmp_path, "test-topic")

        assert result is False

    def test_commit_message_includes_study_count(self, tmp_path):
        """Commit message should include the number of studies."""
        _init_git_repo(tmp_path)

        studies_dir = tmp_path / "datasets" / "my-topic" / "studies"
        studies_dir.mkdir(parents=True)
        for i in range(3):
            (studies_dir / f"{i}.yaml").write_text(f"pmid: '{i}'\n")

        commit_dataset_changes(tmp_path, "my-topic")

        log_result = subprocess.run(
            ["git", "log", "--oneline", "-1"],
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
        )
        assert "3 studies" in log_result.stdout

    def test_handles_non_git_directory(self, tmp_path):
        """Should return False gracefully for non-git directories."""
        studies_dir = tmp_path / "datasets" / "test-topic" / "studies"
        studies_dir.mkdir(parents=True)
        (studies_dir / "12345.yaml").write_text("pmid: '12345'\n")

        result = commit_dataset_changes(tmp_path, "test-topic")

        assert result is False
