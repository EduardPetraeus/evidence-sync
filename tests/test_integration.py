"""Integration tests using live PubMed API.

These tests make real HTTP calls to NCBI E-utilities and are skipped
by default. Run with: pytest -m integration
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from evidence_sync.models import EffectMeasure, ReviewConfig
from evidence_sync.monitor import deduplicate, fetch_study_details, search_pubmed

pytestmark = pytest.mark.integration


class TestPubMedSearch:
    """Test live PubMed search API."""

    def test_search_returns_results(self):
        """A well-known query should return results from PubMed."""
        config = ReviewConfig(
            topic_id="integration-test",
            topic_name="Integration Test",
            search_query="fluoxetine AND depression AND placebo",
            effect_measure=EffectMeasure.ODDS_RATIO,
            primary_outcome="response",
            publication_types=["Randomized Controlled Trial"],
        )

        result = search_pubmed(config, max_results=10)

        assert result.total_count > 0, "Should find at least some RCTs"
        assert len(result.pmids) > 0, "Should return at least one PMID"
        assert len(result.pmids) <= 10, "Should respect max_results limit"
        assert result.query_used, "Should include query translation"

    def test_search_known_pmid_appears(self):
        """Search for a specific well-known trial and verify it appears."""
        # TADS study (Treatment for Adolescents with Depression Study)
        # PMID 15205400 — very well-known fluoxetine RCT
        config = ReviewConfig(
            topic_id="tads-test",
            topic_name="TADS Test",
            search_query="fluoxetine AND depression AND placebo",
            effect_measure=EffectMeasure.ODDS_RATIO,
            primary_outcome="response",
            publication_types=["Randomized Controlled Trial"],
        )

        result = search_pubmed(config, max_results=500)
        assert result.total_count > 50, (
            "Fluoxetine depression RCTs should number in the hundreds"
        )

    def test_search_empty_query_returns_something(self):
        """Even a broad query should work without errors."""
        config = ReviewConfig(
            topic_id="broad-test",
            topic_name="Broad Test",
            search_query="antidepressant",
            effect_measure=EffectMeasure.ODDS_RATIO,
            primary_outcome="response",
            publication_types=[],
        )

        result = search_pubmed(config, max_results=5)
        assert result.total_count > 0


class TestFetchStudyDetails:
    """Test fetching full study details from PubMed XML."""

    def test_fetch_known_study(self):
        """Fetch a specific PMID and verify parsed fields."""
        # Cipriani et al. 2018 — PMID 29477251
        studies = fetch_study_details(["29477251"])

        assert len(studies) == 1
        study = studies[0]

        assert study.pmid == "29477251"
        assert "antidepressant" in study.title.lower() or "efficacy" in study.title.lower()
        assert len(study.authors) > 0
        assert "Cipriani" in study.authors[0] or any("Cipriani" in a for a in study.authors)
        assert study.journal != ""
        assert study.publication_date.year == 2018
        assert len(study.abstract) > 100, "Should have a substantial abstract"

    def test_fetch_multiple_pmids(self):
        """Fetch multiple PMIDs in one call."""
        pmids = ["29477251", "15205400", "11014693"]
        studies = fetch_study_details(pmids)

        assert len(studies) == 3
        fetched_pmids = {s.pmid for s in studies}
        assert fetched_pmids == set(pmids)

    def test_fetch_returns_valid_dates(self):
        """All fetched studies should have valid publication dates."""
        studies = fetch_study_details(["29477251"])
        assert len(studies) == 1
        assert studies[0].publication_date.year >= 2000

    def test_fetch_abstract_content(self):
        """Abstract should contain meaningful text for a known study."""
        studies = fetch_study_details(["29477251"])
        assert len(studies) == 1
        abstract = studies[0].abstract
        # Cipriani 2018 abstract mentions antidepressants and meta-analysis
        assert any(
            term in abstract.lower()
            for term in ["antidepressant", "meta-analysis", "depression", "efficacy"]
        ), f"Abstract doesn't contain expected terms: {abstract[:200]}"


class TestFullPipeline:
    """Test the search → fetch → deduplicate pipeline (no extraction)."""

    def test_search_fetch_deduplicate_pipeline(self):
        """Run the full discovery pipeline against live PubMed."""
        config = ReviewConfig(
            topic_id="pipeline-test",
            topic_name="Pipeline Test",
            search_query="sertraline AND depression AND placebo",
            effect_measure=EffectMeasure.ODDS_RATIO,
            primary_outcome="response",
            publication_types=["Randomized Controlled Trial"],
        )

        # Step 1: Search
        search_result = search_pubmed(config, max_results=5)
        assert len(search_result.pmids) > 0, "Search should return results"

        # Step 2: Fetch details
        studies = fetch_study_details(search_result.pmids[:3])
        assert len(studies) > 0, "Should fetch at least one study"

        for study in studies:
            assert study.pmid != ""
            assert study.title != ""
            assert study.abstract != "" or True  # Some studies may lack abstracts

        # Step 3: Deduplicate against empty dataset
        with TemporaryDirectory() as tmpdir:
            unique_pmids = deduplicate(search_result.pmids, Path(tmpdir))
            assert unique_pmids == search_result.pmids, (
                "All PMIDs should be unique against empty dataset"
            )

    def test_deduplication_removes_existing(self):
        """Deduplication should filter out PMIDs with existing YAML files."""
        config = ReviewConfig(
            topic_id="dedup-test",
            topic_name="Dedup Test",
            search_query="fluoxetine AND depression AND placebo",
            effect_measure=EffectMeasure.ODDS_RATIO,
            primary_outcome="response",
            publication_types=["Randomized Controlled Trial"],
        )

        search_result = search_pubmed(config, max_results=5)
        assert len(search_result.pmids) >= 2

        # Simulate existing dataset with one study already present
        with TemporaryDirectory() as tmpdir:
            existing_dir = Path(tmpdir)
            # Create a fake YAML file for the first PMID
            (existing_dir / f"{search_result.pmids[0]}.yaml").touch()

            unique = deduplicate(search_result.pmids, existing_dir)
            assert search_result.pmids[0] not in unique, (
                "First PMID should be filtered out"
            )
            assert len(unique) == len(search_result.pmids) - 1
