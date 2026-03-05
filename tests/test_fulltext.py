"""Tests for full-text retrieval and data enrichment."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from evidence_sync.fulltext import (
    _parse_pmc_xml,
    _pmid_to_pmcid,
    enrich_study_with_fulltext,
    fetch_ctgov_data,
)
from evidence_sync.models import Study
from evidence_sync.versioning import load_study, save_study


@pytest.fixture
def basic_study() -> Study:
    return Study(
        pmid="12345678",
        title="Test Study on Depression",
        authors=["Smith J", "Doe A"],
        journal="Test Journal",
        publication_date=date(2020, 1, 1),
        abstract="A randomized controlled trial...",
    )


# --- PMC ID Conversion ---


class TestPmidToPmcid:
    def test_successful_conversion(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "records": [{"pmid": "12345678", "pmcid": "PMC7654321"}]
        }

        with patch("evidence_sync.fulltext.httpx.get", return_value=mock_response):
            result = _pmid_to_pmcid("12345678")

        assert result == "PMC7654321"

    def test_no_pmc_article(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"records": [{"pmid": "12345678"}]}

        with patch("evidence_sync.fulltext.httpx.get", return_value=mock_response):
            result = _pmid_to_pmcid("12345678")

        assert result is None

    def test_empty_records(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"records": []}

        with patch("evidence_sync.fulltext.httpx.get", return_value=mock_response):
            result = _pmid_to_pmcid("99999999")

        assert result is None

    def test_api_error_returns_none(self):
        import httpx

        with patch(
            "evidence_sync.fulltext.httpx.get",
            side_effect=httpx.HTTPError("Connection failed"),
        ):
            result = _pmid_to_pmcid("12345678")

        assert result is None


# --- PMC XML Parsing ---


class TestParsePmcXml:
    def test_parse_simple_body(self):
        xml = """<?xml version="1.0"?>
        <pmc-articleset>
          <article>
            <body>
              <sec>
                <title>Introduction</title>
                <p>This is the introduction paragraph.</p>
                <p>Second paragraph with more details.</p>
              </sec>
              <sec>
                <title>Methods</title>
                <p>We conducted a randomized trial.</p>
              </sec>
            </body>
          </article>
        </pmc-articleset>"""

        result = _parse_pmc_xml(xml)
        assert "introduction paragraph" in result
        assert "Second paragraph" in result
        assert "randomized trial" in result

    def test_parse_nested_elements(self):
        xml = """<?xml version="1.0"?>
        <article>
          <body>
            <sec>
              <p>Text with <italic>italic</italic> and <bold>bold</bold> words.</p>
            </sec>
          </body>
        </article>"""

        result = _parse_pmc_xml(xml)
        assert "italic" in result
        assert "bold" in result

    def test_parse_no_body(self):
        xml = """<?xml version="1.0"?>
        <article>
          <front><title>No Body Article</title></front>
        </article>"""

        result = _parse_pmc_xml(xml)
        assert result == ""


# --- ClinicalTrials.gov ---


class TestFetchCtgovData:
    def test_successful_fetch(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "protocolSection": {
                "designModule": {
                    "studyType": "INTERVENTIONAL",
                    "phases": ["PHASE3"],
                    "enrollmentInfo": {"count": 200},
                },
                "statusModule": {
                    "startDateStruct": {"date": "2019-01-15"},
                    "completionDateStruct": {"date": "2021-06-30"},
                },
                "outcomesModule": {
                    "primaryOutcomes": [
                        {"measure": "Change in HAM-D score"}
                    ]
                },
                "conditionsModule": {
                    "conditions": ["Major Depressive Disorder"]
                },
                "armsInterventionsModule": {
                    "interventions": [
                        {"name": "Fluoxetine"},
                        {"name": "Placebo"},
                    ]
                },
                "eligibilityModule": {
                    "eligibilityCriteria": "Ages 18-65"
                },
            }
        }

        with patch("evidence_sync.fulltext.httpx.get", return_value=mock_response):
            result = fetch_ctgov_data("NCT00000001")

        assert result is not None
        assert result["sample_size"] == 200
        assert result["primary_outcome"] == "Change in HAM-D score"
        assert result["conditions"] == ["Major Depressive Disorder"]
        assert "Fluoxetine" in result["interventions"]

    def test_api_error_returns_none(self):
        import httpx

        with patch(
            "evidence_sync.fulltext.httpx.get",
            side_effect=httpx.HTTPError("Not found"),
        ):
            result = fetch_ctgov_data("NCT99999999")

        assert result is None


# --- Study Enrichment ---


class TestEnrichStudy:
    def test_enrich_study_with_pmc(self, basic_study):
        """Study gets enriched with PMC data."""
        # Mock: PMID -> PMCID succeeds, full text fetched
        idconv_response = MagicMock()
        idconv_response.status_code = 200
        idconv_response.json.return_value = {
            "records": [{"pmid": "12345678", "pmcid": "PMC1234567"}]
        }

        efetch_response = MagicMock()
        efetch_response.status_code = 200
        efetch_response.text = """<?xml version="1.0"?>
        <article><body>
          <sec><p>Full text content here.</p></sec>
        </body></article>"""

        # Mock CTgov search to return nothing
        ctgov_response = MagicMock()
        ctgov_response.status_code = 200
        ctgov_response.json.return_value = {"studies": []}

        def mock_get(url, **kwargs):
            if "idconv" in url:
                return idconv_response
            elif "efetch" in url:
                return efetch_response
            else:
                return ctgov_response

        with patch("evidence_sync.fulltext.httpx.get", side_effect=mock_get):
            result = enrich_study_with_fulltext(basic_study)

        assert result is True
        assert basic_study.full_text_available is True
        assert basic_study.data_source == "full_text"
        assert basic_study.pmc_id == "PMC1234567"

    def test_enrich_study_no_pmc(self, basic_study):
        """No PMC article available -> graceful fallback."""
        # Mock: PMID -> PMCID fails (no records)
        idconv_response = MagicMock()
        idconv_response.status_code = 200
        idconv_response.json.return_value = {"records": []}

        ctgov_response = MagicMock()
        ctgov_response.status_code = 200
        ctgov_response.json.return_value = {"studies": []}

        with patch(
            "evidence_sync.fulltext.httpx.get",
            return_value=idconv_response,
        ) as mock_get:
            # Override for CTgov search
            mock_get.side_effect = lambda url, **kwargs: (
                idconv_response if "idconv" in url else ctgov_response
            )
            result = enrich_study_with_fulltext(basic_study)

        assert result is False
        assert basic_study.data_source == "abstract"
        assert basic_study.full_text_available is False

    def test_enrich_study_registry_only(self, basic_study):
        """Study enriched with ClinicalTrials.gov but no PMC."""
        # PMC returns nothing
        idconv_response = MagicMock()
        idconv_response.status_code = 200
        idconv_response.json.return_value = {"records": []}

        # CTgov search finds NCT ID
        ctgov_search_response = MagicMock()
        ctgov_search_response.status_code = 200
        ctgov_search_response.json.return_value = {
            "studies": [
                {
                    "protocolSection": {
                        "identificationModule": {"nctId": "NCT00123456"}
                    }
                }
            ]
        }

        # CTgov fetch returns data
        ctgov_data_response = MagicMock()
        ctgov_data_response.status_code = 200
        ctgov_data_response.json.return_value = {
            "protocolSection": {
                "designModule": {
                    "studyType": "INTERVENTIONAL",
                    "phases": ["PHASE3"],
                    "enrollmentInfo": {"count": 300},
                },
                "statusModule": {},
                "outcomesModule": {},
                "conditionsModule": {"conditions": ["Depression"]},
                "armsInterventionsModule": {"interventions": []},
                "eligibilityModule": {},
            }
        }

        call_count = 0

        def mock_get(url, **kwargs):
            nonlocal call_count
            if "idconv" in url:
                return idconv_response
            elif "clinicaltrials.gov" in url:
                call_count += 1
                if call_count == 1:
                    return ctgov_search_response
                return ctgov_data_response
            return idconv_response

        with patch("evidence_sync.fulltext.httpx.get", side_effect=mock_get):
            result = enrich_study_with_fulltext(basic_study)

        assert result is True
        assert basic_study.nct_id == "NCT00123456"
        assert basic_study.data_source == "registry"


# --- YAML Roundtrip for Provenance Fields ---


class TestProvenanceYamlRoundtrip:
    def test_provenance_fields_yaml_roundtrip(self, tmp_path):
        """Save/load with new provenance fields preserves data."""
        study = Study(
            pmid="99887766",
            title="Provenance Test Study",
            authors=["Author A"],
            journal="Test Journal",
            publication_date=date(2023, 6, 1),
            abstract="Abstract text.",
            data_source="full_text",
            pmc_id="PMC9988776",
            nct_id="NCT00112233",
            full_text_available=True,
        )

        save_study(study, tmp_path)
        loaded = load_study(tmp_path / "99887766.yaml")

        assert loaded.data_source == "full_text"
        assert loaded.pmc_id == "PMC9988776"
        assert loaded.nct_id == "NCT00112233"
        assert loaded.full_text_available is True

    def test_default_provenance_fields(self, tmp_path):
        """Studies without provenance fields get correct defaults."""
        study = Study(
            pmid="11223344",
            title="Default Provenance Study",
            authors=["Author B"],
            journal="Test Journal",
            publication_date=date(2023, 1, 1),
            abstract="Abstract.",
        )

        save_study(study, tmp_path)
        loaded = load_study(tmp_path / "11223344.yaml")

        assert loaded.data_source == "abstract"
        assert loaded.pmc_id is None
        assert loaded.nct_id is None


# --- CLI Enrich Command ---


class TestEnrichCliCommand:
    def test_enrich_cli_command(self, tmp_path):
        """CLI enrich command works with mocked APIs."""
        from click.testing import CliRunner

        from evidence_sync.cli import cli

        # Set up topic directory structure
        topic_dir = tmp_path / "datasets" / "test-topic"
        studies_dir = topic_dir / "studies"
        studies_dir.mkdir(parents=True)
        (topic_dir / "analysis").mkdir()

        # Create config
        import yaml

        config = {
            "topic_id": "test-topic",
            "topic_name": "Test Topic",
            "search_query": "test",
            "effect_measure": "odds_ratio",
            "primary_outcome": "Test outcome",
        }
        with open(topic_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)

        # Save a study
        study = Study(
            pmid="55667788",
            title="CLI Enrich Test Study",
            authors=["Author C"],
            journal="Test Journal",
            publication_date=date(2023, 6, 1),
            abstract="Test abstract.",
        )
        save_study(study, studies_dir)

        # Mock the enrichment to return False (no enrichment)
        with patch(
            "evidence_sync.fulltext.enrich_study_with_fulltext",
            return_value=False,
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["--base-dir", str(tmp_path), "enrich", "test-topic"],
            )

        assert result.exit_code == 0
        assert "0/1" in result.output
