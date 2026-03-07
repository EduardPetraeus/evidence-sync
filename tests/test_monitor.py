"""Tests for PubMed monitoring module."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import httpx
import pytest

from evidence_sync.models import EffectMeasure, ReviewConfig
from evidence_sync.monitor import (
    _build_query,
    _parse_pubmed_xml,
    deduplicate,
    fetch_study_details,
    search_pubmed,
)


@pytest.fixture
def config():
    return ReviewConfig(
        topic_id="test",
        topic_name="Test",
        search_query="fluoxetine AND depression",
        effect_measure=EffectMeasure.ODDS_RATIO,
        primary_outcome="Response rate",
        publication_types=["Randomized Controlled Trial"],
        min_date="2020",
    )


class TestBuildQuery:
    def test_basic_query(self, config):
        q = _build_query(config)
        assert "fluoxetine AND depression" in q
        assert '"Randomized Controlled Trial"[pt]' in q

    def test_date_range(self, config):
        config.max_date = "2024"
        q = _build_query(config)
        assert '"2020"[dp]' in q
        assert '"2024"[dp]' in q

    def test_min_date_only(self, config):
        q = _build_query(config)
        assert '"2020"[dp]' in q
        assert '"3000"[dp]' in q


class TestParsePubmedXml:
    def test_parse_single_article(self):
        xml = """<?xml version="1.0"?>
        <PubmedArticleSet>
            <PubmedArticle>
                <MedlineCitation>
                    <PMID>12345678</PMID>
                    <Article>
                        <Journal>
                            <Title>Test Journal</Title>
                            <JournalIssue>
                                <PubDate>
                                    <Year>2020</Year>
                                    <Month>Mar</Month>
                                    <Day>15</Day>
                                </PubDate>
                            </JournalIssue>
                        </Journal>
                        <ArticleTitle>Test Study Title</ArticleTitle>
                        <Abstract>
                            <AbstractText>This is the abstract text.</AbstractText>
                        </Abstract>
                        <AuthorList>
                            <Author>
                                <LastName>Smith</LastName>
                                <ForeName>John</ForeName>
                            </Author>
                            <Author>
                                <LastName>Doe</LastName>
                                <ForeName>Jane</ForeName>
                            </Author>
                        </AuthorList>
                    </Article>
                </MedlineCitation>
            </PubmedArticle>
        </PubmedArticleSet>"""

        studies = _parse_pubmed_xml(xml)
        assert len(studies) == 1

        s = studies[0]
        assert s.pmid == "12345678"
        assert s.title == "Test Study Title"
        assert s.abstract == "This is the abstract text."
        assert len(s.authors) == 2
        assert s.authors[0] == "John Smith"
        assert s.journal == "Test Journal"
        assert s.publication_date == date(2020, 3, 15)

    def test_parse_structured_abstract(self):
        xml = """<?xml version="1.0"?>
        <PubmedArticleSet>
            <PubmedArticle>
                <MedlineCitation>
                    <PMID>99999</PMID>
                    <Article>
                        <Journal><Title>J</Title>
                            <JournalIssue><PubDate><Year>2021</Year></PubDate></JournalIssue>
                        </Journal>
                        <ArticleTitle>Structured</ArticleTitle>
                        <Abstract>
                            <AbstractText Label="BACKGROUND">Background text.</AbstractText>
                            <AbstractText Label="METHODS">Methods text.</AbstractText>
                            <AbstractText Label="RESULTS">Results text.</AbstractText>
                        </Abstract>
                        <AuthorList>
                            <Author><LastName>Test</LastName></Author>
                        </AuthorList>
                    </Article>
                </MedlineCitation>
            </PubmedArticle>
        </PubmedArticleSet>"""

        studies = _parse_pubmed_xml(xml)
        assert len(studies) == 1
        assert "BACKGROUND: Background text." in studies[0].abstract
        assert "RESULTS: Results text." in studies[0].abstract

    def test_parse_empty_set(self):
        xml = """<?xml version="1.0"?><PubmedArticleSet></PubmedArticleSet>"""
        studies = _parse_pubmed_xml(xml)
        assert len(studies) == 0


class TestDeduplicate:
    def test_dedup_removes_existing(self, tmp_path):
        studies_dir = tmp_path / "studies"
        studies_dir.mkdir()
        (studies_dir / "111.yaml").write_text("pmid: 111")
        (studies_dir / "222.yaml").write_text("pmid: 222")

        result = deduplicate(["111", "222", "333", "444"], studies_dir)
        assert result == ["333", "444"]

    def test_dedup_empty_dir(self, tmp_path):
        studies_dir = tmp_path / "studies"
        result = deduplicate(["111", "222"], studies_dir)
        assert result == ["111", "222"]

    def test_dedup_all_existing(self, tmp_path):
        studies_dir = tmp_path / "studies"
        studies_dir.mkdir()
        (studies_dir / "111.yaml").write_text("pmid: 111")

        result = deduplicate(["111"], studies_dir)
        assert result == []


class TestSearchPubmed:
    @patch("evidence_sync.monitor.httpx.get")
    def test_search_success(self, mock_get, config):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "esearchresult": {
                "count": "150",
                "idlist": ["111", "222", "333"],
                "querytranslation": "test query",
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = search_pubmed(config, max_results=10)

        assert result.total_count == 150
        assert len(result.pmids) == 3
        assert result.pmids == ["111", "222", "333"]
        mock_get.assert_called_once()

    @patch("evidence_sync.monitor.httpx.get")
    def test_search_pubmed_http_429_raises(self, mock_get, config):
        mock_get.side_effect = httpx.HTTPStatusError(
            "429", request=MagicMock(), response=MagicMock()
        )
        with pytest.raises(RuntimeError, match="PubMed search failed"):
            search_pubmed(config)


class TestFetchStudyDetailsErrors:
    @patch("evidence_sync.monitor.httpx.get")
    def test_fetch_study_details_http_error(self, mock_get):
        mock_get.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock()
        )
        result = fetch_study_details(["111", "222"])
        assert result == []

    def test_parse_pubmed_xml_malformed(self):
        with pytest.raises(Exception):
            _parse_pubmed_xml("<not>valid<xml")

    def test_parse_pubmed_xml_empty_string(self):
        with pytest.raises(Exception):
            _parse_pubmed_xml("")
