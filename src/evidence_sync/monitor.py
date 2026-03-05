"""PubMed monitoring — search for new studies and deduplicate against existing dataset."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree

import defusedxml.ElementTree as DefusedET
import httpx

from evidence_sync.models import ReviewConfig, Study

logger = logging.getLogger(__name__)

PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
RATE_LIMIT_DELAY = 0.34  # Max 3 requests/sec without API key


@dataclass
class SearchResult:
    """Result of a PubMed search."""

    pmids: list[str]
    total_count: int
    query_used: str


def search_pubmed(
    config: ReviewConfig,
    max_results: int = 500,
    api_key: Optional[str] = None,
) -> SearchResult:
    """Search PubMed for studies matching the review configuration."""
    params: dict[str, str | int] = {
        "db": "pubmed",
        "term": _build_query(config),
        "retmax": max_results,
        "retmode": "json",
        "sort": "date",
    }
    if api_key:
        params["api_key"] = api_key

    response = httpx.get(PUBMED_ESEARCH_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    esearch = data.get("esearchresult", {})
    pmids = esearch.get("idlist", [])
    total_count = int(esearch.get("count", 0))
    query_translation = esearch.get("querytranslation", str(params["term"]))

    logger.info(f"PubMed search returned {len(pmids)} of {total_count} total results")

    return SearchResult(
        pmids=pmids,
        total_count=total_count,
        query_used=query_translation,
    )


def fetch_study_details(
    pmids: list[str],
    api_key: Optional[str] = None,
    batch_size: int = 50,
) -> list[Study]:
    """Fetch full study details from PubMed for a list of PMIDs."""
    studies = []

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        params: dict[str, str] = {
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "xml",
            "rettype": "abstract",
        }
        if api_key:
            params["api_key"] = api_key

        response = httpx.get(PUBMED_EFETCH_URL, params=params, timeout=30)
        response.raise_for_status()

        batch_studies = _parse_pubmed_xml(response.text)
        studies.extend(batch_studies)

        if i + batch_size < len(pmids):
            time.sleep(RATE_LIMIT_DELAY)

    logger.info(f"Fetched details for {len(studies)} studies")
    return studies


def deduplicate(
    new_pmids: list[str],
    existing_studies_dir: Path,
) -> list[str]:
    """Remove PMIDs that already exist in the dataset."""
    existing_pmids = set()

    if existing_studies_dir.exists():
        for study_file in existing_studies_dir.glob("*.yaml"):
            existing_pmids.add(study_file.stem)

    new_unique = [pmid for pmid in new_pmids if pmid not in existing_pmids]
    logger.info(
        f"Dedup: {len(new_pmids)} total, {len(existing_pmids)} existing, {len(new_unique)} new"
    )
    return new_unique


def _build_query(config: ReviewConfig) -> str:
    """Build PubMed search query from review configuration."""
    parts = [config.search_query]

    if config.publication_types:
        pt_filter = " OR ".join(f'"{pt}"[pt]' for pt in config.publication_types)
        parts.append(f"({pt_filter})")

    if config.min_date and config.max_date:
        parts.append(f'("{config.min_date}"[dp] : "{config.max_date}"[dp])')
    elif config.min_date:
        parts.append(f'("{config.min_date}"[dp] : "3000"[dp])')

    return " AND ".join(parts)


def _parse_pubmed_xml(xml_text: str) -> list[Study]:
    """Parse PubMed eFetch XML response into Study objects."""
    studies = []
    root = DefusedET.fromstring(xml_text)

    for article_el in root.findall(".//PubmedArticle"):
        try:
            study = _parse_single_article(article_el)
            if study:
                studies.append(study)
        except Exception:
            logger.warning("Failed to parse article", exc_info=True)

    return studies


def _parse_single_article(article_el: ElementTree.Element) -> Optional[Study]:
    """Parse a single PubmedArticle XML element."""
    # PMID
    pmid_el = article_el.find(".//PMID")
    if pmid_el is None or pmid_el.text is None:
        return None
    pmid = pmid_el.text

    # Title
    title_el = article_el.find(".//ArticleTitle")
    title = title_el.text if title_el is not None and title_el.text else ""

    # Abstract
    abstract_parts = []
    for abstract_text in article_el.findall(".//AbstractText"):
        label = abstract_text.get("Label", "")
        text = abstract_text.text or ""
        if label:
            abstract_parts.append(f"{label}: {text}")
        else:
            abstract_parts.append(text)
    abstract = " ".join(abstract_parts)

    # Authors
    authors = []
    for author_el in article_el.findall(".//Author"):
        last_name = author_el.find("LastName")
        fore_name = author_el.find("ForeName")
        if last_name is not None and last_name.text:
            name = last_name.text
            if fore_name is not None and fore_name.text:
                name = f"{fore_name.text} {name}"
            authors.append(name)

    # Journal
    journal_el = article_el.find(".//Journal/Title")
    journal = journal_el.text if journal_el is not None and journal_el.text else ""

    # Publication date
    pub_date = _parse_pub_date(article_el)

    return Study(
        pmid=pmid,
        title=title,
        authors=authors,
        journal=journal,
        publication_date=pub_date,
        abstract=abstract,
    )


def _parse_pub_date(article_el: ElementTree.Element) -> date:
    """Parse publication date from PubMed XML."""
    date_el = article_el.find(".//PubDate")
    if date_el is None:
        return date(1900, 1, 1)

    year_el = date_el.find("Year")
    month_el = date_el.find("Month")
    day_el = date_el.find("Day")

    year = int(year_el.text) if year_el is not None and year_el.text else 1900

    month_text = month_el.text if month_el is not None and month_el.text else "1"
    month_map = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
    }
    if month_text in month_map:
        month = month_map[month_text]
    else:
        try:
            month = int(month_text)
        except ValueError:
            month = 1

    day = int(day_el.text) if day_el is not None and day_el.text else 1

    try:
        return date(year, month, day)
    except ValueError:
        return date(year, month, 1)
