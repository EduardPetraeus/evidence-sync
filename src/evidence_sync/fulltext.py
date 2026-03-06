"""Full-text retrieval from PubMed Central and ClinicalTrials.gov."""

from __future__ import annotations

import logging
import re
import time
from typing import Optional

import defusedxml.ElementTree as DefusedET
import httpx

from evidence_sync.models import Study

NCT_ID_PATTERN = re.compile(r"^NCT\d{1,15}$")

logger = logging.getLogger(__name__)

PMC_OA_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa.cgi"
PMC_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PMC_IDCONV_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
CTGOV_API_URL = "https://clinicaltrials.gov/api/v2/studies"

RATE_LIMIT_TIMEOUT = 30.0
RATE_LIMIT_DELAY = 0.34  # Max 3 requests/sec without NCBI API key


def fetch_pmc_fulltext(pmid: str) -> Optional[str]:
    """Fetch full text from PubMed Central for a given PMID.

    Steps:
    1. Convert PMID to PMCID using PMC ID converter API
    2. Fetch full text XML from PMC
    3. Parse and return as plain text

    Returns None if article is not in PMC or not open access.
    """
    pmcid = _pmid_to_pmcid(pmid)
    if pmcid is None:
        logger.info(f"No PMC article found for PMID {pmid}")
        return None
    return _fetch_pmc_by_pmcid(pmcid)


def _fetch_pmc_by_pmcid(pmcid: str) -> Optional[str]:
    """Fetch full text from PMC by PMCID."""
    try:
        response = httpx.get(
            PMC_EFETCH_URL,
            params={
                "db": "pmc",
                "id": pmcid,
                "retmode": "xml",
            },
            timeout=RATE_LIMIT_TIMEOUT,
        )
        response.raise_for_status()
        return _parse_pmc_xml(response.text)
    except (httpx.HTTPError, Exception) as exc:
        logger.warning(f"Failed to fetch PMC full text for {pmcid}: {exc}")
        return None


def _pmid_to_pmcid(pmid: str) -> Optional[str]:
    """Convert PMID to PMCID using NCBI ID converter."""
    try:
        response = httpx.get(
            PMC_IDCONV_URL,
            params={
                "ids": pmid,
                "format": "json",
            },
            timeout=RATE_LIMIT_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

        records = data.get("records", [])
        if not records:
            return None

        pmcid = records[0].get("pmcid")
        if pmcid:
            logger.info(f"Converted PMID {pmid} -> {pmcid}")
        return pmcid

    except (httpx.HTTPError, Exception) as exc:
        logger.warning(f"PMID to PMCID conversion failed for {pmid}: {exc}")
        return None


def _parse_pmc_xml(xml_text: str) -> str:
    """Parse PMC full-text XML to plain text.

    Extracts text from <body> section, concatenating <sec>/<p> elements.
    Uses defusedxml for XXE safety.
    """
    root = DefusedET.fromstring(xml_text)

    # PMC XML wraps articles in <pmc-articleset><article>
    # or sometimes just <article>
    body = root.find(".//body")
    if body is None:
        # Try direct article structure
        body = root.find(".//article/body")
    if body is None:
        return ""

    paragraphs: list[str] = []
    for elem in body.iter():
        if elem.tag == "p":
            # Collect text and tail text from child elements
            text_parts = [elem.text or ""]
            for child in elem:
                if child.text:
                    text_parts.append(child.text)
                if child.tail:
                    text_parts.append(child.tail)
            paragraph = "".join(text_parts).strip()
            if paragraph:
                paragraphs.append(paragraph)

    return "\n\n".join(paragraphs)


def _validate_nct_id(nct_id: str) -> str:
    """Validate NCT ID format to prevent path traversal."""
    if not NCT_ID_PATTERN.match(nct_id):
        raise ValueError(f"Invalid NCT ID '{nct_id}' — must match NCT + digits")
    return nct_id


def fetch_ctgov_data(nct_id: str) -> Optional[dict]:
    """Fetch structured study data from ClinicalTrials.gov API v2.

    Returns dict with: sample_size, primary_outcome, study_design, start_date,
    completion_date, conditions, interventions, results (if available).
    """
    try:
        _validate_nct_id(nct_id)
        response = httpx.get(
            f"{CTGOV_API_URL}/{nct_id}",
            timeout=RATE_LIMIT_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

        protocol = data.get("protocolSection", {})
        design_module = protocol.get("designModule", {})
        eligibility = protocol.get("eligibilityModule", {})
        outcomes_module = protocol.get("outcomesModule", {})
        conditions_module = protocol.get("conditionsModule", {})
        arms_module = protocol.get("armsInterventionsModule", {})
        status_module = protocol.get("statusModule", {})

        # Extract primary outcome
        primary_outcomes = outcomes_module.get("primaryOutcomes", [])
        primary_outcome = primary_outcomes[0].get("measure") if primary_outcomes else None

        # Extract interventions
        interventions = []
        for arm in arms_module.get("interventions", []):
            name = arm.get("name", "")
            if name:
                interventions.append(name)

        # Extract enrollment
        enrollment_info = design_module.get("enrollmentInfo", {})
        sample_size = enrollment_info.get("count")

        return {
            "sample_size": sample_size,
            "primary_outcome": primary_outcome,
            "study_type": design_module.get("studyType"),
            "phases": design_module.get("phases", []),
            "start_date": status_module.get("startDateStruct", {}).get("date"),
            "completion_date": (
                status_module.get("completionDateStruct", {}).get("date")
            ),
            "conditions": conditions_module.get("conditions", []),
            "interventions": interventions,
            "eligibility_criteria": eligibility.get("eligibilityCriteria"),
        }

    except (httpx.HTTPError, Exception) as exc:
        logger.warning(f"Failed to fetch ClinicalTrials.gov data for {nct_id}: {exc}")
        return None


def search_ctgov_by_pmid(pmid: str) -> Optional[str]:
    """Try to find the NCT ID for a given PMID from ClinicalTrials.gov.

    Searches ClinicalTrials.gov API with the PMID as a reference.
    """
    try:
        response = httpx.get(
            CTGOV_API_URL,
            params={
                "query.term": pmid,
                "pageSize": 1,
            },
            timeout=RATE_LIMIT_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

        studies = data.get("studies", [])
        if not studies:
            return None

        protocol = studies[0].get("protocolSection", {})
        id_module = protocol.get("identificationModule", {})
        nct_id = id_module.get("nctId")

        if nct_id:
            logger.info(f"Found NCT ID {nct_id} for PMID {pmid}")
        return nct_id

    except (httpx.HTTPError, Exception) as exc:
        logger.warning(f"ClinicalTrials.gov search failed for PMID {pmid}: {exc}")
        return None


def enrich_study_with_fulltext(study: Study) -> bool:
    """Try to enrich a study with full-text and/or registry data.

    1. Try PMC full-text
    2. Try ClinicalTrials.gov if NCT ID found
    3. Update study.full_text_available, study.data_source,
       study.pmc_id, study.nct_id

    Returns True if any enrichment was successful.
    """
    enriched = False

    # Step 1: Try PMC full-text (resolve PMCID once, reuse)
    pmcid = _pmid_to_pmcid(study.pmid)
    time.sleep(RATE_LIMIT_DELAY)
    if pmcid:
        study.pmc_id = pmcid
        full_text = _fetch_pmc_by_pmcid(pmcid)
        time.sleep(RATE_LIMIT_DELAY)
        if full_text:
            study.full_text_available = True
            study.data_source = "full_text"
            enriched = True
            logger.info(f"Enriched {study.pmid} with PMC full text")

    # Step 2: Try ClinicalTrials.gov
    if study.nct_id is None:
        nct_id = search_ctgov_by_pmid(study.pmid)
        if nct_id:
            study.nct_id = nct_id

    if study.nct_id:
        ctgov_data = fetch_ctgov_data(study.nct_id)
        if ctgov_data:
            # Enrich with registry data if we don't have full text
            if study.data_source == "abstract":
                study.data_source = "registry"
            enriched = True
            logger.info(
                f"Enriched {study.pmid} with ClinicalTrials.gov data "
                f"(NCT: {study.nct_id})"
            )

    return enriched
