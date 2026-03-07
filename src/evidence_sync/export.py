"""Export study data in various formats — CSV, RevMan XML, R-compatible."""

from __future__ import annotations

import csv
import io
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

from evidence_sync.models import AnalysisResult, ReviewConfig, Study

logger = logging.getLogger(__name__)


def _sanitize_csv_cell(value: str) -> str:
    """Prevent formula injection in spreadsheet applications."""
    if value and value[0] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


def _write_output(content: str, output_path: Path | None, base_dir: Path | None = None) -> str:
    """Write content to file if path given, return content either way.

    Args:
        content: The content to write.
        output_path: Optional file path to write to.
        base_dir: If provided, output_path must resolve within this directory.

    Raises:
        ValueError: If output_path escapes base_dir (path traversal attempt).
    """
    if output_path is not None:
        if base_dir is not None:
            resolved = output_path.resolve()
            base_resolved = base_dir.resolve()
            if not resolved.is_relative_to(base_resolved):
                raise ValueError(f"Output path {output_path} escapes base directory {base_dir}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Write with newline="" to preserve csv \r\n line endings
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            f.write(content)
    return content


def export_csv(
    studies: list[Study],
    result: AnalysisResult | None = None,
    output_path: Path | None = None,
) -> str:
    """Export study data as CSV.

    Args:
        studies: Studies to export.
        result: Meta-analysis result (unused, for API consistency).
        output_path: Optional file path to write CSV to.

    Returns:
        CSV string.
    """
    valid = [s for s in studies if s.has_extractable_data]

    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow(
        [
            "pmid",
            "authors",
            "year",
            "title",
            "journal",
            "effect_size",
            "ci_lower",
            "ci_upper",
            "se",
            "p_value",
            "n_treatment",
            "n_control",
            "n_total",
            "study_design",
            "effect_measure",
            "review_status",
            "population",
            "intervention",
            "comparator",
            "outcome",
        ]
    )

    for s in valid:
        authors_str = "; ".join(s.authors) if s.authors else ""
        writer.writerow(
            [
                s.pmid,
                _sanitize_csv_cell(authors_str),
                s.publication_date.year,
                _sanitize_csv_cell(s.title),
                _sanitize_csv_cell(s.journal),
                f"{s.effect_size:.6f}" if s.effect_size is not None else "",
                f"{s.ci_lower:.6f}" if s.ci_lower is not None else "",
                f"{s.ci_upper:.6f}" if s.ci_upper is not None else "",
                f"{s.se_from_ci:.6f}" if s.se_from_ci is not None else "",
                f"{s.p_value:.6f}" if s.p_value is not None else "",
                s.sample_size_treatment or "",
                s.sample_size_control or "",
                s.sample_size_total or "",
                s.study_design.value if s.study_design else "",
                s.effect_measure.value if s.effect_measure else "",
                s.review_status.value,
                _sanitize_csv_cell(s.population or ""),
                _sanitize_csv_cell(s.intervention or ""),
                _sanitize_csv_cell(s.comparator or ""),
                _sanitize_csv_cell(s.outcome or ""),
            ]
        )

    csv_str = output.getvalue()
    return _write_output(csv_str, output_path)


def export_revman_xml(
    studies: list[Study],
    config: ReviewConfig,
    result: AnalysisResult | None = None,
    output_path: Path | None = None,
) -> str:
    """Export study data in RevMan 5 XML format.

    Generates a simplified RevMan-compatible XML structure with study
    data and meta-analysis results.

    Args:
        studies: Studies to export.
        config: Review configuration.
        result: Meta-analysis result (if available).
        output_path: Optional file path to write XML to.

    Returns:
        XML string.
    """
    valid = [s for s in studies if s.has_extractable_data]

    root = ET.Element("cochrane_review")
    root.set("id", config.topic_id)

    # Review metadata
    meta = ET.SubElement(root, "review_metadata")
    ET.SubElement(meta, "title").text = config.topic_name
    ET.SubElement(meta, "search_query").text = config.search_query
    ET.SubElement(meta, "effect_measure").text = config.effect_measure.value
    ET.SubElement(meta, "primary_outcome").text = config.primary_outcome

    # Studies
    studies_elem = ET.SubElement(root, "included_studies")
    for s in valid:
        if not s.pmid or not s.pmid.strip().isdigit():
            logger.warning("Skipping study with non-numeric PMID: %r", s.pmid)
            continue
        study_elem = ET.SubElement(studies_elem, "study")
        study_elem.set("id", s.pmid.strip())

        first_author = s.authors[0] if s.authors else "Unknown"
        ET.SubElement(study_elem, "name").text = f"{first_author} {s.publication_date.year}"
        ET.SubElement(study_elem, "year").text = str(s.publication_date.year)
        ET.SubElement(study_elem, "title").text = s.title

        data_elem = ET.SubElement(study_elem, "data")
        if s.effect_size is not None:
            ET.SubElement(data_elem, "effect_size").text = f"{s.effect_size:.6f}"
        if s.ci_lower is not None:
            ET.SubElement(data_elem, "ci_lower").text = f"{s.ci_lower:.6f}"
        if s.ci_upper is not None:
            ET.SubElement(data_elem, "ci_upper").text = f"{s.ci_upper:.6f}"
        if s.se_from_ci is not None:
            ET.SubElement(data_elem, "se").text = f"{s.se_from_ci:.6f}"
        if s.sample_size_treatment is not None:
            ET.SubElement(data_elem, "n_treatment").text = str(s.sample_size_treatment)
        if s.sample_size_control is not None:
            ET.SubElement(data_elem, "n_control").text = str(s.sample_size_control)

    # Analysis result
    if result is not None:
        analysis_elem = ET.SubElement(root, "analysis")
        ET.SubElement(analysis_elem, "n_studies").text = str(result.n_studies)
        ET.SubElement(analysis_elem, "pooled_effect").text = f"{result.pooled_effect:.6f}"
        ET.SubElement(analysis_elem, "ci_lower").text = f"{result.pooled_ci_lower:.6f}"
        ET.SubElement(analysis_elem, "ci_upper").text = f"{result.pooled_ci_upper:.6f}"
        ET.SubElement(analysis_elem, "p_value").text = f"{result.pooled_p_value:.6f}"
        ET.SubElement(analysis_elem, "i_squared").text = f"{result.i_squared:.1f}"
        ET.SubElement(analysis_elem, "tau_squared").text = f"{result.tau_squared:.6f}"

    # Serialize
    ET.indent(root, space="  ")
    xml_str = ET.tostring(root, encoding="unicode", xml_declaration=True)
    return _write_output(xml_str, output_path)


def export_r_dataframe(
    studies: list[Study],
    result: AnalysisResult | None = None,
    output_path: Path | None = None,
) -> str:
    """Export as R-compatible CSV with metafor column naming conventions.

    Column names follow metafor package conventions:
    - yi: effect size
    - vi: variance (se^2)
    - sei: standard error

    Args:
        studies: Studies to export.
        result: Meta-analysis result (unused, for API consistency).
        output_path: Optional file path to write CSV to.

    Returns:
        CSV string with metafor-compatible columns.
    """
    valid = [s for s in studies if s.has_extractable_data]

    output = io.StringIO()
    writer = csv.writer(output)

    # metafor-compatible column names
    writer.writerow(
        [
            "study",
            "year",
            "yi",
            "vi",
            "sei",
            "ci.lb",
            "ci.ub",
            "ni",
            "n1i",
            "n2i",
        ]
    )

    for s in valid:
        first_author = s.authors[0].split()[-1] if s.authors else "Unknown"
        study_label = f"{first_author} {s.publication_date.year}"
        se = s.se_from_ci
        vi = se**2 if se is not None else None

        writer.writerow(
            [
                _sanitize_csv_cell(study_label),
                s.publication_date.year,
                f"{s.effect_size:.6f}" if s.effect_size is not None else "",
                f"{vi:.6f}" if vi is not None else "",
                f"{se:.6f}" if se is not None else "",
                f"{s.ci_lower:.6f}" if s.ci_lower is not None else "",
                f"{s.ci_upper:.6f}" if s.ci_upper is not None else "",
                s.sample_size_total or "",
                s.sample_size_treatment or "",
                s.sample_size_control or "",
            ]
        )

    csv_str = output.getvalue()
    return _write_output(csv_str, output_path)
