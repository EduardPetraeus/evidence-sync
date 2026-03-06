"""Streamlit dashboard for Evidence Sync — interactive meta-analysis visualization."""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from evidence_sync.config import load_review_config
from evidence_sync.models import AnalysisResult, BiasRisk, RiskOfBias, Study
from evidence_sync.statistics import compute_study_weights
from evidence_sync.versioning import load_all_studies, load_analysis

# ---------------------------------------------------------------------------
# Plotly figure builders (exported for testing)
# ---------------------------------------------------------------------------


def build_forest_plot(
    studies: list[Study],
    result: AnalysisResult,
    title: str = "Forest Plot",
) -> go.Figure:
    """Build an interactive forest plot using Plotly.

    Args:
        studies: Studies with extracted data.
        result: Meta-analysis result.
        title: Plot title.

    Returns:
        A plotly Figure object.
    """
    valid = [s for s in studies if s.has_extractable_data and s.se_from_ci]
    if not valid:
        fig = go.Figure()
        fig.add_annotation(text="No valid studies to display", showarrow=False)
        return fig

    weights = compute_study_weights(valid, result.tau_squared)
    weight_map = {s.pmid: w for s, w in weights}

    valid.sort(key=lambda s: s.effect_size or 0)

    labels: list[str] = []
    effects: list[float] = []
    ci_lowers: list[float] = []
    ci_uppers: list[float] = []
    sizes: list[float] = []
    hover_texts: list[str] = []

    for study in valid:
        first_author = study.authors[0].split()[-1] if study.authors else "Unknown"
        year = study.publication_date.year
        label = f"{first_author} {year}"
        labels.append(label)
        effects.append(study.effect_size)
        ci_lowers.append(study.ci_lower)
        ci_uppers.append(study.ci_upper)
        w = weight_map.get(study.pmid, 0)
        sizes.append(max(6, w * 0.8))
        hover_texts.append(
            f"PMID: {study.pmid}<br>"
            f"Effect: {study.effect_size:.3f} "
            f"[{study.ci_lower:.3f}, {study.ci_upper:.3f}]<br>"
            f"Weight: {w:.1f}%"
        )

    # Add pooled estimate
    labels.append("Pooled (RE)")
    effects.append(result.pooled_effect)
    ci_lowers.append(result.pooled_ci_lower)
    ci_uppers.append(result.pooled_ci_upper)
    sizes.append(14)
    hover_texts.append(
        f"Pooled effect: {result.pooled_effect:.3f} "
        f"[{result.pooled_ci_lower:.3f}, {result.pooled_ci_upper:.3f}]<br>"
        f"p = {result.pooled_p_value:.4f}"
    )

    fig = go.Figure()

    # Error bars for CIs
    fig.add_trace(
        go.Scatter(
            x=effects,
            y=labels,
            mode="markers",
            marker=dict(
                size=sizes,
                color=["steelblue"] * (len(labels) - 1) + ["firebrick"],
                symbol=["square"] * (len(labels) - 1) + ["diamond"],
            ),
            error_x=dict(
                type="data",
                symmetric=False,
                array=[u - e for e, u in zip(effects, ci_uppers)],
                arrayminus=[e - lo for e, lo in zip(effects, ci_lowers)],
                color="gray",
                thickness=1.5,
            ),
            text=hover_texts,
            hoverinfo="text",
        )
    )

    # Null effect line
    null_effect = 0.0 if "difference" in result.effect_measure.value else 1.0
    fig.add_vline(x=null_effect, line_dash="dash", line_color="gray", opacity=0.6)

    het_text = (
        f"I\u00b2 = {result.i_squared:.1f}%, "
        f"\u03c4\u00b2 = {result.tau_squared:.4f}, "
        f"Q = {result.q_statistic:.2f} (p = {result.q_p_value:.3f})"
    )

    fig.update_layout(
        title=title,
        xaxis_title=f"Effect Size ({result.effect_measure.value.replace('_', ' ').title()})",
        yaxis_title="",
        showlegend=False,
        height=max(400, len(labels) * 40 + 100),
        annotations=[
            dict(
                x=0.5,
                y=-0.12,
                xref="paper",
                yref="paper",
                text=het_text,
                showarrow=False,
                font=dict(size=10, color="gray"),
            )
        ],
    )

    return fig


def build_funnel_plot(
    studies: list[Study],
    result: AnalysisResult,
) -> go.Figure:
    """Build an interactive funnel plot using Plotly.

    Args:
        studies: Studies with extracted data.
        result: Meta-analysis result.

    Returns:
        A plotly Figure object.
    """
    valid = [s for s in studies if s.has_extractable_data and s.se_from_ci]
    if not valid:
        fig = go.Figure()
        fig.add_annotation(text="No valid studies to display", showarrow=False)
        return fig

    effect_vals = [s.effect_size for s in valid]
    se_vals = [s.se_from_ci for s in valid]
    hover_labels = []
    for s in valid:
        first_author = s.authors[0].split()[-1] if s.authors else "Unknown"
        hover_labels.append(
            f"PMID: {s.pmid}<br>{first_author} {s.publication_date.year}<br>"
            f"Effect: {s.effect_size:.3f}, SE: {s.se_from_ci:.3f}"
        )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=effect_vals,
            y=se_vals,
            mode="markers",
            marker=dict(size=8, color="steelblue"),
            text=hover_labels,
            hoverinfo="text",
            name="Studies",
        )
    )

    # Pooled effect line
    fig.add_vline(
        x=result.pooled_effect,
        line_color="firebrick",
        line_width=1,
    )

    # Pseudo 95% CI funnel boundaries
    import numpy as np

    max_se = max(se_vals) * 1.1 if se_vals else 1.0
    se_range = np.linspace(0, max_se, 100)
    ci_lo = result.pooled_effect - 1.96 * se_range
    ci_hi = result.pooled_effect + 1.96 * se_range

    fig.add_trace(
        go.Scatter(
            x=ci_lo.tolist(),
            y=se_range.tolist(),
            mode="lines",
            line=dict(dash="dash", color="gray", width=1),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ci_hi.tolist(),
            y=se_range.tolist(),
            mode="lines",
            line=dict(dash="dash", color="gray", width=1),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Egger's test annotation
    egger_text = ""
    if result.egger_p_value is not None:
        egger_text = (
            f"Egger's test: intercept = {result.egger_intercept:.3f}, "
            f"p = {result.egger_p_value:.3f}"
        )

    fig.update_layout(
        title="Funnel Plot",
        xaxis_title="Effect Size",
        yaxis_title="Standard Error",
        yaxis=dict(autorange="reversed"),
        showlegend=False,
        height=500,
        annotations=[
            dict(
                x=0.02,
                y=0.02,
                xref="paper",
                yref="paper",
                text=egger_text,
                showarrow=False,
                font=dict(size=10, color="gray"),
            )
        ]
        if egger_text
        else [],
    )

    return fig


def build_evidence_timeline(
    studies: list[Study],
    result: AnalysisResult,
) -> go.Figure:
    """Build an evidence timeline showing when studies were published.

    Args:
        studies: Studies with extracted data.
        result: Meta-analysis result (for current pooled effect line).

    Returns:
        A plotly Figure object.
    """
    valid = [s for s in studies if s.has_extractable_data]
    if not valid:
        fig = go.Figure()
        fig.add_annotation(text="No valid studies to display", showarrow=False)
        return fig

    valid.sort(key=lambda s: s.publication_date)

    dates = [s.publication_date.isoformat() for s in valid]
    effects = [s.effect_size for s in valid]
    labels = []
    for s in valid:
        first_author = s.authors[0].split()[-1] if s.authors else "Unknown"
        labels.append(
            f"PMID: {s.pmid}<br>{first_author} {s.publication_date.year}<br>"
            f"Effect: {s.effect_size:.3f}"
        )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=effects,
            mode="markers+lines",
            marker=dict(size=8, color="steelblue"),
            line=dict(color="steelblue", width=1, dash="dot"),
            text=labels,
            hoverinfo="text",
            name="Individual studies",
        )
    )

    # Current pooled effect line
    fig.add_hline(
        y=result.pooled_effect,
        line_dash="solid",
        line_color="firebrick",
        annotation_text=f"Pooled: {result.pooled_effect:.3f}",
        annotation_position="top right",
    )

    # Pooled CI band
    fig.add_hrect(
        y0=result.pooled_ci_lower,
        y1=result.pooled_ci_upper,
        fillcolor="firebrick",
        opacity=0.1,
        line_width=0,
    )

    # Analysis date marker (use add_shape to avoid plotly annotation bug with date strings)
    if result.analysis_date:
        date_str = result.analysis_date.isoformat()
        fig.add_shape(
            type="line",
            x0=date_str,
            x1=date_str,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(dash="dash", color="green", width=1),
        )
        fig.add_annotation(
            x=date_str,
            y=1.05,
            yref="paper",
            text="Analysis date",
            showarrow=False,
            font=dict(size=10, color="green"),
        )

    fig.update_layout(
        title="Evidence Timeline",
        xaxis_title="Publication Date",
        yaxis_title="Effect Size",
        showlegend=True,
        height=450,
    )

    return fig


def build_rob_heatmap(
    studies: list[Study],
) -> go.Figure:
    """Build a risk of bias heatmap across studies.

    Args:
        studies: Studies with risk_of_bias data.

    Returns:
        A plotly Figure object.
    """
    with_rob = [s for s in studies if s.risk_of_bias is not None]
    if not with_rob:
        fig = go.Figure()
        fig.add_annotation(text="No risk of bias data available", showarrow=False)
        return fig

    domains = [
        ("Randomization", "random_sequence_generation"),
        ("Allocation", "allocation_concealment"),
        ("Blinding (participants)", "blinding_participants"),
        ("Blinding (outcome)", "blinding_outcome"),
        ("Incomplete outcome", "incomplete_outcome"),
        ("Selective reporting", "selective_reporting"),
    ]

    risk_to_num = {BiasRisk.LOW: 0, BiasRisk.UNCLEAR: 1, BiasRisk.HIGH: 2}
    risk_to_text = {BiasRisk.LOW: "Low", BiasRisk.UNCLEAR: "Unclear", BiasRisk.HIGH: "High"}

    study_labels: list[str] = []
    z_data: list[list[int]] = []
    text_data: list[list[str]] = []

    for study in with_rob:
        first_author = study.authors[0].split()[-1] if study.authors else "Unknown"
        study_labels.append(f"{first_author} {study.publication_date.year}")

        row_z: list[int] = []
        row_text: list[str] = []
        rob: RiskOfBias = study.risk_of_bias
        for _, attr in domains:
            risk = getattr(rob, attr)
            row_z.append(risk_to_num[risk])
            row_text.append(risk_to_text[risk])
        z_data.append(row_z)
        text_data.append(row_text)

    fig = go.Figure(
        data=go.Heatmap(
            z=z_data,
            x=[d[0] for d in domains],
            y=study_labels,
            text=text_data,
            texttemplate="%{text}",
            colorscale=[
                [0, "#2ecc71"],  # green = low
                [0.5, "#f1c40f"],  # yellow = unclear
                [1, "#e74c3c"],  # red = high
            ],
            showscale=False,
            hovertemplate="Study: %{y}<br>Domain: %{x}<br>Risk: %{text}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Risk of Bias Assessment",
        xaxis_title="Domain",
        yaxis_title="Study",
        height=max(300, len(with_rob) * 35 + 100),
    )

    return fig


def build_study_table(
    studies: list[Study],
    result: AnalysisResult,
) -> list[dict]:
    """Build study table data for display.

    Args:
        studies: Studies with extracted data.
        result: Meta-analysis result for weight computation.

    Returns:
        List of dicts suitable for st.dataframe.
    """
    valid = [s for s in studies if s.has_extractable_data and s.se_from_ci]
    if not valid:
        return []

    weights = compute_study_weights(valid, result.tau_squared)
    weight_map = {s.pmid: w for s, w in weights}

    rows = []
    for study in valid:
        first_author = study.authors[0].split()[-1] if study.authors else "Unknown"
        row = {
            "PMID": study.pmid,
            "Author": f"{first_author} {study.publication_date.year}",
            "Effect Size": round(study.effect_size, 3),
            "CI Lower": round(study.ci_lower, 3),
            "CI Upper": round(study.ci_upper, 3),
            "Weight (%)": round(weight_map.get(study.pmid, 0), 1),
            "N (total)": study.sample_size_total or "N/A",
        }
        if study.population:
            row["Population"] = study.population
        if study.intervention:
            row["Intervention"] = study.intervention
        if study.comparator:
            row["Comparator"] = study.comparator
        if study.outcome:
            row["Outcome"] = study.outcome
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Topic discovery
# ---------------------------------------------------------------------------


def discover_topics(base_dir: Path) -> list[str]:
    """Find available topics from the datasets directory.

    Args:
        base_dir: Base directory containing a datasets/ subdirectory.

    Returns:
        List of topic IDs (directory names with config.yaml present).
    """
    datasets_dir = base_dir / "datasets"
    if not datasets_dir.exists():
        return []

    topics = []
    for child in sorted(datasets_dir.iterdir()):
        if child.is_dir() and (child / "config.yaml").exists():
            topics.append(child.name)
    return topics


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------


def main() -> None:
    """Streamlit dashboard entry point."""
    st.set_page_config(
        page_title="Evidence Sync Dashboard",
        page_icon="",
        layout="wide",
    )

    st.title("Evidence Sync Dashboard")

    # Determine base directory from CLI args or default to cwd
    if len(sys.argv) > 1:
        base_dir = Path(sys.argv[-1]).resolve()
    else:
        base_dir = Path.cwd()

    if not base_dir.is_dir():
        st.error(f"Base directory does not exist: {base_dir}")
        st.stop()

    # Sidebar: topic selector
    topics = discover_topics(base_dir)
    if not topics:
        st.warning(
            f"No topics found in {base_dir / 'datasets'}. "
            "Initialize a topic with `evidence-sync init <topic_id>` first."
        )
        return

    selected_topic = st.sidebar.selectbox("Select Topic", topics)

    # Load data
    config_path = base_dir / "datasets" / selected_topic / "config.yaml"
    config = load_review_config(config_path)
    studies_dir = base_dir / "datasets" / selected_topic / "studies"
    analysis_dir = base_dir / "datasets" / selected_topic / "analysis"

    studies = load_all_studies(studies_dir)
    result = load_analysis(analysis_dir)

    # Sidebar: summary stats
    st.sidebar.markdown("---")
    st.sidebar.subheader("Summary")
    st.sidebar.metric("Total studies", len(studies))
    with_data = [s for s in studies if s.has_extractable_data]
    st.sidebar.metric("With extracted data", len(with_data))

    # Review status counts
    from evidence_sync.review import get_review_summary

    review_summary = get_review_summary(studies)
    st.sidebar.metric("Approved", review_summary["approved"])
    st.sidebar.metric("Rejected", review_summary["rejected"])

    if result:
        st.sidebar.metric("Pooled effect", f"{result.pooled_effect:.4f}")
        st.sidebar.metric(
            "95% CI",
            f"[{result.pooled_ci_lower:.4f}, {result.pooled_ci_upper:.4f}]",
        )
        st.sidebar.metric("p-value", f"{result.pooled_p_value:.6f}")
        st.sidebar.metric("I\u00b2", f"{result.i_squared:.1f}%")

        # Drift check
        from evidence_sync.drift import detect_drift

        drift = detect_drift(result, None, config)
        if drift.alert_triggered:
            st.sidebar.error("Drift detected")
        else:
            st.sidebar.success("No drift detected")
    else:
        st.sidebar.info("No analysis yet. Run `evidence-sync analyze`.")

    if result is None:
        st.info(
            f"No analysis found for **{selected_topic}**. "
            "Run `evidence-sync analyze` to generate results."
        )
        return

    # Main page tabs
    (
        tab_forest,
        tab_funnel,
        tab_timeline,
        tab_quality,
        tab_review,
        tab_prisma,
        tab_export,
    ) = st.tabs(
        [
            "Forest Plot",
            "Funnel Plot",
            "Evidence Timeline",
            "Study Quality",
            "Review Queue",
            "PRISMA",
            "Export",
        ]
    )

    with tab_forest:
        st.subheader(f"Forest Plot: {config.topic_name}")
        fig = build_forest_plot(studies, result, title=config.topic_name)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Study Table")
        table_data = build_study_table(studies, result)
        if table_data:
            st.dataframe(table_data, use_container_width=True)
        else:
            st.info("No study data available for table.")

    with tab_funnel:
        st.subheader("Funnel Plot")
        fig = build_funnel_plot(studies, result)
        st.plotly_chart(fig, use_container_width=True)

        if result.egger_p_value is not None:
            if result.egger_p_value < 0.05:
                st.warning(
                    f"Egger's test suggests potential publication bias "
                    f"(intercept = {result.egger_intercept:.3f}, "
                    f"p = {result.egger_p_value:.3f})."
                )
            else:
                st.success(
                    f"Egger's test does not indicate significant publication bias "
                    f"(p = {result.egger_p_value:.3f})."
                )

    with tab_timeline:
        st.subheader("Evidence Timeline")
        fig = build_evidence_timeline(studies, result)
        st.plotly_chart(fig, use_container_width=True)

        if result.analysis_date:
            st.caption(f"Latest analysis: {result.analysis_date.isoformat()}")

    with tab_quality:
        st.subheader("Risk of Bias Assessment")
        fig = build_rob_heatmap(studies)
        st.plotly_chart(fig, use_container_width=True)

    with tab_review:
        st.subheader("Review Queue")

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Pending", review_summary["pending"])
        col2.metric("Approved", review_summary["approved"])
        col3.metric("Rejected", review_summary["rejected"])
        col4.metric("Corrected", review_summary["corrected"])

        # Pending studies table
        from evidence_sync.review import get_pending_studies

        pending = get_pending_studies(studies)
        if pending:
            st.markdown("### Pending Studies")
            rows = []
            for s in pending:
                conf = s.extraction_confidence
                if conf is not None:
                    if conf >= 0.8:
                        conf_display = f":green[{conf:.2f}]"
                    elif conf >= 0.5:
                        conf_display = f":orange[{conf:.2f}]"
                    else:
                        conf_display = f":red[{conf:.2f}]"
                else:
                    conf_display = "N/A"

                row = {
                    "PMID": s.pmid,
                    "Title": s.title[:60],
                    "Effect Size": (f"{s.effect_size:.3f}" if s.effect_size is not None else "N/A"),
                    "CI": (
                        f"[{s.ci_lower:.3f}, {s.ci_upper:.3f}]"
                        if s.ci_lower is not None and s.ci_upper is not None
                        else "N/A"
                    ),
                    "Confidence": conf_display,
                }
                if s.population:
                    row["Population"] = s.population
                if s.intervention:
                    row["Intervention"] = s.intervention
                rows.append(row)
            st.dataframe(rows, use_container_width=True)
        else:
            st.success("All studies have been reviewed.")

    with tab_prisma:
        st.subheader("PRISMA 2020 Compliance")

        from evidence_sync.prisma import (
            format_prisma_flow_text,
            generate_prisma_checklist,
            generate_prisma_flow,
        )

        flow = generate_prisma_flow(studies, [], config)

        # Flow diagram
        st.markdown("### Flow Diagram")
        st.code(format_prisma_flow_text(flow), language=None)

        # Checklist
        st.markdown("### PRISMA Checklist")
        checklist = generate_prisma_checklist(
            studies,
            config,
            result=result,
            flow=flow,
        )
        compliant_count = sum(1 for i in checklist if i.compliant)
        st.metric(
            "Compliance",
            f"{compliant_count}/{len(checklist)}",
            f"{compliant_count / len(checklist) * 100:.0f}%",
        )

        checklist_rows = []
        for item in checklist:
            checklist_rows.append(
                {
                    "#": item.number,
                    "Section": item.section,
                    "Topic": item.topic,
                    "Compliant": "Yes" if item.compliant else "No",
                    "Evidence": item.evidence[:60] if item.evidence else "-",
                }
            )
        st.dataframe(checklist_rows, use_container_width=True)

    with tab_export:
        st.subheader("Export Data")

        from evidence_sync.export import (
            export_csv,
            export_r_dataframe,
            export_revman_xml,
        )

        col_csv, col_xml, col_r = st.columns(3)

        with col_csv:
            csv_data = export_csv(studies)
            st.download_button(
                "Download CSV",
                csv_data,
                file_name=f"{selected_topic}_studies.csv",
                mime="text/csv",
            )

        with col_xml:
            xml_data = export_revman_xml(studies, config, result=result)
            st.download_button(
                "Download RevMan XML",
                xml_data,
                file_name=f"{selected_topic}_revman.xml",
                mime="application/xml",
            )

        with col_r:
            r_data = export_r_dataframe(studies)
            st.download_button(
                "Download R Data",
                r_data,
                file_name=f"{selected_topic}_metafor.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
