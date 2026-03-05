"""Visualization — forest plots, funnel plots, and evidence timeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from evidence_sync.models import AnalysisResult, Study
from evidence_sync.statistics import compute_study_weights

matplotlib.use("Agg")
logger = logging.getLogger(__name__)


def generate_forest_plot(
    studies: list[Study],
    result: AnalysisResult,
    output_path: Path,
    title: Optional[str] = None,
) -> Path:
    """Generate a forest plot showing individual study effects and pooled estimate.

    Args:
        studies: Studies included in the analysis.
        result: The meta-analysis result.
        output_path: Where to save the plot (PNG).
        title: Optional plot title.

    Returns:
        Path to the saved plot.
    """
    valid = [s for s in studies if s.has_extractable_data and s.se_from_ci]
    if not valid:
        logger.warning("No valid studies for forest plot")
        return output_path

    weights = compute_study_weights(valid, result.tau_squared)
    weight_map = {s.pmid: w for s, w in weights}

    # Sort by effect size
    valid.sort(key=lambda s: s.effect_size or 0)

    n = len(valid)
    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.5 + 2)))

    y_positions = list(range(n + 1))  # +1 for pooled

    for i, study in enumerate(valid):
        es = study.effect_size
        ci_lo = study.ci_lower
        ci_hi = study.ci_upper
        weight_pct = weight_map.get(study.pmid, 0)

        # Individual study
        marker_size = max(4, weight_pct * 0.8)
        ax.plot(es, i, "s", color="steelblue", markersize=marker_size, zorder=3)
        ax.plot([ci_lo, ci_hi], [i, i], "-", color="steelblue", linewidth=1.5, zorder=2)

        # Label
        first_author = study.authors[0].split()[-1] if study.authors else "Unknown"
        year = study.publication_date.year
        label = f"{first_author} {year}"
        ax.text(-0.02, i, label, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=8)

        # Weight and effect text on right
        info = f"{es:.2f} [{ci_lo:.2f}, {ci_hi:.2f}]  ({weight_pct:.1f}%)"
        ax.text(1.02, i, info, transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=7)

    # Pooled estimate (diamond)
    pooled_y = n
    diamond_x = [result.pooled_ci_lower, result.pooled_effect, result.pooled_ci_upper,
                 result.pooled_effect]
    diamond_y = [pooled_y, pooled_y + 0.3, pooled_y, pooled_y - 0.3]
    ax.fill(diamond_x, diamond_y, color="firebrick", zorder=3)
    ax.text(-0.02, pooled_y, "Pooled", transform=ax.get_yaxis_transform(), ha="right",
            va="center", fontsize=8, fontweight="bold")
    info = (
        f"{result.pooled_effect:.2f} "
        f"[{result.pooled_ci_lower:.2f}, {result.pooled_ci_upper:.2f}]"
    )
    ax.text(1.02, pooled_y, info, transform=ax.get_yaxis_transform(), ha="left", va="center",
            fontsize=7, fontweight="bold")

    # Reference line at null effect
    null_effect = 0.0 if "difference" in result.effect_measure.value else 1.0
    ax.axvline(x=null_effect, color="gray", linestyle="--", linewidth=0.8, zorder=1)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([""] * len(y_positions))
    ax.set_xlabel(f"Effect Size ({result.effect_measure.value.replace('_', ' ').title()})")

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")

    # Heterogeneity annotation
    het_text = (
        f"I² = {result.i_squared:.1f}%, "
        f"τ² = {result.tau_squared:.4f}, "
        f"Q = {result.q_statistic:.2f} (p = {result.q_p_value:.3f})"
    )
    ax.text(0.5, -0.08, het_text, transform=ax.transAxes, ha="center", fontsize=8, color="gray")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Forest plot saved to {output_path}")
    return output_path


def generate_funnel_plot(
    studies: list[Study],
    result: AnalysisResult,
    output_path: Path,
) -> Path:
    """Generate a funnel plot for publication bias assessment."""
    valid = [s for s in studies if s.has_extractable_data and s.se_from_ci]
    if not valid:
        return output_path

    effects = [s.effect_size for s in valid]
    ses = [s.se_from_ci for s in valid]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(effects, ses, color="steelblue", s=40, zorder=3)

    # Pooled effect line
    ax.axvline(x=result.pooled_effect, color="firebrick", linestyle="-", linewidth=1, zorder=2)

    # Pseudo 95% CI funnel
    se_range = np.linspace(0, max(ses) * 1.1, 100)
    ci_lo = result.pooled_effect - 1.96 * se_range
    ci_hi = result.pooled_effect + 1.96 * se_range
    ax.plot(ci_lo, se_range, "--", color="gray", linewidth=0.8)
    ax.plot(ci_hi, se_range, "--", color="gray", linewidth=0.8)

    ax.set_xlabel("Effect Size")
    ax.set_ylabel("Standard Error")
    ax.set_title("Funnel Plot")
    ax.invert_yaxis()

    if result.egger_p_value is not None:
        ax.text(
            0.02, 0.98,
            f"Egger's test: intercept = {result.egger_intercept:.3f}, "
            f"p = {result.egger_p_value:.3f}",
            transform=ax.transAxes, fontsize=8, va="top", color="gray",
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Funnel plot saved to {output_path}")
    return output_path
