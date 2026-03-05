"""CLI entry point for Evidence Sync."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from evidence_sync.config import (
    get_analysis_dir,
    get_studies_dir,
    load_review_config,
    save_review_config,
)
from evidence_sync.drift import detect_drift
from evidence_sync.models import EffectMeasure, ReviewConfig
from evidence_sync.statistics import run_meta_analysis
from evidence_sync.versioning import load_all_studies, load_analysis, save_analysis, save_study

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")
@click.option(
    "--base-dir",
    type=click.Path(exists=True, path_type=Path),
    default=".",
    help="Base directory for datasets",
)
@click.pass_context
def cli(ctx: click.Context, verbose: bool, base_dir: Path) -> None:
    """Evidence Sync — Living meta-analysis engine."""
    _setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["base_dir"] = base_dir.resolve()


@cli.command()
@click.argument("topic_id")
@click.option("--name", prompt="Review topic name", help="Human-readable topic name")
@click.option("--query", prompt="PubMed search query", help="PubMed search query string")
@click.option(
    "--effect-measure",
    type=click.Choice([e.value for e in EffectMeasure]),
    default="odds_ratio",
    help="Effect measure type",
)
@click.option("--outcome", prompt="Primary outcome", help="Primary outcome measure")
@click.pass_context
def init(
    ctx: click.Context,
    topic_id: str,
    name: str,
    query: str,
    effect_measure: str,
    outcome: str,
) -> None:
    """Initialize a new review topic."""
    base_dir = ctx.obj["base_dir"]
    dataset_dir = base_dir / "datasets" / topic_id

    if dataset_dir.exists():
        click.echo(f"Topic '{topic_id}' already exists at {dataset_dir}")
        return

    config = ReviewConfig(
        topic_id=topic_id,
        topic_name=name,
        search_query=query,
        effect_measure=EffectMeasure(effect_measure),
        primary_outcome=outcome,
    )

    config_path = dataset_dir / "config.yaml"
    save_review_config(config, config_path)

    (dataset_dir / "studies").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "analysis").mkdir(parents=True, exist_ok=True)

    click.echo(f"Initialized topic '{topic_id}' at {dataset_dir}")


@cli.command()
@click.argument("topic_id")
@click.option("--max-results", default=100, help="Maximum search results")
@click.pass_context
def search(ctx: click.Context, topic_id: str, max_results: int) -> None:
    """Search PubMed for new studies."""
    from evidence_sync.monitor import deduplicate, fetch_study_details, search_pubmed

    base_dir = ctx.obj["base_dir"]
    config_path = base_dir / "datasets" / topic_id / "config.yaml"

    if not config_path.exists():
        click.echo(f"Topic '{topic_id}' not found. Run 'evidence-sync init {topic_id}' first.")
        return

    config = load_review_config(config_path)
    studies_dir = get_studies_dir(base_dir, topic_id)

    click.echo(f"Searching PubMed for: {config.search_query}")
    result = search_pubmed(config, max_results=max_results)
    click.echo(f"Found {result.total_count} total, retrieved {len(result.pmids)} PMIDs")

    new_pmids = deduplicate(result.pmids, studies_dir)
    click.echo(f"After dedup: {len(new_pmids)} new studies")

    if not new_pmids:
        click.echo("No new studies found.")
        return

    click.echo(f"Fetching details for {len(new_pmids)} studies...")
    studies = fetch_study_details(new_pmids)

    for study in studies:
        save_study(study, studies_dir)

    click.echo(f"Saved {len(studies)} new studies to {studies_dir}")


@cli.command()
@click.argument("topic_id")
@click.option("--model", default="claude-sonnet-4-20250514", help="Claude model for extraction")
@click.pass_context
def extract(ctx: click.Context, topic_id: str, model: str) -> None:
    """Extract data from studies using Claude."""
    import anthropic

    from evidence_sync.extractor import extract_study_data

    base_dir = ctx.obj["base_dir"]
    studies_dir = get_studies_dir(base_dir, topic_id)
    studies = load_all_studies(studies_dir)

    unextracted = [s for s in studies if not s.has_extractable_data]
    if not unextracted:
        click.echo("All studies already have extracted data.")
        return

    click.echo(f"Extracting data from {len(unextracted)} studies using {model}...")
    client = anthropic.Anthropic()

    for i, study in enumerate(unextracted, 1):
        click.echo(f"  [{i}/{len(unextracted)}] {study.pmid}: {study.title[:60]}...")
        extract_study_data(study, client, model=model)
        save_study(study, studies_dir)

    click.echo(f"Extraction complete. {len(unextracted)} studies processed.")


@cli.command()
@click.argument("topic_id")
@click.pass_context
def analyze(ctx: click.Context, topic_id: str) -> None:
    """Run meta-analysis on extracted studies."""
    base_dir = ctx.obj["base_dir"]
    config_path = base_dir / "datasets" / topic_id / "config.yaml"
    config = load_review_config(config_path)

    studies_dir = get_studies_dir(base_dir, topic_id)
    analysis_dir = get_analysis_dir(base_dir, topic_id)

    studies = load_all_studies(studies_dir)
    valid = [s for s in studies if s.has_extractable_data]
    click.echo(f"Running meta-analysis on {len(valid)} studies...")

    # Load previous analysis for drift detection
    previous = load_analysis(analysis_dir)

    result = run_meta_analysis(valid, config.effect_measure, topic=topic_id)
    if result is None:
        click.echo("Insufficient data for meta-analysis (need >= 2 studies with extracted data).")
        return

    save_analysis(result, analysis_dir)

    # Report
    click.echo("\n--- Meta-Analysis Results ---")
    click.echo(f"Studies included: {result.n_studies}")
    click.echo(
        f"Pooled effect: {result.pooled_effect:.4f} "
        f"[{result.pooled_ci_lower:.4f}, {result.pooled_ci_upper:.4f}]"
    )
    click.echo(f"P-value: {result.pooled_p_value:.6f}")
    click.echo(f"I²: {result.i_squared:.1f}%")
    click.echo(f"τ²: {result.tau_squared:.6f}")

    # Drift detection
    drift = detect_drift(result, previous, config)
    if drift.alert_triggered:
        click.echo("\n*** DRIFT ALERT ***")
        for reason in drift.alert_reasons:
            click.echo(f"  - {reason}")


@cli.command()
@click.argument("topic_id")
@click.option("--output-dir", type=click.Path(path_type=Path), default=None)
@click.pass_context
def report(ctx: click.Context, topic_id: str, output_dir: Path | None) -> None:
    """Generate report with forest plot."""
    from evidence_sync.dashboard import generate_forest_plot, generate_funnel_plot

    base_dir = ctx.obj["base_dir"]
    config_path = base_dir / "datasets" / topic_id / "config.yaml"
    config = load_review_config(config_path)

    studies_dir = get_studies_dir(base_dir, topic_id)
    analysis_dir = get_analysis_dir(base_dir, topic_id)

    studies = load_all_studies(studies_dir)
    result = load_analysis(analysis_dir)

    if result is None:
        click.echo("No analysis found. Run 'evidence-sync analyze' first.")
        return

    out = output_dir or (analysis_dir / "plots")

    forest_path = generate_forest_plot(
        studies, result, out / "forest_plot.png",
        title=f"Forest Plot: {config.topic_name}",
    )
    click.echo(f"Forest plot: {forest_path}")

    funnel_path = generate_funnel_plot(studies, result, out / "funnel_plot.png")
    click.echo(f"Funnel plot: {funnel_path}")


@cli.command()
@click.argument("topic_id")
@click.option("--max-results", default=100)
@click.option("--model", default="claude-sonnet-4-20250514")
@click.pass_context
def run(ctx: click.Context, topic_id: str, max_results: int, model: str) -> None:
    """Full pipeline: search -> extract -> analyze -> report."""
    ctx.invoke(search, topic_id=topic_id, max_results=max_results)
    ctx.invoke(extract, topic_id=topic_id, model=model)
    ctx.invoke(analyze, topic_id=topic_id)
    ctx.invoke(report, topic_id=topic_id)


if __name__ == "__main__":
    cli()
