"""CLI entry point for Evidence Sync."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from evidence_sync import __version__
from evidence_sync.config import (
    get_analysis_dir,
    get_studies_dir,
    load_review_config,
    save_review_config,
    validate_topic_id,
)
from evidence_sync.drift import detect_drift
from evidence_sync.models import EffectMeasure, ReviewConfig
from evidence_sync.statistics import run_meta_analysis
from evidence_sync.versioning import (
    commit_dataset_changes,
    load_all_studies,
    load_analysis,
    save_analysis,
    save_study,
)

logger = logging.getLogger(__name__)


def _validated_topic_id(ctx: click.Context, param: click.Parameter, value: str) -> str:
    """Click callback to validate topic_id argument."""
    try:
        return validate_topic_id(value)
    except ValueError as e:
        raise click.BadParameter(str(e)) from e


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@click.group()
@click.version_option(version=__version__, prog_name="evidence-sync")
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
@click.argument("topic_id", callback=_validated_topic_id)
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
@click.argument("topic_id", callback=_validated_topic_id)
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
@click.argument("topic_id", callback=_validated_topic_id)
@click.option("--model", default="claude-sonnet-4-20250514", help="Model for extraction")
@click.option(
    "--provider",
    type=click.Choice(["claude", "gemini"]),
    default="claude",
    help="AI provider for extraction",
)
@click.option(
    "--auto-commit",
    is_flag=True,
    default=False,
    help="Auto-commit changes to git",
)
@click.pass_context
def extract(
    ctx: click.Context,
    topic_id: str,
    model: str,
    provider: str,
    auto_commit: bool,
) -> None:
    """Extract data from studies using Claude or Gemini."""
    import os

    from evidence_sync.extractor import extract_study_data, extract_study_data_gemini

    base_dir = ctx.obj["base_dir"]
    studies_dir = get_studies_dir(base_dir, topic_id)
    studies = load_all_studies(studies_dir)

    unextracted = [s for s in studies if not s.has_extractable_data]
    if not unextracted:
        click.echo("All studies already have extracted data.")
        return

    label = f"{provider}:{model}" if provider == "gemini" else model
    click.echo(f"Extracting data from {len(unextracted)} studies using {label}...")

    if provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            click.echo(
                "Error: GEMINI_API_KEY environment variable is required for Gemini provider."
            )
            return
        for i, study in enumerate(unextracted, 1):
            click.echo(f"  [{i}/{len(unextracted)}] {study.pmid}: {study.title[:60]}...")
            extract_study_data_gemini(study, api_key, model=model)
            save_study(study, studies_dir)
    else:
        import anthropic

        client = anthropic.Anthropic()
        for i, study in enumerate(unextracted, 1):
            click.echo(f"  [{i}/{len(unextracted)}] {study.pmid}: {study.title[:60]}...")
            extract_study_data(study, client, model=model)
            save_study(study, studies_dir)

    click.echo(f"Extraction complete. {len(unextracted)} studies processed.")

    if auto_commit:
        if commit_dataset_changes(base_dir, topic_id):
            click.echo("Changes committed to git.")
        else:
            click.echo("No changes to commit.")


@cli.command()
@click.argument("topic_id", callback=_validated_topic_id)
@click.pass_context
def enrich(ctx: click.Context, topic_id: str) -> None:
    """Enrich studies with full-text and registry data."""
    from evidence_sync.fulltext import enrich_study_with_fulltext

    base_dir = ctx.obj["base_dir"]
    studies_dir = get_studies_dir(base_dir, topic_id)
    studies = load_all_studies(studies_dir)

    if not studies:
        click.echo(f"No studies found for topic '{topic_id}'.")
        return

    enriched_count = 0
    for i, study in enumerate(studies, 1):
        click.echo(f"  [{i}/{len(studies)}] {study.pmid}: {study.title[:60]}...")
        if enrich_study_with_fulltext(study):
            save_study(study, studies_dir)
            enriched_count += 1
            source = study.data_source
            extras = []
            if study.pmc_id:
                extras.append(f"PMC:{study.pmc_id}")
            if study.nct_id:
                extras.append(f"NCT:{study.nct_id}")
            detail = f" ({', '.join(extras)})" if extras else ""
            click.echo(f"    -> enriched [{source}]{detail}")

    click.echo(f"Enrichment complete. {enriched_count}/{len(studies)} studies enriched.")


@cli.command()
@click.argument("topic_id", callback=_validated_topic_id)
@click.option("--auto-commit", is_flag=True, default=False, help="Auto-commit changes to git")
@click.pass_context
def analyze(ctx: click.Context, topic_id: str, auto_commit: bool) -> None:
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

    if auto_commit:
        if commit_dataset_changes(base_dir, topic_id):
            click.echo("Changes committed to git.")
        else:
            click.echo("No changes to commit.")


@cli.command()
@click.argument("topic_id", callback=_validated_topic_id)
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
        studies,
        result,
        out / "forest_plot.png",
        title=f"Forest Plot: {config.topic_name}",
    )
    click.echo(f"Forest plot: {forest_path}")

    funnel_path = generate_funnel_plot(studies, result, out / "funnel_plot.png")
    click.echo(f"Funnel plot: {funnel_path}")


@cli.command()
@click.argument("topic_id", callback=_validated_topic_id)
@click.option("--max-results", default=100)
@click.option("--model", default="claude-sonnet-4-20250514")
@click.pass_context
def run(ctx: click.Context, topic_id: str, max_results: int, model: str) -> None:
    """Full pipeline: search -> extract -> analyze -> report."""
    ctx.invoke(search, topic_id=topic_id, max_results=max_results)
    ctx.invoke(extract, topic_id=topic_id, model=model)
    ctx.invoke(analyze, topic_id=topic_id)
    ctx.invoke(report, topic_id=topic_id)


@cli.command()
@click.argument("topic_id", callback=_validated_topic_id)
@click.pass_context
def status(ctx: click.Context, topic_id: str) -> None:
    """Show status for a review topic."""
    base_dir = ctx.obj["base_dir"]
    config_path = base_dir / "datasets" / topic_id / "config.yaml"

    if not config_path.exists():
        click.echo(f"Topic '{topic_id}' not found.")
        return

    config = load_review_config(config_path)
    studies_dir = get_studies_dir(base_dir, topic_id)
    analysis_dir = get_analysis_dir(base_dir, topic_id)

    studies = load_all_studies(studies_dir)
    with_data = [s for s in studies if s.has_extractable_data]
    without_data = [s for s in studies if not s.has_extractable_data]

    click.echo(f"Topic: {config.topic_name} ({topic_id})")
    click.echo(
        f"Studies: {len(studies)} total, {len(with_data)} with data, "
        f"{len(without_data)} without data"
    )

    result = load_analysis(analysis_dir)
    if result:
        click.echo(f"\nLatest analysis ({result.analysis_date}):")
        click.echo(
            f"  Pooled effect: {result.pooled_effect:.4f} "
            f"[{result.pooled_ci_lower:.4f}, {result.pooled_ci_upper:.4f}]"
        )
        click.echo(f"  P-value: {result.pooled_p_value:.6f}")
        click.echo(f"  I-squared: {result.i_squared:.1f}%")

        drift = detect_drift(result, None, config)
        if drift.alert_triggered:
            click.echo("  Drift: ALERT")
            for reason in drift.alert_reasons:
                click.echo(f"    - {reason}")
        else:
            click.echo("  Drift: None detected")
    else:
        click.echo("\nNo analysis available yet.")


@cli.command(name="list")
@click.pass_context
def list_topics(ctx: click.Context) -> None:
    """List all configured review topics."""
    base_dir = ctx.obj["base_dir"]
    datasets_dir = base_dir / "datasets"

    if not datasets_dir.exists():
        click.echo("No datasets directory found.")
        return

    topics_found = False
    for child in sorted(datasets_dir.iterdir()):
        if child.is_dir() and (child / "config.yaml").exists():
            config = load_review_config(child / "config.yaml")
            studies_dir = child / "studies"
            n_studies = len(list(studies_dir.glob("*.yaml"))) if studies_dir.exists() else 0
            has_analysis = (child / "analysis" / "summary.yaml").exists()
            analysis_indicator = "analyzed" if has_analysis else "no analysis"
            click.echo(
                f"  {config.topic_id}: {config.topic_name} "
                f"({n_studies} studies, {analysis_indicator})"
            )
            topics_found = True

    if not topics_found:
        click.echo("No topics configured. Run 'evidence-sync init <topic_id>' to create one.")


@cli.command()
@click.argument("topic_id", callback=_validated_topic_id)
@click.option(
    "--ground-truth-dir",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Directory with ground truth YAML files (default: tests/ground_truth/)",
)
@click.pass_context
def validate(ctx: click.Context, topic_id: str, ground_truth_dir: Path | None) -> None:
    """Validate extraction accuracy against ground truth."""
    from datetime import date as date_cls

    from evidence_sync.accuracy import (
        compare_extraction,
        compute_accuracy_report,
        format_accuracy_report,
        load_ground_truth,
    )
    from evidence_sync.extractor import extract_study_data_from_dict
    from evidence_sync.models import Study
    from evidence_sync.versioning import load_study as load_study_from_file

    base_dir = ctx.obj["base_dir"]
    studies_dir = get_studies_dir(base_dir, topic_id)

    # Determine ground truth directory
    if ground_truth_dir is None:
        ground_truth_dir = Path(__file__).parent.parent.parent / "tests" / "ground_truth"

    if not ground_truth_dir.exists():
        click.echo(f"Ground truth directory not found: {ground_truth_dir}")
        return

    gt_entries = load_ground_truth(ground_truth_dir)
    if not gt_entries:
        click.echo("No ground truth entries found.")
        return

    click.echo(f"Loaded {len(gt_entries)} ground truth entries from {ground_truth_dir}")

    results = []
    for gt_entry in gt_entries:
        pmid = str(gt_entry["pmid"])
        gt_data = gt_entry["ground_truth"]

        # Check if extracted study exists on disk
        study_path = studies_dir / f"{pmid}.yaml"
        if study_path.exists():
            extracted_study = load_study_from_file(study_path)
        else:
            # Create study from ground truth metadata and apply as extracted data
            pub_date = gt_entry.get("publication_date", "2020-01-01")
            if isinstance(pub_date, str):
                pub_date = date_cls.fromisoformat(pub_date)

            study = Study(
                pmid=pmid,
                title=gt_entry.get("title", ""),
                authors=gt_entry.get("authors", []),
                journal=gt_entry.get("journal", ""),
                publication_date=pub_date,
                abstract=gt_entry.get("abstract", ""),
            )
            extracted_study = extract_study_data_from_dict(study, gt_data, model="ground-truth")

        comparison = compare_extraction(extracted_study, gt_data)
        results.append(comparison)
        acc = comparison["accuracy"]
        if acc >= 0.8:
            tag = "OK"
        elif acc >= 0.5:
            tag = "WARN"
        else:
            tag = "FAIL"
        click.echo(
            f"  [{tag}] {pmid}: {acc:.0%} ({comparison['n_correct']}/{comparison['n_total']})"
        )

    report = compute_accuracy_report(results)
    click.echo("")
    click.echo(format_accuracy_report(report))


@cli.command()
@click.option("--port", default=8501, help="Port for dashboard")
@click.pass_context
def dashboard(ctx: click.Context, port: int) -> None:
    """Launch interactive Streamlit dashboard."""
    import subprocess

    app_path = Path(__file__).parent / "app.py"
    subprocess.run(
        [
            "streamlit",
            "run",
            str(app_path),
            "--server.port",
            str(port),
            "--",
            str(ctx.obj["base_dir"]),
        ]
    )


@cli.command()
@click.argument("topic_id", callback=_validated_topic_id)
@click.option(
    "--model",
    default="claude-sonnet-4-20250514",
    help="Claude model for screening",
)
@click.pass_context
def screen(
    ctx: click.Context,
    topic_id: str,
    model: str,
) -> None:
    """Screen studies for relevance using AI."""
    import anthropic

    from evidence_sync.screening import (
        get_screening_summary,
        screen_study,
    )

    base_dir = ctx.obj["base_dir"]
    config_path = base_dir / "datasets" / topic_id / "config.yaml"
    config = load_review_config(config_path)
    studies_dir = get_studies_dir(base_dir, topic_id)
    studies = load_all_studies(studies_dir)

    # Screen unscreened studies (no PICO data yet)
    unscreened = [s for s in studies if s.population is None]
    if not unscreened:
        click.echo("All studies already screened.")
        return

    click.echo(f"Screening {len(unscreened)} studies...")
    client = anthropic.Anthropic()

    results = []
    for i, study in enumerate(unscreened, 1):
        click.echo(f"  [{i}/{len(unscreened)}] {study.pmid}: {study.title[:60]}...")
        result = screen_study(study, config, client, model)
        results.append(result)
        save_study(study, studies_dir)

    # Show summary
    summary = get_screening_summary(results)
    click.echo("\nScreening complete:")
    click.echo(f"  Include: {summary['include']}")
    click.echo(f"  Exclude: {summary['exclude']}")
    click.echo(f"  Uncertain: {summary['uncertain']}")
    click.echo(f"  Avg relevance: {summary['avg_relevance']:.2f}")


@cli.group()
def review():
    """Manage study review workflow."""
    pass


@review.command(name="pending")
@click.argument("topic_id", callback=_validated_topic_id)
@click.pass_context
def review_pending(ctx: click.Context, topic_id: str) -> None:
    """List studies pending review."""
    from evidence_sync.review import get_pending_studies

    base_dir = ctx.obj["base_dir"]
    studies_dir = get_studies_dir(base_dir, topic_id)
    studies = load_all_studies(studies_dir)
    pending = get_pending_studies(studies)

    if not pending:
        click.echo("No studies pending review.")
        return

    click.echo(f"Pending review ({len(pending)} studies):\n")
    for s in pending:
        conf = f" (confidence: {s.extraction_confidence:.2f})" if s.extraction_confidence else ""
        data = ""
        if s.has_extractable_data:
            data = f" | ES={s.effect_size:.3f} [{s.ci_lower:.3f}, {s.ci_upper:.3f}]"
        click.echo(f"  {s.pmid}: {s.title[:60]}{data}{conf}")


@review.command(name="approve")
@click.argument("topic_id", callback=_validated_topic_id)
@click.argument("pmid")
@click.option("--reviewer", default="cli-user", help="Reviewer name")
@click.option("--notes", default=None, help="Review notes")
@click.pass_context
def review_approve(
    ctx: click.Context,
    topic_id: str,
    pmid: str,
    reviewer: str,
    notes: str | None,
) -> None:
    """Approve a study's extracted data."""
    from evidence_sync.review import approve_study

    base_dir = ctx.obj["base_dir"]
    studies_dir = get_studies_dir(base_dir, topic_id)
    studies = load_all_studies(studies_dir)

    study = next((s for s in studies if s.pmid == pmid), None)
    if study is None:
        click.echo(f"Study {pmid} not found in topic '{topic_id}'.")
        return

    approve_study(study, reviewer, notes)
    save_study(study, studies_dir)
    click.echo(f"Approved study {pmid} (reviewer: {reviewer})")


@review.command(name="reject")
@click.argument("topic_id", callback=_validated_topic_id)
@click.argument("pmid")
@click.option("--reviewer", default="cli-user", help="Reviewer name")
@click.option("--notes", default=None, help="Review notes")
@click.pass_context
def review_reject(
    ctx: click.Context,
    topic_id: str,
    pmid: str,
    reviewer: str,
    notes: str | None,
) -> None:
    """Reject a study's extracted data."""
    from evidence_sync.review import reject_study

    base_dir = ctx.obj["base_dir"]
    studies_dir = get_studies_dir(base_dir, topic_id)
    studies = load_all_studies(studies_dir)

    study = next((s for s in studies if s.pmid == pmid), None)
    if study is None:
        click.echo(f"Study {pmid} not found in topic '{topic_id}'.")
        return

    reject_study(study, reviewer, notes)
    save_study(study, studies_dir)
    click.echo(f"Rejected study {pmid} (reviewer: {reviewer})")


@review.command(name="approve-all")
@click.argument("topic_id", callback=_validated_topic_id)
@click.option("--reviewer", default="cli-user", help="Reviewer name")
@click.pass_context
def review_approve_all(ctx: click.Context, topic_id: str, reviewer: str) -> None:
    """Approve all pending studies (batch operation)."""
    from evidence_sync.review import approve_study, get_pending_studies

    base_dir = ctx.obj["base_dir"]
    studies_dir = get_studies_dir(base_dir, topic_id)
    studies = load_all_studies(studies_dir)
    pending = get_pending_studies(studies)

    if not pending:
        click.echo("No studies pending review.")
        return

    for study in pending:
        approve_study(study, reviewer)
        save_study(study, studies_dir)

    click.echo(f"Approved {len(pending)} studies (reviewer: {reviewer})")


@review.command(name="summary")
@click.argument("topic_id", callback=_validated_topic_id)
@click.pass_context
def review_summary(ctx: click.Context, topic_id: str) -> None:
    """Show review status summary."""
    from evidence_sync.review import get_review_summary

    base_dir = ctx.obj["base_dir"]
    studies_dir = get_studies_dir(base_dir, topic_id)
    studies = load_all_studies(studies_dir)
    summary = get_review_summary(studies)

    click.echo(f"Review summary for '{topic_id}':")
    click.echo(f"  Total:     {summary['total']}")
    click.echo(f"  Pending:   {summary['pending']}")
    click.echo(f"  Approved:  {summary['approved']}")
    click.echo(f"  Rejected:  {summary['rejected']}")
    click.echo(f"  Corrected: {summary['corrected']}")


@cli.command(name="prisma-flow")
@click.argument("topic_id", callback=_validated_topic_id)
@click.pass_context
def prisma_flow(ctx: click.Context, topic_id: str) -> None:
    """Generate PRISMA 2020 flow diagram."""
    from evidence_sync.prisma import format_prisma_flow_text, generate_prisma_flow

    base_dir = ctx.obj["base_dir"]
    config_path = base_dir / "datasets" / topic_id / "config.yaml"
    config = load_review_config(config_path)
    studies_dir = get_studies_dir(base_dir, topic_id)
    studies = load_all_studies(studies_dir)

    flow = generate_prisma_flow(studies, [], config)
    click.echo(format_prisma_flow_text(flow))


@cli.command(name="prisma-checklist")
@click.argument("topic_id", callback=_validated_topic_id)
@click.pass_context
def prisma_checklist(ctx: click.Context, topic_id: str) -> None:
    """Show PRISMA 2020 27-item compliance checklist."""
    from evidence_sync.prisma import (
        format_checklist_text,
        generate_prisma_checklist,
        generate_prisma_flow,
    )

    base_dir = ctx.obj["base_dir"]
    config_path = base_dir / "datasets" / topic_id / "config.yaml"
    config = load_review_config(config_path)
    studies_dir = get_studies_dir(base_dir, topic_id)
    analysis_dir = get_analysis_dir(base_dir, topic_id)
    studies = load_all_studies(studies_dir)
    result = load_analysis(analysis_dir)
    flow = generate_prisma_flow(studies, [], config)

    items = generate_prisma_checklist(
        studies,
        config,
        result=result,
        flow=flow,
    )
    click.echo(format_checklist_text(items))


@cli.command(name="prisma-export")
@click.argument("topic_id", callback=_validated_topic_id)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["csv", "xml", "r"]),
    default="csv",
    help="Export format",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Output file path",
)
@click.pass_context
def prisma_export(
    ctx: click.Context,
    topic_id: str,
    fmt: str,
    output_path: Path | None,
) -> None:
    """Export study data in various formats."""
    from evidence_sync.export import (
        export_csv,
        export_r_dataframe,
        export_revman_xml,
    )

    base_dir = ctx.obj["base_dir"]
    config_path = base_dir / "datasets" / topic_id / "config.yaml"
    config = load_review_config(config_path)
    studies_dir = get_studies_dir(base_dir, topic_id)
    analysis_dir = get_analysis_dir(base_dir, topic_id)
    studies = load_all_studies(studies_dir)
    result = load_analysis(analysis_dir)

    if fmt == "csv":
        data = export_csv(studies, result=result, output_path=output_path)
    elif fmt == "xml":
        data = export_revman_xml(
            studies,
            config,
            result=result,
            output_path=output_path,
        )
    elif fmt == "r":
        data = export_r_dataframe(
            studies,
            result=result,
            output_path=output_path,
        )
    else:
        click.echo(f"Unknown format: {fmt}")
        return

    if output_path:
        click.echo(f"Exported to {output_path}")
    else:
        click.echo(data)


@cli.command()
@click.argument("topic_id", callback=_validated_topic_id)
@click.pass_context
def protocol(ctx: click.Context, topic_id: str) -> None:
    """Generate PROSPERO-compatible protocol template."""
    from evidence_sync.protocol import format_protocol_text, generate_protocol

    base_dir = ctx.obj["base_dir"]
    config_path = base_dir / "datasets" / topic_id / "config.yaml"
    config = load_review_config(config_path)
    analysis_dir = get_analysis_dir(base_dir, topic_id)
    result = load_analysis(analysis_dir)

    proto = generate_protocol(config, result=result)
    click.echo(format_protocol_text(proto))


if __name__ == "__main__":
    cli()
