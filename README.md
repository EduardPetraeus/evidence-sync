# Evidence Sync

**Living meta-analysis engine** — an autonomous pipeline that monitors PubMed, extracts study data via Claude, and maintains always-up-to-date meta-analyses.

Systematic reviews take 6-18 months and become stale the day they're published. Evidence Sync turns static medical reviews into living, versioned datasets with automatic drift detection.

## How It Works

```
PubMed API → Study Finder → Claude Extractor → Statistical Engine → Dashboard
                                    ↓
                              Git Dataset → Drift Detector → Alerts
                            (YAML, versioned)
```

1. **Monitor** — Daily PubMed search for new RCTs matching your review topic
2. **Extract** — Claude reads abstracts and extracts structured data (effect sizes, CIs, bias assessment)
3. **Analyze** — Random-effects meta-analysis (DerSimonian-Laird) with heterogeneity metrics
4. **Detect** — Compare against previous analysis, alert on significant drift
5. **Visualize** — Forest plots, funnel plots, evidence timeline

## Quick Start

```bash
pip install -e .

# Initialize a review topic
evidence-sync init ssri-depression \
  --name "SSRIs vs. Placebo for MDD" \
  --query "(SSRI OR fluoxetine) AND depression AND placebo" \
  --effect-measure odds_ratio \
  --outcome "Response rate"

# Run the full pipeline
evidence-sync run ssri-depression

# Or run individual steps
evidence-sync search ssri-depression
evidence-sync extract ssri-depression
evidence-sync analyze ssri-depression
evidence-sync report ssri-depression
```

## Requirements

- Python 3.11+
- `ANTHROPIC_API_KEY` environment variable (for Claude extraction)
- Internet access (for PubMed API)

## Validation

Evidence Sync validates against published Cochrane reviews. The pilot topic (SSRIs for depression) targets <5% deviation from Cipriani et al. 2018 network meta-analysis.

## Project Status

**v0.1.0-dev** — Phase 1 (core extraction pipeline) in progress.

## License

MIT
