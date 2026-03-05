# Evidence Sync — CLAUDE.md

## What This Is
Living meta-analysis engine. Autonomous pipeline that monitors PubMed, extracts study data via Claude, and maintains always-up-to-date meta-analyses.

## Stack
- Python 3.11+, Click CLI, httpx, scipy, matplotlib, anthropic SDK
- PubMed E-utilities API for study discovery
- Claude API for structured data extraction from abstracts
- Git-based dataset versioning (YAML per study)

## Code Conventions
- `snake_case` for variables/functions/modules
- `kebab-case` for file/directory names in datasets
- `PascalCase` for classes
- All code, comments, docstrings in English
- Type hints on all public functions
- Dataclasses for data models (not Pydantic)

## Project Structure
```
src/evidence_sync/
├── cli.py          # Click CLI entry point
├── config.py       # YAML config loading
├── models.py       # Core dataclasses (Study, ReviewConfig, AnalysisResult)
├── monitor.py      # PubMed API search + dedup
├── extractor.py    # Claude-powered data extraction
├── statistics.py   # Meta-analysis engine (DerSimonian-Laird)
├── drift.py        # Evidence drift detection + alerts
├── versioning.py   # Git-based dataset management
└── dashboard.py    # Visualization (forest plots, funnel plots)
```

## Key Design Decisions
- YAML for study data (human-readable, git-diff-friendly)
- One file per study in `datasets/<topic>/studies/`
- Random-effects model (DerSimonian-Laird) as default
- Abstract-only extraction as fallback when full text unavailable

## Testing
```bash
pytest                    # Run all tests
pytest -x                 # Stop on first failure
pytest tests/test_statistics.py  # Run specific module
```

## CLI
```bash
evidence-sync init <topic>    # Initialize new review topic
evidence-sync search <topic>  # Search PubMed for new studies
evidence-sync extract <topic> # Extract data from found studies
evidence-sync analyze <topic> # Run meta-analysis
evidence-sync run <topic>     # Full pipeline (search → extract → analyze)
evidence-sync report <topic>  # Generate report with forest plot
```

## Pilot Topic
SSRI vs placebo for major depressive disorder (validates against Cipriani et al. 2018).
