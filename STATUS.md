# Evidence Sync — STATUS

## Current Phase: Phase 3 — Automation, Drift Alerts, and GitHub Actions

### Status: COMPLETE

### Date: 2026-03-05

## Phase 3 — What Was Done

### GitHub Actions Workflows
- [x] `.github/workflows/ci.yml` — lint + test on push/PR to main (Python 3.12, ruff, pytest)
- [x] `.github/workflows/monitor.yml` — weekly PubMed monitoring cron (Monday 8am UTC) + manual dispatch

### Webhook Alert System
- [x] `send_alert()` in `drift.py` — POST JSON to configured webhook on drift detection
- [x] SSRF protection — webhook URL must use `https://` scheme
- [x] Email alerts logged as warning (actual sending out of scope for v0.1)
- [x] Graceful error handling (log, don't crash)

### Auto-versioning on Extract/Analyze
- [x] `commit_dataset_changes()` in `versioning.py` — git add + commit for dataset changes
- [x] Only commits when actual changes exist (checks `git status` first)
- [x] Conventional commit message: `data: update <topic_id> — N studies`
- [x] Does NOT auto-push (destructive action left to user)
- [x] `--auto-commit` flag on `extract` and `analyze` CLI commands (default: off)

### Tests (9 new tests)
- [x] test_automation.py — webhook payload structure, SSRF rejection, network error handling,
      email warning, git commit with changes, no-commit without changes, study count in message,
      non-git directory handling

## Phase 1 — Core Extraction Pipeline (COMPLETE)

### Project Setup
- [x] Git repo initialized with feature branch `feat/phase-1-core-pipeline`
- [x] pyproject.toml with all dependencies (pip-installable)
- [x] CLAUDE.md with project conventions
- [x] README.md with project overview
- [x] LICENSE (MIT)
- [x] Pilot dataset config (SSRI depression topic)
- [x] BRIEF.md in `~/build-logs/deep-research/active/evidence-sync/`

### Source Code (8 modules)
- [x] `models.py` — 7 dataclasses (Study, ReviewConfig, AnalysisResult, DriftResult, RiskOfBias, etc.)
- [x] `config.py` — YAML config loading/saving with topic_id validation
- [x] `monitor.py` — PubMed E-utilities search, XML parsing (defusedxml), deduplication
- [x] `extractor.py` — Claude-powered data extraction with input validation
- [x] `statistics.py` — DerSimonian-Laird random-effects meta-analysis, Egger's test
- [x] `drift.py` — Evidence drift detection with configurable thresholds + webhook alerts
- [x] `versioning.py` — YAML study serialization with PMID validation + git auto-commit
- [x] `dashboard.py` — Forest plot and funnel plot generation
- [x] `cli.py` — Click CLI (init, search, extract, analyze, report, run) with --auto-commit

### Tests (59 original + 9 new = 68 total)
- [x] test_models.py — 9 tests (properties, risk of bias logic)
- [x] test_config.py — 2 tests (roundtrip, defaults)
- [x] test_monitor.py — 7 tests (query building, XML parsing, dedup, search mock)
- [x] test_extractor.py — 8 tests (JSON parsing, data application, edge cases)
- [x] test_statistics.py — 11 tests (meta-analysis, heterogeneity, weights, edge cases)
- [x] test_drift.py — 6 tests (no drift, effect change, significance flip, multiple alerts)
- [x] test_versioning.py — 9 tests (study serialization, analysis serialization, loading)
- [x] test_automation.py — 9 tests (webhook alerts, SSRF, git auto-commit)
- [x] conftest.py — shared fixtures with 5 realistic SSRI trial studies

### Security Fixes Applied
- [x] **C1 (CRITICAL):** XXE protection — replaced `xml.etree.ElementTree` with `defusedxml`
- [x] **W1:** Path traversal — `topic_id` validated with regex `^[a-z0-9][a-z0-9\-]{0,63}$`
- [x] **W2:** Path traversal — PMID validated as numeric before use in file paths
- [x] **W3:** Prompt injection hardening — XML tag delimiters around user content in Claude prompts
- [x] **W3b:** Input validation — `_safe_int()`, `_safe_float()` for all extracted numeric fields
- [x] **W4:** SSRF protection — webhook URL scheme validated (https only)

## Validation Results

| Check | Result |
|-------|--------|
| `pip install -e ".[dev]"` | PASS |
| `pytest tests/ -v` | 68/68 PASS |
| `ruff check src/ tests/` | All checks passed |
| Security review | 1 CRITICAL + 4 WARNING fixed |

## Remaining (Future Phases)

### Phase 4: Dashboard + Polish
- Streamlit interactive dashboard
- CLI improvements
- Documentation

### Phase 5: Ship v0.1.0
- PyPI publication
- GitHub repo with CI/CD
- Demo dataset included

## Blocked Questions
None.
