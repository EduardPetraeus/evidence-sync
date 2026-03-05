# Methodology

This document describes the statistical methods, data extraction processes, and quality assessment approaches used by Evidence Sync.

## Statistical Methods

Evidence Sync uses the **DerSimonian-Laird (DL) random-effects model** for pooling study results. This is the most widely used approach in meta-analysis and assumes that the true treatment effects vary across studies due to differences in populations, interventions, and settings (clinical heterogeneity).

The DL method estimates between-study variance (tau-squared) using a method-of-moments estimator. The pooled effect is computed as a weighted average where each study's weight is the inverse of its total variance (within-study variance plus the between-study variance estimate). This down-weights studies that are less precise while also accounting for genuine variation between studies.

The pooled estimate is accompanied by a 95% confidence interval and a two-tailed p-value derived from a Wald-type z-test. For ratio measures (odds ratio, risk ratio, hazard ratio), the null value is 1.0; for difference measures (mean difference, standardized mean difference), the null is 0.0.

## Heterogeneity Metrics

Three complementary metrics quantify how much the individual study results differ from each other:

**I-squared** describes the percentage of total variation across studies that is due to heterogeneity rather than sampling error. Values of roughly 25%, 50%, and 75% are conventionally interpreted as low, moderate, and high heterogeneity. An I-squared of 0% means all variation is consistent with chance alone.

**Cochran's Q statistic** tests the null hypothesis that all studies share a common true effect. It follows an approximate chi-squared distribution with k-1 degrees of freedom (k = number of studies). A significant Q (p < 0.10 by convention) indicates the presence of heterogeneity. However, Q has low power when the number of studies is small.

**Tau-squared** is the estimated between-study variance on the effect-size scale. Unlike I-squared, it is an absolute measure of dispersion: a tau-squared of 0.05 on a log-odds-ratio scale means the true effects are spread around the pooled estimate with that variance. Evidence Sync reports tau-squared alongside I-squared so users can gauge both relative and absolute heterogeneity.

## Publication Bias

Evidence Sync assesses publication bias through **funnel plots** and **Egger's regression test**. The funnel plot displays each study's effect size (x-axis) against its standard error (y-axis, inverted). In the absence of bias, studies should scatter symmetrically around the pooled effect, with imprecise (high SE) studies spreading wider.

Egger's test formalizes asymmetry detection by regressing the standardized effect (effect / SE) on precision (1 / SE). A statistically significant intercept (p < 0.05) suggests the funnel is asymmetric, which may indicate publication bias, though other causes (such as genuine heterogeneity linked to study size) are possible. The test requires at least three studies to run.

Users should interpret funnel plot asymmetry cautiously. Small numbers of studies, substantial heterogeneity, or differences in study quality can all produce apparent asymmetry without true publication bias.

## Drift Detection

Evidence Sync monitors the stability of meta-analytic conclusions over time by comparing successive analysis snapshots. Three configurable thresholds trigger alerts:

- **Effect size change**: If the pooled effect changes by more than X% (default 10%) relative to the previous analysis, an alert is raised.
- **Significance flip**: If the conclusion changes from statistically significant to non-significant (or vice versa) at the 0.05 level, an alert fires.
- **Heterogeneity change**: If I-squared changes by more than Y percentage points (default 15pp), an alert is raised.

These thresholds are configurable per topic in `config.yaml`. When any threshold is exceeded, the system reports which conditions triggered the alert. This enables researchers to detect when new evidence materially alters the meta-analytic conclusion.

## Data Extraction

Evidence Sync uses Claude (Anthropic's LLM) to extract structured data from PubMed abstracts. The extraction prompt asks the model to identify sample sizes, effect sizes, confidence intervals, p-values, study design, and risk of bias assessments from the abstract text.

The extractor returns a structured JSON object for each study. Extracted fields undergo input validation: numeric values are type-checked and range-clamped, enum fields (effect measure, study design, bias risk) are validated against allowed values, and confidence scores are clamped to [0, 1]. If the model cannot determine a field from the abstract, it returns null, and the field remains unpopulated.

When both intention-to-treat (ITT) and per-protocol results are reported, the extractor is instructed to prefer the ITT analysis. For studies reporting multiple outcomes, the primary outcome is extracted.

## Quality Assessment

Risk of bias is assessed across six domains aligned with a simplified version of the Cochrane Risk of Bias tool (RoB 2.0):

1. **Random sequence generation** — Was the randomization sequence adequately generated?
2. **Allocation concealment** — Was the allocation sequence concealed from those enrolling participants?
3. **Blinding of participants and personnel** — Were participants and study personnel blinded?
4. **Blinding of outcome assessment** — Were outcome assessors blinded?
5. **Incomplete outcome data** — Were losses to follow-up balanced and handled appropriately?
6. **Selective reporting** — Are there signs that outcomes were selectively reported?

Each domain is rated as "low", "unclear", or "high" risk. The overall study rating is "high" if any domain is rated high, "low" if all domains are low, and "unclear" otherwise. These ratings are extracted from the abstract by the LLM and should be considered preliminary — full risk of bias assessment ideally requires access to the complete study protocol and full-text article.

## Limitations

- **Abstract-only extraction**: By default, Evidence Sync extracts data from PubMed abstracts rather than full-text articles. Abstracts may omit key details, leading to incomplete or less reliable extraction. Effect sizes not reported in the abstract cannot be captured.
- **LLM accuracy**: While Claude performs well at structured data extraction, it is not infallible. Extraction confidence scores are provided but should not substitute for human verification of critical results. Hallucinated values are possible, particularly for numerical data not explicitly stated in the abstract.
- **Pairwise meta-analysis only**: The current engine supports only pairwise (two-arm) comparisons. Network meta-analysis, which can simultaneously compare multiple treatments, is not yet implemented.
- **Single effect measure per topic**: Each review topic uses one effect measure (e.g., odds ratio). Studies reporting different measures require manual harmonization or separate topics.
- **DerSimonian-Laird limitations**: The DL estimator can underestimate tau-squared when the number of studies is small, leading to confidence intervals that are too narrow. More robust alternatives (REML, Knapp-Hartung) exist but are not yet implemented.
- **No subgroup or sensitivity analysis**: The current version does not support automatic subgroup analysis, leave-one-out sensitivity analysis, or meta-regression. These are planned for future releases.
