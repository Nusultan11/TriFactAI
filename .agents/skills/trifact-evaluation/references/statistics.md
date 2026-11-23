# Statistical Comparison of Models

Resample count and CI level: `params.yaml: evaluation`. Reference implementation: `trifact.evaluation.eval_docproc.bootstrap_ci`.

## Paired Bootstrap of Differences

Given systems A and B on identical examples:
1. For $b = 1 \dots B$: draw $n$ indices with replacement; compute $\Delta_b = M(B, \text{idx}) - M(A, \text{idx})$ on the **same** indices.
2. Compute the CI from percentiles of $\Delta_b$ (e.g., 2.5% and 97.5%).
3. Significant gain holds when the entire CI > 0 (or > gate minimum gain).
4. Recompute nonlinear corpus metrics (entity F1, macro F1, nDCG) from scratch on each resample rather than averaging per-item scores.

## Cluster Bootstrap

When examples are dependent, resample **clusters**, not individual rows:
- Multiple questions sharing a context/document → cluster by `doc_key`.
- Translations of one query across 9 retrieval directions → cluster by source query ID.
- Sentences from the same NER document → cluster by `doc_id`.

Ignoring clusters produces artificially narrow CIs.

## Multi-Seed Variance

Run ≥3 seeds for trained models. On each bootstrap resample, draw one random seed per system and compute $\Delta_b$ so the CI captures both data and training variance. Report seed `mean ± std` and difference CI.

## Permutation Test

For small sets or exact p-values, randomly swap A and B outputs per cluster and compute the fraction of permutations with $\Delta \ge \Delta_{\text{obs}}$.

## Multiple Comparisons

Testing 9 directions × multiple metrics inflates false positives:
- Follow the pre-specified gate rule (e.g., no direction worse, mean better) when defined.
- Otherwise apply Holm–Bonferroni across the test family.
- Publish every CI, not only statistically significant ones.

## Power & Sample Size

- For proportion $p$ on $n$ items, 95% CI half-width $\approx 1.96 \sqrt{p(1-p)/n}$ ($p=0.8, n=300 \Rightarrow \pm 4.5$ pp; $n=1000 \Rightarrow \pm 2.5$ pp).
- Size human review sheets to achieve the required CI width before labeling.
- Paired evaluation sharply narrows difference CIs because system errors correlate.

## Aggregation

- Use **macro** across languages/directions for fairness decisions; **micro** for aggregate volume.
- Never compare averages across mismatched example sets; intersect IDs first.
