# Error Analysis: From Failures to Verified Fixes

Goal: isolate the single root cause explaining the largest share of metric loss, then verify the fix by measurement.

## Procedure

1. **Save traces:** Record input, intermediate stage outputs, final output, reference, and slice metadata (`lang`, `direction`, `source`, `length`, `kind`) for every eval item.
2. **Sample failures:** Draw 50–100 random failures per slice (heavier on the worst slice) — never take the first $N$ rows in file order.
3. **Open coding:** Label each failure with a short plain-language cause; after 30–50 items, consolidate into 5–12 mutually exclusive taxonomy categories.
4. **Quantify:** Tag the full sample, compute category shares with binomial CIs, and convert each share into potential metric headroom.
5. **Formulate hypothesis:** Write a falsifiable statement (e.g., *"Kazakh inflected mentions miss exact alias lookup → `kk` candidate recall trails `ru` by X pp"*) and confirm on a targeted slice or synthetic probe.
6. **Intervene & verify:** State expected effect before running; validate on dev, then confirm on frozen test via the promotion gate.
7. **Lock in regression:** Add fixed cases to the component's challenge/behavioral test suite (`trifact-mlops`).

## Starter Taxonomies

| Component | Starter Categories |
|---|---|
| Documents | missing text, reading order, merge/split, columnar table, heading, encoding, OCR substitution, header/footer leak |
| NER | boundary, type (metonymy), rare-entity miss, sentence-initial false positive, nesting, transliteration, gold label error |
| Linking | absent from KB, candidate miss (alias/suffix), ranker error (prior bias), NIL threshold error |
| Retrieval | missing from index, chunk split, representation miss, fusion loss, ANN loss, reranker demotion, unjudged relevant |
| Generation | ignored evidence, wrong citation, wrong language, false refusal, missed refusal, incomplete answer, unsupported claim |
| Verification | long premise, number/date error, cross-lingual pair, compound claim, reference error |

## Systematic Bias & CheckList Probes

- Cross-tabulate `category × slice`: skewed counts expose systematic defects (e.g., wrong language concentrated in `kk → ru`).
- Run cheap behavioral tests:
  - **Invariance:** transliteration/casing of names, passage reordering, query paraphrase.
  - **Directional:** adding a contradicting passage triggers caveat; removing sole evidence triggers refusal.
  - **Minimum functionality:** simple template facts in `kk`/`ru`/`en`.

## Required Record

Log taxonomy, category shares + CIs per slice, example IDs, root-cause hypothesis, and intervention outcome (including negative results) in MLflow and `PROGRESS.md`.
