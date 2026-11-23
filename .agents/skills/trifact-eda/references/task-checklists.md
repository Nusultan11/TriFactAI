# Task-Specific EDA Checklists

Format: **what to measure → decision it informs**. Slice every metric by `language × source × split × origin`.

## Corpora (`raw`, `clean`, `curated`)

- Quality-filter signals before/after filtering, signal correlations, agreement with human review → `curation.quality` thresholds.
- Boilerplate: high-frequency lines per source (nav, cookie banners, templates) → boilerplate list, duplicate-line filter.
- Vocabulary: frequency spectrum, hapax ratio, Zipf slope, top n-grams → detect machine/template text.
- Domains & time: host, TLD, publish dates → reweighting, temporal slice for `real_docs`.
- Intra-document language mixing (`para_lang`) → paragraph filtering or code-switch handling.
- Wikidata `entities`: `kk`/`ru`/`en` label & alias coverage, sitelinks, type distribution → cross-lingual `kk` linking strategy.

## Documents & OCR (`data/eval/docproc`, `real_docs`, uploads)

- CER/WER by `kind × lang`, tail quantiles (p90) → routing and `docproc.router` thresholds.
- Kazakh letter retention (`ә ғ қ ң ө ұ ү һ і`) vs. reference and confusion matrix (`қ→к`, `ң→н`, `ү→у`) → OCR engine selection.
- Share of pages with no text layer, broken encoding, tables, multi-column layout → Docling/OCR load and latency budget.
- `ParseError` reason distribution → `docproc.limits` and parser fixes.
- Page count and char length tails → timeouts and RAM limits.

## NER

- Label distribution, entity density per sentence, span token length → `max_length`, class handling.
- BIO validity (`I-` without `B-`, type switch mid-span), nested/overlapping spans → conversion rules.
- Consistency: same surface with multiple types in one source (ORG/LOC metonymy), gazetteer agreement → noise estimate, silver filter.
- Kazakh suffixes: share of entities with case/possessive endings inside the token (`Астанада`, `Абайдың`) → alias normalization for linking.
- Silver vs. gold: type confusion on shared surfaces, missed-entity rate (unlabeled capitalized words) → `ner.silver.*`, curriculum.
- Entity frequency: share of eval surfaces unseen in train → `unseen-F1` slice.

## Entity Linking & KB

- Ambiguity: QIDs per normalized alias by language, prior distribution → ranker features, NIL threshold.
- Coverage: share of gold mentions whose QID is in KB and whose alias matches exact lookup → candidate recall ceiling.
- Popularity: sitelink distribution of target entities → popularity bias audit.

## Retrieval

- Positives per query; query and passage lengths in embedder tokens → `max_length`, chunking.
- BM25 rank of positive among negatives (easiness) and negative/positive score ratio (hardness) → hard-negative mining.
- Query–passage lexical overlap per direction → expected sparse-leg gain.
- False-negative rate: manual audit of 100 top mined negatives → cross-encoder filtering threshold.
- Volume across 9 language directions and origins (human, MT, synthetic, parallel sentences) → sampling weights.

## QA & SFT

- Question, context, and answer lengths; answer position in context → context budget, lost-in-the-middle risk.
- Unanswerable and cross-lingual shares → compliance with `sft.mix`.
- Question types (`who`/`when`/`how many`), question–context overlap → shortcut audit.
- Citations: count per answer, ID validity, answer language vs. requested → synthetic filters.
- SFT ↔ eval question overlap → leakage removal.

## NLI

- Class balance by language/origin; premise & hypothesis token lengths → `max_length`, windowing.
- Hypothesis-only baseline and lexical cues (negation) → hard evaluation subset.
- MT label consistency (with caveat that judge was trained on XNLI) → `kk` filtering.

## Synthetic Data

- Filter funnel: drop rate per filter and language → bottleneck identification.
- Diversity: distinct-1/2/3, top opening n-grams, template share → prompt/temperature tuning.
- Topic coverage: source chunk domain/length distribution vs. curated corpus → chunk sampling.
- Synthetic vs. human eval shift (lengths, question types, lexical overlap) → mix adjustment.

## Train vs. Eval

- Length JS-divergence, classifier two-sample test, entity/question/doc/near-dup overlap → leakage fix or explicit shift slice.
