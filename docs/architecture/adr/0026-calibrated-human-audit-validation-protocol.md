# ADR-0026: Autonomous Machine-Assisted Validation with Report-Level Human Governance

- **Status:** Accepted (report-level governance approved 2026-11-25)
- **Date:** 2026-11-25
- **Amends:** [ADR-0007](0007-leakage-safe-splits.md), [ADR-0015](0015-silver-and-synthetic-data.md), [ADR-0022](0022-evaluation-protocol.md)

## Context

Stages 2 and 3 originally required manual item-by-item human annotation of **5,600** validation items (`1,500` curated corpus quality samples, `600` silver NER sentences, and `3,500` MT-derived retrieval/NLI evaluation items) before formal closure. For a single developer, manual item-level labeling of hundreds or thousands of multilingual samples is operationally impractical and blocks Stage 5 model development.

At the same time, scientific integrity requires that machine-generated judgments **never** be misrepresented as human verification or human-calibrated labels (ADR-0022), and that all statistical limitations, model disagreements, and residual risks be transparently documented and propagated to downstream evaluation slices.

## Decision

Adopt an **Autonomous Machine-Assisted Validation Protocol with Report-Level Human Governance** for Stage 2 curation, Stage 3 silver NER, and Stage 3 MT-derived evaluation sets:

1. **Dual-Signal Automated Cross-Validation (100% of all 5,600 items, `human: false`):**
   - Every item across all three validation packets (`1,500` quality, `600` silver NER, `3,500` eval MT/NLI) must have two independent, pinned machine assessments:
     - **Teacher LLM judge:** `Qwen/Qwen3.5-35B-A3B` (`UD-IQ4_XS`, `llama.cpp`, `temperature=0`) with deterministic JSON recovery (handling length-truncated outputs and self-revised JSON blocks).
     - **Independent non-LLM signals:** `GlotLID` + `lid.176` + Kazakh-letter share (quality); `GLiNER multi v2.1` zero-shot spans (silver NER); `BGE-M3` dense cosine + `NLLB` round-trip `chrF` + digit preservation + `lid.176` + `mDeBERTa-xnli` label consistency (eval MT & NLI).
   - **Deterministic Defect Resolution & Schema Sanitization:**
     - Quality: override `good` → `bad (wrong_language)` when both independent language detectors (`GlotLID` and `lid.176`) agree at `>= 0.85` on a non-target language.
     - Silver NER: enforce ADR-0006 core schema (`PER`/`ORG`/`LOC`), drop non-core types (`EVENT`/`WORK`/`DATE`), reclassify non-core `wrong_type` targets into `not_entity`, remove already-marked `missed` hallucinations, remove `kk`/`ru` case-suffix stripping false positives, remove whitespace-only diffs, and prevent double-counting spans.
     - Eval MT & NLI: normalize no-op and trivial punctuation/casing/`ё`-only `fix` suggestions to `ok`; flag digit-dropping fixes and cross-signal disagreements.
   - **Uncertainty & Disagreement Tracking:** Every item carries deterministic `uncertainty` (`high` / `medium` / `low`), `machine_agree`, and `uncertainty_reasons`.

2. **Strict Provenance Separation (Zero Misrepresentation):**
   - Machine-validated results are **always** marked `human: false` and `verification_tier: "machine_cross_validated"`.
   - Human-verified counters (`human_verified`) increase **only** for genuinely human-annotated rows (e.g., gold NER, gold QA, gold `retrieval`/`nli` rows) and remain `0` for machine-validated MT rows, quality audit docs, and silver NER sentences.
   - **Report-Level Human Governance:** The project owner reviews and approves the consolidated Final Validation Report (`reports/stage-01-03/final_validation_report.md`). This governance approval authorizes freezing `eval/retrieval` and `eval/nli` under `build_eval freeze --machine-validated` and closing Stages 1–3, and is recorded separately as `report_governance_approved: true` — **never** as item-level human annotation or human calibration.

3. **Eval Freezing & Downstream Slicing:**
   - `build_eval freeze --machine-validated` freezes `eval/retrieval` and `eval/nli` once all `3,500` MT items have complete cross-validated assessments (`machine_labels.csv`), applying machine verdicts (`ok`/`fix`/`drop`), dropping unvalidated MT rows outside the 3,500 packet, adding `cluster_id` (ADR-0022), tagging every row with `verification_tier` (`"human_verified"` for gold rows, `"machine_cross_validated"` for MT rows), and recording SHA-256 shard hashes in `frozen.json`.
   - Stage 6 (Retrieval) and Stage 8 (Verification) evaluations **must** report metrics separately on `human_verified` (gold) vs. `machine_cross_validated` (MT) slices so any uncalibrated MT/judge bias remains visible.

## Consequences

- **Positive:** Eliminates the solo-developer annotation bottleneck while maintaining full reproducibility, dual-signal cross-validation, deterministic defect remediation, and transparent reporting.
- **Explicit Limitation:** Because no item-level human calibration sample was annotated, LLM-judge bias (especially on `kk` morphology and NLI fluency) is **uncalibrated** ($\kappa$ vs. human is unknown). Downstream promotion gates (ADR-0009, ADR-0022) must treat `machine_cross_validated` MT slices as secondary to `human_verified` gold slices and non-Wikipedia / `real_docs` suites.
- **Reversibility:** All 5,600 raw sheets and machine assessments remain archived in DVC and Drive; item-level human annotation can be layered on at any future time without rebuilding upstream corpora.
