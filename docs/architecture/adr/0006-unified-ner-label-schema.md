# 6. Unified NER Label Schema with Kazakh Extension

Date: 2026-11-18

## Status

Accepted

## Context

KazNERD defines 25 classes whereas WikiANN defines only `PER`/`ORG`/`LOC` and lacks `DATE`. Merging them directly produces conflicting labels and leaves `ru`/`en` without date supervision.

## Decision

Train the neural NER model on a unified `PER, ORG, LOC` core across KK/RU/EN (KazNERD class mapping in `configs/labels.yaml`). Extract `DATE` via a dedicated rule-based extractor (regex + `dateparser` for `kk`/`ru`/`en`), evaluated on KazNERD `DATE` spans. Treat full `KazNERD-25` as a separate experiment (`NER-4`).

## Consequences

Metrics remain directly comparable across languages without blocking v1 on fine-grained taxonomies.
