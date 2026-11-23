# 8. Evidence Verification and Refusal Policy

Date: 2026-11-18

## Status

Accepted

## Context

Even grounded RAG models hallucinate facts or cite non-supporting passages.

## Decision

Train the LLM to cite `[chunk_id]` after each fact and refuse when evidence is absent. Post-generation, split answers into claims, validate citation IDs, and run NLI (`supported` / `contradicted` / `not_enough_info`). Remove `contradicted` claims; mark answers as partially verified when NEI share exceeds threshold; refuse if zero claims are supported.

## Consequences

Adds ~0.3 s latency per answer; verdicts are exposed to users and logged as online quality metrics.
