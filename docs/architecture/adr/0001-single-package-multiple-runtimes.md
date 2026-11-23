# 1. Single Python Package, Multiple Runtimes

Date: 2026-11-18

## Status

Accepted

## Context

The platform requires an API, background ingestion worker, model inference service, and offline DVC stages. Separate microservice repositories add unnecessary overhead for a solo engineer, while services have distinct compute profiles (GPU inference vs. CPU API).

## Decision

Maintain one package `src/trifact` organized by domain (`ingestion`, `extraction`, `retrieval`, `generation`, `verification`) with separate entry points (`trifact.api`, `trifact.worker`, `trifact.inference`, `python -m trifact.data.*`). Each service runs in its own Docker container built from a shared image.

## Consequences

Shared schemas and logic avoid duplication while allowing independent scaling and restarts. Module boundaries are enforced by import discipline (`trifact.core` depends on nothing else), verified by architectural tests.
