# Improvement Report — G3 text corpus builder

Date: 2026-08-12
Owner: RAG ingestion/retrieval
Delivery-gate impact: G3 preparation only; no quality gate is advanced.

## Change

Added `scripts/build_text_index.py`, which builds a BM25 index from a
versioned JSON corpus manifest and checked-in UTF-8 text/Markdown files. It
uses the existing chunker, docstore format, tokenizer, and BM25 retriever, then
writes a build manifest with source and output digests.

## Intended effect

G3 can bind the candidate to an auditable, task-independent engineering corpus
without converting Markdown to an opaque PDF or committing an index artifact.

## Compatibility and rollback

The runtime PDF ingestion queue and HTTP API are unchanged. The builder is an
optional offline utility; removing it affects no existing runtime path.

## Verification

- A new test builds an index from a temporary Markdown corpus and retrieves it
  through the real `IndexRegistry` and `BM25Retriever`.
- Full RAG suite is required before merge.

## Evidence boundary

The builder proves index reproducibility, not RAG quality improvement. G3
results remain pending authorized Agent/model execution.
