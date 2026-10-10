# ValuerWallet Hybrid RAG — Research Code

This repository documents research on information-need-aware retrieval for property-valuation workflows.

## Release status

**Initial public release scaffold. This is not yet a complete reproduction package for the Paper 2 results.**

The research materials inspected during preparation include:
- a preliminary structured-SQL benchmark script;
- a TF-IDF retrieval baseline over attachment/proof metadata;
- separate Colab artifacts recording dense, exact-identifier, hybrid, and identifier-hidden experiments.

The original Paper 2 Colab notebook requires an evaluation audit. In the inspected notebook, ground-truth file identifiers are inserted into some query templates and are also used by exact/hybrid routing. For tasks where the identifier is meant to be unknown, this can leak the target into the retrieval input. Therefore, the recorded exact and aggregate hybrid scores must not be represented as validated, unbiased retrieval performance until the protocol is corrected and the experiments are rerun.

## Repository layout

- `src/`: reviewed starter code and guidance for implementation.
- `evaluation/`: evaluation protocol and leakage checks.
- `docs/data_availability.md`: data privacy and access limitations.
- `docs/reproducibility.md`: current status and steps needed for reproduction.
- `requirements.txt`: dependencies for the preliminary metadata baseline.

## Data availability and privacy

The underlying ValuerWallet database, valuation files, physical documents, photographs, private file mappings, and raw operational records are proprietary and are not included. Do not commit them.

## What this release does and does not provide

This release provides documentation and a small, generic retrieval/evaluation scaffold. It does **not** include the proprietary dataset, embeddings, private benchmark outputs, or a verified implementation capable of reproducing all reported Paper 2 metrics. Do not cite this repository as a complete reproduction package until the remaining experiments have been audited and made reproducible.
