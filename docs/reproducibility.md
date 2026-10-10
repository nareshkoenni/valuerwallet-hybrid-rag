# Reproducibility status

## Current status

This is not yet a complete reproduction package.

The inspected implementation archive supports structured SQL reference outputs and TF-IDF retrieval over attachment/proof metadata. Its own documentation states that full document-text RAG requires the separate physical corpus and a fixed model/embedding configuration.

Separate Colab artifacts record dense semantic, exact file-number, hybrid, and identifier-hidden experiments. However, the evaluation protocol needs correction because ground-truth file identifiers are used in some query templates and exact/hybrid routing. This risks target leakage.

## Required before claiming reproducibility

1. Correct query construction so unknown target identifiers are not included in retrieval inputs.
2. Use ground-truth labels exclusively in the scoring layer.
3. Rerun baselines with comparable inputs and frozen configurations.
4. Record package versions, model identifiers, seeds, benchmark version, and metric definitions.
5. Verify aggregate results from query-level outputs.
6. Run a clean reproduction from documented inputs.
7. Audit every public artifact for private identifiers and personal data.

Until these steps are complete, existing exact/hybrid metrics should be described as recorded preliminary results, not validated unbiased retrieval performance.
