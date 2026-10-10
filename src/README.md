# Source code

Only add code that has been reviewed for confidentiality and verified against the experiment it claims to reproduce.

The previously inspected `metadata_retriever.py` uses TF-IDF over attachment/proof metadata and references a private SQLite database. It is not the dense semantic or query-aware hybrid implementation, so it is intentionally not copied here as if it were the full Paper 2 method.

Before publishing an implementation:
1. Replace real identifiers with synthetic examples.
2. Remove credentials, personal data, absolute local paths, and private records.
3. Ensure ground-truth labels are used only for scoring, not for query construction or routing.
4. Document inputs, configuration, dependencies, and expected outputs.
