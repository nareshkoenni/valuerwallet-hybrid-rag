# Evaluation

## Required leakage controls

- Build the retrieval query from the user-visible query alone.
- Keep the ground-truth target in a separate label field used only after retrieval.
- Do not choose an exact route based on a target identifier taken from the label.
- If a task explicitly supplies an identifier, label it as identifier lookup and evaluate it separately from historical property matching.
- Compare baselines using equivalent query information.
- Record benchmark version, model/version, configuration, seeds, and metric definitions.
- Recompute aggregate metrics from per-query outputs.
- Inspect all outputs for real file numbers, addresses, personal information, and private paths before release.

The original experiment CSVs and manifests are intentionally excluded from this public starter because they require protocol and privacy review.
