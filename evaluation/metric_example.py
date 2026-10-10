"""Example use of the generic ranking metrics; uses synthetic IDs only."""
from src.retrieval_metrics import recall_at_k, reciprocal_rank

# Synthetic example. The target label is used only after the ranking is fixed.
ranked_results = ["synthetic-A", "synthetic-C", "synthetic-B"]
ground_truth = {"synthetic-B"}

print("Recall@3:", recall_at_k(ranked_results, ground_truth, 3))
print("MRR@3:", reciprocal_rank(ranked_results, ground_truth, 3))
