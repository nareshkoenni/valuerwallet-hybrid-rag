"""Command-line interface.

    python -m vwrag.cli build-corpus    --db DB.sqlite --out WORKDIR [--strict]
    python -m vwrag.cli build-benchmark --db DB.sqlite --out WORKDIR [--strict]
    python -m vwrag.cli build-index     --corpus WORKDIR/rag_documents.csv --out WORKDIR
    python -m vwrag.cli evaluate        --workdir WORKDIR --out RESULTS_DIR

The proprietary database and every file derived from it stay in WORKDIR, which is
never part of the repository (see DATA_AVAILABILITY.md).
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .benchmark import (EXPECTED_V2_TOTAL, QUESTION_TEMPLATES, build_evaluation_set, build_pools,
                        check_pool_counts)
from .corpus import build_rag_corpus, build_unified, count_linked, load_tables
from .io_utils import sha256_file, write_json
from .retrieval import SYSTEMS, evaluate_systems, system_summaries

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_CORPUS_SIZE = 7573


def _connect(db: str) -> sqlite3.Connection:
    # Open read-only: the research copy of the database is never modified.
    return sqlite3.connect(f"file:{Path(db).resolve()}?mode=ro", uri=True)


def cmd_build_corpus(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with _connect(args.db) as conn:
        raw = {
            name: int(conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0])
            for name in ("inward_register", "inward_file_attachments", "PROOFS")
        }
        inward, attachments, proofs = load_tables(conn)
    unified = build_unified(inward, attachments, proofs)
    corpus = build_rag_corpus(unified)
    if args.strict:
        assert len(corpus) == EXPECTED_CORPUS_SIZE, len(corpus)
    assert corpus["file_number"].is_unique and corpus["rag_text"].str.strip().ne("").all()

    path = out / "rag_documents.csv"
    corpus.to_csv(path, index=False)
    write_json(out / "rag_corpus_fingerprint.json", {
        "artifact": "rag_documents.csv",
        "created_at": datetime.now().isoformat(),
        "shape": list(corpus.shape),
        "unique_file_numbers": int(corpus["file_number"].nunique()),
        "raw_inward_records": raw["inward_register"],
        "normalized_inward_files": len(inward),
        "raw_attachment_records": raw["inward_file_attachments"],
        "attachment_records_with_identifier": len(attachments),
        "blank_identifier_attachment_records_excluded": raw["inward_file_attachments"] - len(attachments),
        "attachment_records_linked_to_inward": count_linked(attachments, inward),
        "raw_proof_records": raw["PROOFS"],
        "proof_records_with_identifier": len(proofs),
        "blank_identifier_proof_records_excluded": raw["PROOFS"] - len(proofs),
        "proof_records_linked_to_inward": count_linked(proofs, inward),
        "rag_columns": list(corpus.columns),
        "sha256": sha256_file(path),
    })
    print(f"corpus: {corpus.shape}  sha256={sha256_file(path)}")


def cmd_build_benchmark(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with _connect(args.db) as conn:
        inward, attachments, proofs = load_tables(conn)
    unified = build_unified(inward, attachments, proofs)
    pools = build_pools(unified, attachments)
    if args.strict:
        check_pool_counts(pools)
    evaluation = build_evaluation_set(pools)
    if args.strict:
        assert len(evaluation) == EXPECTED_V2_TOTAL, len(evaluation)
    evaluation.to_csv(out / "evaluation_queries.csv", index=False)
    counts = {qid: int(len(s)) for qid, s in pools.items()}
    write_json(out / "benchmark_v2_manifest.json", {
        "benchmark_version": "reconstructed_v2",
        "created_at": datetime.now().isoformat(),
        "total_queries": int(len(evaluation)),
        "counts": counts,
        "question_templates": QUESTION_TEMPLATES,
    })
    print("pools:", counts, "total:", len(evaluation))


def _load_encoder(model_name: str):
    from sentence_transformers import SentenceTransformer  # imported lazily

    model = SentenceTransformer(model_name)

    def encode(texts):
        return model.encode(texts, batch_size=32, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False).astype("float32")

    return encode


def cmd_build_index(args) -> None:
    from sentence_transformers import SentenceTransformer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    corpus = pd.read_csv(args.corpus, dtype={"file_number": str})
    model = SentenceTransformer(args.model)
    texts = corpus["rag_text"].fillna("").astype(str).tolist()
    batches = [
        model.encode(texts[i:i + args.batch_size], batch_size=args.batch_size,
                     normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
        for i in range(0, len(texts), args.batch_size)
    ]
    emb = np.vstack(batches).astype(np.float32)
    assert emb.shape[0] == len(corpus) and np.allclose(np.linalg.norm(emb[:100], axis=1), 1.0, atol=1e-5)
    emb_path, map_path = out / "semantic_embeddings.npy", out / "semantic_file_mapping.csv"
    np.save(emb_path, emb)
    corpus[["file_number"]].to_csv(map_path, index=False)
    write_json(out / "semantic_index_manifest.json", {
        "model": args.model, "embedding_dimension": int(emb.shape[1]),
        "normalization": "normalize_embeddings=True", "batch_size": args.batch_size,
        "documents": int(emb.shape[0]), "embedding_dtype": str(emb.dtype),
        "corpus_sha256": sha256_file(args.corpus),
        "embedding_sha256": sha256_file(emb_path), "mapping_sha256": sha256_file(map_path),
    })
    print("index:", emb.shape)


def cmd_evaluate(args) -> None:
    work, out = Path(args.workdir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    evaluation = pd.read_csv(work / "evaluation_queries.csv", dtype=str)
    mapping = pd.read_csv(work / "semantic_file_mapping.csv", dtype=str)
    emb = np.load(work / "semantic_embeddings.npy").astype(np.float32)
    assert emb.shape[0] == len(mapping)
    per_query = evaluate_systems(evaluation, mapping["file_number"], emb, _load_encoder(args.model),
                                 k=args.top_k)
    # Per-query ranks contain no identifiers or text: they are safe to publish.
    per_query.to_csv(out / "per_query_ranks.csv", index=False)
    for system, table in system_summaries(per_query).items():
        table.to_csv(out / f"metrics_{system}.csv", index=False)
        overall = table[table["question_id"] == "ALL"].iloc[0]
        print(f"{system:30s} R@5={overall.recall_at_5:.4%} R@10={overall.recall_at_10:.4%} "
              f"R@20={overall.recall_at_20:.4%} MRR@20={overall.mrr_at_20:.4%}")
    write_json(out / "evaluation_manifest.json", {
        "package_version": __version__, "created_at": datetime.now().isoformat(),
        "model": args.model, "top_k": args.top_k, "systems": list(SYSTEMS),
        "total_queries": int(len(evaluation)),
        "exact_route_queries": int((per_query["route"] == "exact_file_number").sum()),
        "semantic_fallback_queries": int((per_query["route"] == "semantic_fallback").sum()),
        "evaluation_queries_sha256": sha256_file(work / "evaluation_queries.csv"),
        "embeddings_sha256": sha256_file(work / "semantic_embeddings.npy"),
    })


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="vwrag", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-corpus")
    p.add_argument("--db", required=True); p.add_argument("--out", required=True)
    p.add_argument("--strict", action="store_true", help="assert the published corpus size (7,573)")
    p.set_defaults(func=cmd_build_corpus)

    p = sub.add_parser("build-benchmark")
    p.add_argument("--db", required=True); p.add_argument("--out", required=True)
    p.add_argument("--strict", action="store_true", help="assert the published V2 pool sizes")
    p.set_defaults(func=cmd_build_benchmark)

    p = sub.add_parser("build-index")
    p.add_argument("--corpus", required=True); p.add_argument("--out", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL); p.add_argument("--batch-size", type=int, default=64)
    p.set_defaults(func=cmd_build_index)

    p = sub.add_parser("evaluate")
    p.add_argument("--workdir", required=True, help="directory holding evaluation_queries.csv and the index")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL); p.add_argument("--top-k", type=int, default=20)
    p.set_defaults(func=cmd_evaluate)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
