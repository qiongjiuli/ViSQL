"""Retrieval modules — schema linking + Spider few-shot exemplars.

Two retrievers, both using sentence-transformers MiniLM-L6-v2 + FAISS:

  1. SchemaEmbedder: per-table embeddings, top-K table linker (slide 3 stage 2).
  2. ExemplarRetriever: Spider 1.0 (NL, SQL) pairs, top-3 exemplar lookup
     for SQL agent's few-shot prompting (slide 7).
"""
from __future__ import annotations
from typing import Optional, Iterable
from pathlib import Path
import json
import numpy as np

from . import config as cfg
from .schemas import DatabaseSchema, SchemaTable

# Singleton embedder — shared across retrievers
_EMBEDDER = None

def _load_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        print(f"[retrievers] loading embedder {cfg.EMBED_MODEL}")
        _EMBEDDER = SentenceTransformer(cfg.EMBED_MODEL, device=cfg.DEVICE)
    return _EMBEDDER

# ════════════════════════════════════════════════════════════════════
# SCHEMA LINKING
# ════════════════════════════════════════════════════════════════════
class SchemaEmbedder:
    """Embed each table as a single text and FAISS-index them.

    A table's embedding text concatenates: table name, description, column
    names + types, and (truncated) sample row values. This puts both
    schema-level and value-level signal into the index.
    """

    def __init__(self, schema: DatabaseSchema):
        self.schema = schema
        self.embedder = _load_embedder()
        self._build_index()

    def _table_text(self, t: SchemaTable) -> str:
        cols = ", ".join(f"{c.name} ({c.type})" for c in t.columns)
        sample = ""
        if t.sample_rows:
            sample = " values: " + " ".join(
                str(v)[:30] for row in t.sample_rows[:1]
                for v in row.values()
            )[:200]
        return f"{t.name}: {t.description} columns: {cols}{sample}"

    def _build_index(self) -> None:
        import faiss
        texts = [self._table_text(t) for t in self.schema.tables]
        embs = self.embedder.encode(texts, normalize_embeddings=True)
        embs = np.asarray(embs, dtype="float32")
        self._index = faiss.IndexFlatIP(embs.shape[1])  # inner product on normalized vecs = cosine
        self._index.add(embs)
        self._table_texts = texts

    def link(self, question: str, k: int = cfg.TOP_K_TABLES) -> list[SchemaTable]:
        """Return top-k tables most relevant to the question."""
        q_emb = self.embedder.encode([question], normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype="float32")
        _, idx = self._index.search(q_emb, min(k, len(self.schema.tables)))
        return [self.schema.tables[i] for i in idx[0]]

    def link_to_schema(self, question: str, k: int = cfg.TOP_K_TABLES) -> DatabaseSchema:
        """Convenience: return a sub-schema with only the linked tables."""
        linked = self.link(question, k=k)
        return DatabaseSchema(
            db_id=self.schema.db_id,
            description=self.schema.description,
            tables=linked,
            bq_project=self.schema.bq_project,
            bq_dataset=self.schema.bq_dataset,
        )

# ════════════════════════════════════════════════════════════════════
# EXEMPLAR RETRIEVAL (Spider 1.0 — slide 7: 7K pairs, k=3)
# ════════════════════════════════════════════════════════════════════
class ExemplarRetriever:
    """Retrieve k most similar (NL, SQL) Spider exemplars for few-shot prompting."""

    def __init__(self, exemplars: list[dict]):
        """Args:
            exemplars: list of {'question': str, 'sql': str, 'db_id': str}
        """
        self.exemplars = exemplars
        self.embedder = _load_embedder()
        self._build_index()

    def _build_index(self) -> None:
        import faiss
        texts = [ex["question"] for ex in self.exemplars]
        embs = self.embedder.encode(texts, normalize_embeddings=True,
                                     show_progress_bar=False, batch_size=64)
        embs = np.asarray(embs, dtype="float32")
        self._index = faiss.IndexFlatIP(embs.shape[1])
        self._index.add(embs)

    def retrieve(self, question: str, k: int = cfg.TOP_K_EXEMPLARS) -> list[dict]:
        q_emb = self.embedder.encode([question], normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype="float32")
        _, idx = self._index.search(q_emb, min(k, len(self.exemplars)))
        return [self.exemplars[i] for i in idx[0]]

    def format_for_prompt(self, question: str, k: int = cfg.TOP_K_EXEMPLARS) -> str:
        examples = self.retrieve(question, k=k)
        blocks = []
        for i, ex in enumerate(examples, 1):
            blocks.append(
                f"Example {i}:\n"
                f"Question: {ex['question']}\n"
                f"SQL: {ex['sql']}\n"
            )
        return "\n".join(blocks)

# ════════════════════════════════════════════════════════════════════
# SPIDER LOADING
# ════════════════════════════════════════════════════════════════════
def build_spider_exemplars(
    spider_train_path: str | Path,
    max_examples: int = 7000,
) -> list[dict]:
    """Load Spider 1.0 train as a list of {question, sql, db_id} dicts.

    Pass the path to spider/train_spider.json. Returns ~7000 exemplars.
    """
    path = Path(spider_train_path)
    with open(path) as f:
        data = json.load(f)

    out: list[dict] = []
    for ex in data[:max_examples]:
        q = ex.get("question", "").strip()
        sql = ex.get("query", "").strip()
        db = ex.get("db_id", "")
        if q and sql:
            out.append({"question": q, "sql": sql, "db_id": db})
    print(f"[exemplars] loaded {len(out)} (NL, SQL) pairs from Spider")
    return out
