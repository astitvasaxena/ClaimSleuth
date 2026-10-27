"""
RAG retriever: embeds chunks, stores a local vector index, and returns the passages most
relevant to a question, with their source file and section.

Embeddings: sentence-transformers "all-MiniLM-L6-v2" (384-dim), loaded once per process
from the local Hugging Face cache, OFFLINE (no download, no network). If the model cannot
be loaded, a TF-IDF keyword retriever (scikit-learn) is used instead and every result is
labelled with the retriever that produced it.

Index: normalised vectors in a NumPy array + chunk metadata (JSONL) + a manifest, in
CLAIMSLEUTH_RAG_INDEX_DIR (default data/rag/index/, gitignored). Search is exact cosine
similarity (a dot product). FAISS is not used: it is not installed and, for a few dozen
chunks, an exact NumPy search is just as fast and easier to explain.

Safety: retrieved text is UNTRUSTED evidence. Every passage is scanned with the prompt-
injection guardrail; matching passages are quarantined (kept in the record, flagged, never
used as evidence). Passages below MIN_SCORE are dropped: no match means an empty result,
never an invented one.
"""
import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field

import configs.settings as settings
from src.core.guardrails import scan_injection
from src.rag.corpus import Chunk, chunk_claim_documents, load_policy_corpus

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Minimum cosine similarity for a passage to count as evidence. Chosen from measured scores on
# 24 development questions: correct top-1 passages scored min 0.20 / median 0.40; off-topic questions <= 0.11.
MIN_SCORE = {"embedding": 0.15, "tfidf": 0.05}
_LOCK = threading.Lock()
_STATE: Dict = {"embedder": None, "index": None}


class RetrievedPassage(BaseModel):
    chunk_id: str
    source_file: str
    doc_type: str
    section: str
    policy_id: Optional[str] = None
    claim_id: Optional[str] = None
    text: str
    score: float
    rank: int
    retriever: str
    untrusted: bool = True                    # always: retrieved text is evidence, never instructions
    quarantined: bool = False                 # True if it contains instruction-like text
    guardrail_hits: List[str] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    question: str
    passages: List[RetrievedPassage] = Field(default_factory=list)       # usable evidence only
    quarantined: List[RetrievedPassage] = Field(default_factory=list)
    retriever: str
    warnings: List[str] = Field(default_factory=list)


# ─── Embedders ──────────────────────────────────────────────────────────────

class EmbeddingEmbedder:
    name = "embedding"

    def __init__(self):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")          # never download at run time
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(MODEL_NAME, device="cpu", local_files_only=True)
        self.description = f"embedding:{MODEL_NAME}"

    def fit(self, texts: List[str]) -> None:
        pass

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
                          dtype=np.float32)


class TfidfEmbedder:
    """Fallback when the embedding model is unavailable: TF-IDF vectors, L2-normalised."""
    name = "tfidf"
    description = "tfidf:scikit-learn (fallback)"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.fitted = False

    def fit(self, texts: List[str]) -> None:
        self.vectorizer.fit(texts or ["empty"])
        self.fitted = True

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        m = self.vectorizer.transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.where(norms == 0, 1, norms)


def get_embedder():
    """Load the embedder once per process (preferred: embeddings; fallback: TF-IDF)."""
    with _LOCK:
        if _STATE["embedder"] is None:
            wanted = os.environ.get("CLAIMSLEUTH_RAG_RETRIEVER", "embedding")
            if wanted == "embedding":
                try:
                    _STATE["embedder"] = EmbeddingEmbedder()
                except Exception as exc:
                    _STATE["embedder"] = TfidfEmbedder()
                    _STATE["embedder"].fallback_reason = f"{type(exc).__name__}: {str(exc)[:200]}"
            else:
                _STATE["embedder"] = TfidfEmbedder()
        return _STATE["embedder"]


# ─── Index build / load ─────────────────────────────────────────────────────

def corpus_dir() -> Path:
    return Path(settings.RAG_CORPUS_DIR)


def index_dir() -> Path:
    return Path(settings.RAG_INDEX_DIR)


def _corpus_hash(chunks: List[Chunk]) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(json.dumps(c.to_dict(), sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def build_index(force: bool = False) -> Dict:
    """Chunk the policy corpus, embed it and save the index. Returns the manifest."""
    embedder = get_embedder()
    loaded = load_policy_corpus(corpus_dir())
    chunks = loaded.chunks
    embedder.fit([c.text for c in chunks])
    vectors = embedder.encode([c.text for c in chunks])
    folder = index_dir()
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "vectors.npy", vectors)
    with open(folder / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict()) + "\n")
    manifest = {"retriever": embedder.description, "dim": int(vectors.shape[1]) if len(chunks) else 0,
                "chunks": len(chunks), "corpus_sha256": _corpus_hash(chunks), "skipped": loaded.skipped,
                "built_at": datetime.now(timezone.utc).isoformat(),
                "fallback_reason": getattr(embedder, "fallback_reason", None)}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with _LOCK:
        _STATE["index"] = {"chunks": chunks, "vectors": vectors, "manifest": manifest}
    return manifest


def _rebuild() -> Dict:
    build_index()
    with _LOCK:
        return _STATE["index"]


def load_index() -> Dict:
    """
    Load the saved index; rebuild it automatically if it is missing, unreadable, built by a
    different retriever, or out of date with the corpus files.
    """
    with _LOCK:
        cached = _STATE["index"]
    embedder = get_embedder()
    current = load_policy_corpus(corpus_dir())
    if cached and cached["manifest"]["corpus_sha256"] == _corpus_hash(current.chunks):
        return cached
    folder = index_dir()
    try:
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if (manifest["retriever"] != embedder.description
                or manifest["corpus_sha256"] != _corpus_hash(current.chunks)
                or embedder.name == "tfidf"):            # TF-IDF must be refitted in this process
            return _rebuild()
        chunks = [Chunk(**json.loads(l)) for l in open(folder / "chunks.jsonl", encoding="utf-8") if l.strip()]
        vectors = np.load(folder / "vectors.npy")
        if len(chunks) != vectors.shape[0]:
            return _rebuild()
    except (OSError, ValueError, KeyError, TypeError):
        return _rebuild()
    index = {"chunks": chunks, "vectors": vectors, "manifest": manifest}
    with _LOCK:
        _STATE["index"] = index
    return index


# ─── Retrieval ──────────────────────────────────────────────────────────────

def retrieve(question: str, policy_id: Optional[str] = None, claim_id: Optional[str] = None,
             claim_documents: Optional[Dict[str, str]] = None, k: int = 3,
             include_guidelines: bool = True) -> RetrievalResult:
    """
    Return up to k passages relevant to `question`, restricted to:
      - the given policy's wording (and the shared guidelines, if include_guidelines), and/or
      - the given claim's own documents (chunked and embedded now; never other claims').
    """
    embedder = get_embedder()
    index = load_index()
    warnings = list(f"skipped {s['file']}: {s['reason']}" for s in index["manifest"].get("skipped", []))

    candidates: List[Chunk] = []
    vectors: List[np.ndarray] = []
    for chunk, vec in zip(index["chunks"], index["vectors"]):
        if (policy_id and chunk.doc_type == "policy" and chunk.policy_id == policy_id) or \
           (include_guidelines and policy_id and chunk.doc_type == "guideline"):
            candidates.append(chunk)
            vectors.append(vec)
    if claim_id and claim_documents is not None:
        claim_chunks = chunk_claim_documents(claim_id, claim_documents)
        warnings += [f"skipped {s['file']}: {s['reason']}" for s in claim_chunks.skipped]
        if claim_chunks.chunks:
            claim_vecs = embedder.encode([c.text for c in claim_chunks.chunks])
            candidates += claim_chunks.chunks
            vectors += list(claim_vecs)
    if not candidates:
        return RetrievalResult(question=question, retriever=embedder.description,
                               warnings=warnings + ["no documents available for this policy/claim"])

    q = embedder.encode([question])[0]
    scores = np.vstack(vectors) @ q
    order = np.argsort(-scores)
    threshold = MIN_SCORE[embedder.name]
    result = RetrievalResult(question=question, retriever=embedder.description, warnings=warnings)
    for idx in order:
        if scores[idx] < threshold or len(result.passages) >= k:
            break
        c = candidates[idx]
        hits = scan_injection(c.text)
        passage = RetrievedPassage(chunk_id=c.chunk_id, source_file=c.source_file, doc_type=c.doc_type,
                                   section=c.section, policy_id=c.policy_id, claim_id=c.claim_id, text=c.text,
                                   score=round(float(scores[idx]), 4), rank=len(result.passages) + 1,
                                   retriever=embedder.description, quarantined=bool(hits), guardrail_hits=hits)
        (result.quarantined if hits else result.passages).append(passage)
    if not result.passages:
        result.warnings.append(f"no passage scored above {threshold} for this question")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build or query the ClaimSleuth RAG index")
    parser.add_argument("--rebuild", action="store_true", help="regenerate policy wording and rebuild the index")
    parser.add_argument("--query", default=None)
    parser.add_argument("--policy", default=None)
    args = parser.parse_args()
    if args.rebuild:
        from src.rag.corpus import write_policy_corpus
        write_policy_corpus(corpus_dir())
        print(json.dumps(build_index(force=True), indent=2))
    if args.query:
        res = retrieve(args.query, policy_id=args.policy)
        for p in res.passages:
            print(f"#{p.rank} {p.score:.3f} {p.source_file} [{p.section}]")
        print("warnings:", res.warnings)
