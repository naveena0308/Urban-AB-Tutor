"""
searcher.py
-----------
Helpers for FAISS indexing and ChromaDB operations.
These wrap the lower-level library calls so notebooks stay readable.
"""

import numpy as np
import faiss
import chromadb


# ─────────────────────────────────────────────
# FAISS helpers
# ─────────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a flat L2 FAISS index from a 2D array of embeddings.

    embeddings : shape (N, D) — N vectors, each of dimension D
    Returns a FAISS index ready for search.

    Flat L2 is the simplest index type: exact search, no approximation.
    For normalized vectors, L2 distance and cosine distance are equivalent.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))
    return index


def faiss_search(index: faiss.Index, query_vector: np.ndarray, k: int = 3):
    """
    Search a FAISS index for the k nearest neighbors of query_vector.

    Returns (distances, indices) — both are 1D arrays of length k.
    distances : L2 distances (lower = more similar)
    indices   : positions in the original embeddings array
    """
    query = query_vector.reshape(1, -1).astype(np.float32)
    distances, indices = index.search(query, k)
    return distances[0], indices[0]


# ─────────────────────────────────────────────
# ChromaDB helpers
# ─────────────────────────────────────────────

def get_chroma_collection(collection_name: str, persist_path: str = None):
    """
    Create or retrieve a ChromaDB collection.

    persist_path : if given, data is saved to disk at that path.
                   if None, data lives only in memory (lost on restart).
    """
    if persist_path:
        client = chromadb.PersistentClient(path=persist_path)
    else:
        client = chromadb.Client()
    return client.get_or_create_collection(name=collection_name)


def chroma_add(collection, ids: list, texts: list, embeddings: list, metadatas: list = None):
    """
    Add documents to a ChromaDB collection.

    ids        : unique string id for each document
    texts      : the original text of each document
    embeddings : precomputed embedding vectors (list of lists)
    metadatas  : optional list of dicts with extra information per document
    """
    kwargs = dict(ids=ids, documents=texts, embeddings=embeddings)
    if metadatas:
        kwargs["metadatas"] = metadatas
    collection.add(**kwargs)


def chroma_query(collection, query_embedding: list, n_results: int = 3, filters: dict = None):
    """
    Query a ChromaDB collection by embedding similarity.

    query_embedding : a single embedding vector (list of floats)
    n_results       : how many results to return
    filters         : optional metadata filter dict, e.g. {"source": "faq.txt"}

    Returns the raw ChromaDB result dict.
    """
    kwargs = dict(query_embeddings=[query_embedding], n_results=n_results)
    if filters:
        kwargs["where"] = filters
    return collection.query(**kwargs)


def print_chroma_results(results: dict):
    """Pretty-print ChromaDB query results."""
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
        similarity = round(1 - dist, 4)
        print(f"Result {i + 1}  (similarity: {similarity})")
        if meta:
            print(f"  Metadata : {meta}")
        preview = doc[:150] + "..." if len(doc) > 150 else doc
        print(f"  Text     : {preview}")
        print()
