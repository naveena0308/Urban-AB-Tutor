# Day 18 — RAG Pipeline: Embeddings, FAISS, and ChromaDB

This folder contains hands-on notebooks for understanding Retrieval-Augmented Generation (RAG)
using only open-source tools. No paid API required for the core concepts.

---

## Folder Structure

```
day18_rag/
|
|-- notebooks/
|   |-- 01_embeddings_intro.ipynb        # What embeddings are, how to create them
|   |-- 02_faiss_similarity_search.ipynb # FAISS: indexing vectors, nearest-neighbor search
|   |-- 03_chromadb_vector_store.ipynb   # ChromaDB: storing, querying, filtering documents
|   |-- 04_chunking_strategies.ipynb     # Fixed-size, sentence, and paragraph chunking
|   |-- 05_full_rag_pipeline.ipynb       # End-to-end: text -> chunks -> embed -> store -> query
|
|-- data/
|   |-- sample_sales_notes.txt           # Sample document used across notebooks
|   |-- sample_faq.txt                   # Another sample document for variety
|
|-- utils/
|   |-- embedder.py                      # Reusable embedding function (sentence-transformers)
|   |-- chunker.py                       # Reusable chunking functions
|   |-- searcher.py                      # Reusable FAISS and ChromaDB search helpers
|
|-- requirements.txt                     # All dependencies
|-- README.md                            # This file
```

---

## Tools Used

| Tool                  | Purpose                                       |
|-----------------------|-----------------------------------------------|
| sentence-transformers | Local embedding model (no API key needed)     |
| FAISS                 | Fast similarity search over raw vectors       |
| ChromaDB              | Full vector store with metadata and persistence|
| pypdf                 | Extract text from PDF files                   |
| numpy                 | Vector math (cosine similarity)               |

---

## Notebook Order

Run them in order. Each builds on the previous one.

1. **01_embeddings_intro** — Understand what an embedding is before touching any database.
2. **02_faiss_similarity_search** — Use FAISS to search raw vectors. Pure math, no frills.
3. **03_chromadb_vector_store** — Use ChromaDB to store documents with metadata and query them.
4. **04_chunking_strategies** — Learn how to split a long document before embedding it.
5. **05_full_rag_pipeline** — Combine everything into a working retrieval system.

---

## Setup

```bash
pip install -r requirements.txt
```

All notebooks are self-contained. Run cells top to bottom.

---

## Key Concept

Keyword search matches exact words.
Semantic search matches meaning.

Embeddings make semantic search possible by converting text into numbers
that capture meaning. Similar meanings produce similar numbers.
FAISS and ChromaDB let you search through millions of those numbers in milliseconds.

That is the foundation of every RAG system.
