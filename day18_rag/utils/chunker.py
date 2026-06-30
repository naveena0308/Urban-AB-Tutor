"""
chunker.py
----------
Three chunking strategies for splitting text before embedding.
Each function returns a list of strings (the chunks).
"""

import re


def fixed_size_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into fixed-size character chunks with overlap.

    chunk_size : number of characters per chunk
    overlap    : characters shared between consecutive chunks
                 (prevents cutting a sentence at the exact boundary)
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if c]


def sentence_chunks(text: str, max_size: int = 500) -> list[str]:
    """
    Split text at sentence boundaries, grouping sentences until max_size
    characters is reached.

    Preserves complete sentences. No sentence is split in half.
    """
    # Split on period, exclamation, or question mark followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) > max_size and current:
            chunks.append(current.strip())
            current = sentence
        else:
            current = (current + " " + sentence).strip()

    if current:
        chunks.append(current)

    return chunks


def paragraph_chunks(text: str, max_size: int = 1000) -> list[str]:
    """
    Split text at paragraph boundaries (blank lines).
    Merges small paragraphs together until max_size is reached.

    Best for structured documents where each paragraph is a distinct idea.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) > max_size and current:
            chunks.append(current.strip())
            current = para
        else:
            current = (current + "\n\n" + para).strip()

    if current:
        chunks.append(current)

    return chunks
