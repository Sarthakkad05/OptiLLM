"""
Multi-Strategy Document Chunker
Supports fixed-window, recursive-character, and semantic chunking strategies.
"""

import re
from typing import Any, Dict, List


def chunk_text(
    text: str,
    strategy: str = "recursive",
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Dict[str, Any]]:
    """
    Split input text into chunks based on specified chunking strategy.

    Returns:
        List of dicts: [{"chunk_index": int, "text": str, "char_count": int}]
    """
    if not text or not text.strip():
        return []

    strategy = (strategy or "recursive").lower()

    if strategy == "fixed":
        chunks = _chunk_fixed(text, chunk_size, overlap)
    elif strategy == "semantic":
        chunks = _chunk_semantic(text, chunk_size, overlap)
    else:
        chunks = _chunk_recursive(text, chunk_size, overlap)

    result = []
    for idx, c_text in enumerate(chunks):
        c_clean = c_text.strip()
        if c_clean:
            result.append(
                {
                    "chunk_index": idx,
                    "text": c_clean,
                    "char_count": len(c_clean),
                }
            )

    return result


def _chunk_fixed(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    text_len = len(text)
    step = max(1, chunk_size - overlap)

    while start < text_len:
        end = min(text_len, start + chunk_size)
        chunks.append(text[start:end])
        start += step
        if start >= text_len or end == text_len:
            break

    return chunks


def _chunk_recursive(text: str, chunk_size: int, overlap: int) -> List[str]:
    # Separator priority: double newlines, single newlines, sentences, spaces
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) + 2 <= chunk_size:
            current_chunk = (current_chunk + "\n\n" + para).strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            
            # If paragraph itself exceeds chunk_size, split by sentences
            if len(para) > chunk_size:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                sub_chunk = ""
                for sent in sentences:
                    if len(sub_chunk) + len(sent) + 1 <= chunk_size:
                        sub_chunk = (sub_chunk + " " + sent).strip()
                    else:
                        if sub_chunk:
                            chunks.append(sub_chunk)
                        sub_chunk = sent
                current_chunk = sub_chunk
            else:
                current_chunk = para

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _chunk_semantic(text: str, chunk_size: int, overlap: int) -> List[str]:
    # Group sentences by topical coherence
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk = (current_chunk + " " + sentence).strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks
