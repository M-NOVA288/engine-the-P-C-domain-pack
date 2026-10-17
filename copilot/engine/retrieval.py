"""Guideline retrieval (the "R" in RAG).

Local mode uses TF-IDF over markdown sections so it runs anywhere.
Azure mode swaps in Azure AI Search (hybrid keyword + vector) behind the same
interface; see `AzureSearchRetriever`.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    section_id: str
    title: str
    text: str
    source: str


def chunk_markdown(path: Path) -> list[Chunk]:
    """Split a guideline document on headings like '## 3.2 Loss history'."""
    chunks: list[Chunk] = []
    current_id, current_title, buf = "0", "Preamble", []
    for line in path.read_text().splitlines():
        m = re.match(r"^#{2,3}\s+(\d+(?:\.\d+)*)\s+(.*)", line)
        if m:
            if buf:
                chunks.append(Chunk(current_id, current_title, "\n".join(buf).strip(), path.name))
            current_id, current_title, buf = m.group(1), m.group(2).strip(), []
        else:
            buf.append(line)
    if buf:
        chunks.append(Chunk(current_id, current_title, "\n".join(buf).strip(), path.name))
    return [c for c in chunks if c.text]


class LocalRetriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.matrix = self.vectorizer.fit_transform([f"{c.title} {c.text}" for c in chunks])

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        ranked = sorted(zip(self.chunks, scores), key=lambda x: x[1], reverse=True)
        return [(c, float(s)) for c, s in ranked[:k]]


class AzureSearchRetriever:  # pragma: no cover - requires Azure resources
    """Same interface backed by Azure AI Search. Index the same chunks with
    fields: section_id, title, text, source, and a vector field for embeddings."""

    def __init__(self) -> None:
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents import SearchClient

        self.client = SearchClient(
            endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
            index_name=os.environ.get("AZURE_SEARCH_INDEX", "uw-guidelines"),
            credential=AzureKeyCredential(os.environ["AZURE_SEARCH_KEY"]),
        )

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        results = self.client.search(search_text=query, top=k)
        return [
            (Chunk(r["section_id"], r["title"], r["text"], r["source"]), float(r["@search.score"]))
            for r in results
        ]


def get_retriever(guideline_path: Path):
    if os.environ.get("COPILOT_RETRIEVER", "local").lower() == "azure":
        return AzureSearchRetriever()
    return LocalRetriever(chunk_markdown(guideline_path))
