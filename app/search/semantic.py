from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from app.utils.config import EMBEDDING_MODEL

MODEL_NAME = EMBEDDING_MODEL


@dataclass
class SemanticSearchResult:
    note_id: int
    title: str
    score: float
    content: str


class SemanticSearch:
    def __init__(self, notes: pd.DataFrame, model: SentenceTransformer | None = None) -> None:
        self.notes = notes.reset_index(drop=True)
        self.model = model or SentenceTransformer(MODEL_NAME)
        self.embeddings = self.model.encode(
            self.notes["text"].tolist(),
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def search(self, query: str, top_k: int = 5) -> list[SemanticSearchResult]:
        if not query.strip():
            return []

        query_embedding = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        scores = cosine_similarity(query_embedding, self.embeddings).ravel()
        ranked_indexes = np.argsort(scores)[::-1][:top_k]

        return [
            SemanticSearchResult(
                note_id=int(self.notes.iloc[index]["id"]),
                title=str(self.notes.iloc[index]["title"]),
                score=float(scores[index]),
                content=str(self.notes.iloc[index]["content"]),
            )
            for index in ranked_indexes
        ]

    def related_notes(self, note_id: int, top_k: int = 5) -> list[SemanticSearchResult]:
        matches = self.notes.index[self.notes["id"] == note_id].tolist()
        if not matches:
            return []

        source_index = matches[0]
        scores = cosine_similarity([self.embeddings[source_index]], self.embeddings).ravel()
        ranked_indexes = [index for index in np.argsort(scores)[::-1] if index != source_index][:top_k]

        return [
            SemanticSearchResult(
                note_id=int(self.notes.iloc[index]["id"]),
                title=str(self.notes.iloc[index]["title"]),
                score=float(scores[index]),
                content=str(self.notes.iloc[index]["content"]),
            )
            for index in ranked_indexes
        ]
