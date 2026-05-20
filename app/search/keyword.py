from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class KeywordSearchResult:
    note_id: int
    title: str
    score: float
    content: str


class TfidfKeywordSearch:
    def __init__(self, notes: pd.DataFrame) -> None:
        self.notes = notes.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(self.notes["text"])

    def search(self, query: str, top_k: int = 5) -> list[KeywordSearchResult]:
        if not query.strip():
            return []

        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        ranked_indexes = np.argsort(scores)[::-1][:top_k]

        return [
            KeywordSearchResult(
                note_id=int(self.notes.iloc[index]["id"]),
                title=str(self.notes.iloc[index]["title"]),
                score=float(scores[index]),
                content=str(self.notes.iloc[index]["content"]),
            )
            for index in ranked_indexes
            if scores[index] > 0
        ]

