from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_TAGS = [
    "python",
    "machine learning",
    "databases",
    "frontend",
    "backend",
    "ai",
    "productivity",
    "architecture",
    "research",
    "embeddings",
]

CATEGORY_DESCRIPTIONS = {
    "Programming": "software development, code, backend, frontend, APIs, testing, architecture",
    "AI": "artificial intelligence, machine learning, embeddings, semantic search, models",
    "Databases": "databases, SQL, storage, indexing, retrieval, data modeling",
    "Research": "papers, experiments, metrics, hypotheses, evaluation, methodology",
    "Productivity": "personal knowledge management, habits, planning, workflows, focus",
    "Personal": "personal reflection, learning journal, life notes, decisions, routines",
    "Projects": "project ideas, implementation plans, prototypes, tasks, milestones",
}


@dataclass
class TaggingResult:
    tags: list[str]
    category: str
    category_score: float


class EmbeddingTagger:
    def __init__(
        self,
        model: SentenceTransformer,
        tags: list[str] | None = None,
        category_descriptions: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        self.tags = tags or DEFAULT_TAGS
        self.category_descriptions = category_descriptions or CATEGORY_DESCRIPTIONS
        self.tag_embeddings = self.model.encode(self.tags, normalize_embeddings=True, show_progress_bar=False)
        self.category_names = list(self.category_descriptions.keys())
        self.category_embeddings = self.model.encode(
            list(self.category_descriptions.values()),
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def classify(self, text: str, top_k_tags: int = 3) -> TaggingResult:
        text_embedding = self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)

        tag_scores = cosine_similarity(text_embedding, self.tag_embeddings).ravel()
        tag_indexes = np.argsort(tag_scores)[::-1][:top_k_tags]
        tags = [self.tags[index] for index in tag_indexes]

        category_scores = cosine_similarity(text_embedding, self.category_embeddings).ravel()
        category_index = int(np.argmax(category_scores))

        return TaggingResult(
            tags=tags,
            category=self.category_names[category_index],
            category_score=float(category_scores[category_index]),
        )

