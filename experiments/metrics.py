from __future__ import annotations

import math


def precision_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    retrieved_at_k = retrieved_ids[:k]
    hits = sum(1 for note_id in retrieved_at_k if note_id in relevant_ids)
    return hits / k


def recall_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    if not relevant_ids:
        return 0.0
    retrieved_at_k = retrieved_ids[:k]
    hits = sum(1 for note_id in retrieved_at_k if note_id in relevant_ids)
    return hits / len(relevant_ids)


def dcg_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    score = 0.0
    for index, note_id in enumerate(retrieved_ids[:k], start=1):
        if note_id in relevant_ids:
            score += 1.0 / math.log2(index + 1)
    return score


def ndcg_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    ideal_hits = min(len(relevant_ids), k)
    if ideal_hits == 0:
        return 0.0
    ideal_dcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg_at_k(retrieved_ids, relevant_ids, k) / ideal_dcg


def reciprocal_rank(retrieved_ids: list[int], relevant_ids: set[int]) -> float:
    for index, note_id in enumerate(retrieved_ids, start=1):
        if note_id in relevant_ids:
            return 1.0 / index
    return 0.0

