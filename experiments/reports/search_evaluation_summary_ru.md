# Search Evaluation Report

Автоматический прогон сравнивает TF-IDF keyword search и semantic search на synthetic PKM dataset.

## Dataset

- Language: Russian.
- Notes: 240.
- Scenarios: 15 labeled search scenarios.
- Dataset file: `datasets/pkm_notes_ru.csv`.
- Scenario file: `experiments/search_scenarios_ru.csv`.
- Relevance rule: relevant notes are selected by manually assigned scenario topics.

## Average Metrics

| method | k | precision | recall | ndcg | mrr |
| --- | --- | --- | --- | --- | --- |
| keyword | 3 | 0.578 | 0.058 | 0.580 | 0.656 |
| semantic | 3 | 0.689 | 0.069 | 0.659 | 0.674 |
| keyword | 5 | 0.640 | 0.107 | 0.624 | 0.656 |
| semantic | 5 | 0.640 | 0.107 | 0.634 | 0.674 |
| keyword | 10 | 0.567 | 0.189 | 0.578 | 0.656 |
| semantic | 10 | 0.593 | 0.198 | 0.602 | 0.674 |

## Main Conclusion

At K=5, the best average nDCG is produced by **semantic** with nDCG=0.634.

Semantic search has higher average nDCG for 3 of 3 tested K values.

## How To Interpret

- Precision@K: how many top-K results are relevant.
- Recall@K: how many known relevant notes were found in top-K.
- nDCG@K: whether relevant notes appear near the top.
- MRR: how early the first relevant result appears.

## Notes

This is prototype-level evidence. The dataset is synthetic, so the result should be presented as an experimental demonstration, not as a universal claim.
