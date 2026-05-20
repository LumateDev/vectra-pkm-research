# Search Evaluation Report

Автоматический прогон сравнивает TF-IDF keyword search и semantic search на synthetic PKM dataset.

## Dataset

- Language: English.
- Notes: 240.
- Scenarios: 15 labeled search scenarios.
- Dataset file: `datasets/pkm_notes_large.csv`.
- Scenario file: `experiments/search_scenarios.csv`.
- Relevance rule: relevant notes are selected by manually assigned scenario topics.

## Average Metrics

| method | k | precision | recall | ndcg | mrr |
| --- | --- | --- | --- | --- | --- |
| keyword | 3 | 0.733 | 0.073 | 0.722 | 0.733 |
| semantic | 3 | 0.889 | 0.089 | 0.898 | 0.943 |
| keyword | 5 | 0.733 | 0.122 | 0.726 | 0.733 |
| semantic | 5 | 0.840 | 0.140 | 0.862 | 0.943 |
| keyword | 10 | 0.700 | 0.233 | 0.705 | 0.733 |
| semantic | 10 | 0.793 | 0.264 | 0.821 | 0.943 |

## Main Conclusion

At K=5, the best average nDCG is produced by **semantic** with nDCG=0.862.

Semantic search has higher average nDCG for 3 of 3 tested K values.

## How To Interpret

- Precision@K: how many top-K results are relevant.
- Recall@K: how many known relevant notes were found in top-K.
- nDCG@K: whether relevant notes appear near the top.
- MRR: how early the first relevant result appears.

## Notes

This is prototype-level evidence. The dataset is synthetic, so the result should be presented as an experimental demonstration, not as a universal claim.
