from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.data.loader import load_notes
from app.search.keyword import TfidfKeywordSearch
from app.search.semantic import MODEL_NAME, SemanticSearch
from experiments.metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank


REPORTS_DIR = ROOT / "experiments" / "reports"
K_VALUES = [3, 5, 10]

DATASET_CONFIGS = {
    "en": {
        "dataset": ROOT / "datasets" / "pkm_notes_large.csv",
        "scenarios": ROOT / "experiments" / "search_scenarios.csv",
        "suffix": "en",
        "language_name": "English",
    },
    "ru": {
        "dataset": ROOT / "datasets" / "pkm_notes_ru.csv",
        "scenarios": ROOT / "experiments" / "search_scenarios_ru.csv",
        "suffix": "ru",
        "language_name": "Russian",
    },
}


def parse_topics(value: str) -> set[str]:
    return {topic.strip() for topic in value.split("|") if topic.strip()}


def relevant_ids_for_topics(notes: pd.DataFrame, topics: set[str]) -> set[int]:
    return set(notes.loc[notes["topic"].isin(topics), "id"].astype(int).tolist())


def ids(results) -> list[int]:
    return [int(result.note_id) for result in results]


def evaluate_method(method_name: str, retrieved_ids: list[int], relevant_ids: set[int], query: str) -> list[dict]:
    rows = []
    for k in K_VALUES:
        rows.append(
            {
                "method": method_name,
                "query": query,
                "k": k,
                "precision": precision_at_k(retrieved_ids, relevant_ids, k),
                "recall": recall_at_k(retrieved_ids, relevant_ids, k),
                "ndcg": ndcg_at_k(retrieved_ids, relevant_ids, k),
                "mrr": reciprocal_rank(retrieved_ids, relevant_ids),
            }
        )
    return rows


def build_summary(
    results: pd.DataFrame,
    notes_count: int,
    scenarios_count: int,
    language_name: str,
    dataset_path: Path,
    scenarios_path: Path,
) -> str:
    averages = (
        results.groupby(["method", "k"], as_index=False)[["precision", "recall", "ndcg", "mrr"]]
        .mean()
        .sort_values(["k", "method"])
    )
    pivot = averages.pivot(index="k", columns="method", values="ndcg")
    best_at_5 = (
        averages.loc[averages["k"] == 5]
        .sort_values("ndcg", ascending=False)
        .iloc[0]
    )

    lines = [
        "# Search Evaluation Report",
        "",
        "Автоматический прогон сравнивает TF-IDF keyword search и semantic search на synthetic PKM dataset.",
        "",
        "## Dataset",
        "",
        f"- Language: {language_name}.",
        f"- Notes: {notes_count}.",
        f"- Scenarios: {scenarios_count} labeled search scenarios.",
        f"- Dataset file: `{dataset_path.as_posix()}`.",
        f"- Scenario file: `{scenarios_path.as_posix()}`.",
        "- Relevance rule: relevant notes are selected by manually assigned scenario topics.",
        "",
        "## Average Metrics",
        "",
        dataframe_to_markdown(averages),
        "",
        "## Main Conclusion",
        "",
        (
            f"At K=5, the best average nDCG is produced by **{best_at_5['method']}** "
            f"with nDCG={best_at_5['ndcg']:.3f}."
        ),
        "",
    ]

    if {"semantic", "keyword"}.issubset(set(pivot.columns)):
        wins = (pivot["semantic"] > pivot["keyword"]).sum()
        lines.extend(
            [
                f"Semantic search has higher average nDCG for {wins} of {len(pivot)} tested K values.",
                "",
            ]
        )

    lines.extend(
        [
            "## How To Interpret",
            "",
            "- Precision@K: how many top-K results are relevant.",
            "- Recall@K: how many known relevant notes were found in top-K.",
            "- nDCG@K: whether relevant notes appear near the top.",
            "- MRR: how early the first relevant result appears.",
            "",
            "## Notes",
            "",
            "This is prototype-level evidence. The dataset is synthetic, so the result should be presented as an experimental demonstration, not as a universal claim.",
            "",
        ]
    )
    return "\n".join(lines)


def dataframe_to_markdown(dataframe: pd.DataFrame) -> str:
    headers = list(dataframe.columns)
    rows = []
    for record in dataframe.to_dict("records"):
        row = []
        for header in headers:
            value = record[header]
            if isinstance(value, float):
                row.append(f"{value:.3f}")
            else:
                row.append(str(value))
        rows.append(row)

    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join("---" for _ in headers) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *body_lines])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare keyword and semantic search on labeled scenarios.")
    parser.add_argument(
        "--language",
        choices=sorted(DATASET_CONFIGS),
        default="ru",
        help="Dataset/scenario language to evaluate. Default: ru.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DATASET_CONFIGS[args.language]
    dataset_path = config["dataset"]
    scenarios_path = config["scenarios"]

    notes = load_notes(dataset_path)
    scenarios = pd.read_csv(scenarios_path)

    keyword_search = TfidfKeywordSearch(notes)
    model = SentenceTransformer(MODEL_NAME)
    semantic_search = SemanticSearch(notes, model=model)

    metric_rows = []
    top_rows = []

    for scenario in scenarios.to_dict("records"):
        query = scenario["query"]
        relevant_topics = parse_topics(scenario["relevant_topics"])
        relevant_ids = relevant_ids_for_topics(notes, relevant_topics)

        keyword_results = keyword_search.search(query, top_k=10)
        semantic_results = semantic_search.search(query, top_k=10)

        keyword_ids = ids(keyword_results)
        semantic_ids = ids(semantic_results)

        metric_rows.extend(evaluate_method("keyword", keyword_ids, relevant_ids, query))
        metric_rows.extend(evaluate_method("semantic", semantic_ids, relevant_ids, query))

        for method, results in [("keyword", keyword_results), ("semantic", semantic_results)]:
            for rank, result in enumerate(results, start=1):
                top_rows.append(
                    {
                        "query": query,
                        "method": method,
                        "rank": rank,
                        "note_id": result.note_id,
                        "title": result.title,
                        "score": result.score,
                        "is_relevant": result.note_id in relevant_ids,
                    }
                )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(metric_rows)
    top_df = pd.DataFrame(top_rows)
    suffix = config["suffix"]
    metrics_path = REPORTS_DIR / f"search_metrics_{suffix}.csv"
    top_path = REPORTS_DIR / f"search_top_results_{suffix}.csv"
    summary_path = REPORTS_DIR / f"search_evaluation_summary_{suffix}.md"

    metrics_df.to_csv(metrics_path, index=False)
    top_df.to_csv(top_path, index=False)
    summary = build_summary(
        metrics_df,
        notes_count=len(notes),
        scenarios_count=len(scenarios),
        language_name=config["language_name"],
        dataset_path=dataset_path.relative_to(ROOT),
        scenarios_path=scenarios_path.relative_to(ROOT),
    )
    summary_path.write_text(summary, encoding="utf-8")

    print(f"Saved metrics: {metrics_path}")
    print(f"Saved top results: {top_path}")
    print(f"Saved summary: {summary_path}")
    print()
    print(summary)


if __name__ == "__main__":
    main()
