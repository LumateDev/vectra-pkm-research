from __future__ import annotations

import csv
from pathlib import Path
from random import Random


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "datasets" / "pkm_notes_large.csv"


TOPICS = {
    "semantic_search": {
        "titles": [
            "Semantic retrieval for personal notes",
            "Finding ideas without exact words",
            "Meaning-based search workflow",
            "Embedding search in a PKM system",
        ],
        "terms": ["embeddings", "cosine similarity", "semantic search", "vectors", "retrieval"],
        "templates": [
            "A personal note search system should retrieve related ideas even when the query uses different wording. The prototype uses {a}, {b}, and ranked retrieval to connect scattered notes.",
            "The main research value is visible when a vague question still finds a relevant note. This note discusses {a}, {b}, and why semantic matching is useful for PKM.",
            "Semantic retrieval treats each note as a vector representation. It can surface concepts about {a} and {b} without relying on exact keyword overlap.",
        ],
    },
    "keyword_search": {
        "titles": [
            "TF-IDF baseline for note search",
            "Keyword matching experiment",
            "Search baseline with word overlap",
            "BM25 and TF-IDF comparison note",
        ],
        "terms": ["TF-IDF", "BM25", "keywords", "word overlap", "baseline"],
        "templates": [
            "The baseline search method uses {a} and {b}. It is simple, interpretable, and useful as a comparison point for semantic retrieval.",
            "Keyword search performs well when the query contains exact terms from the note. This note records observations about {a}, {b}, and ranking.",
            "A strong experiment needs a transparent baseline. Here the focus is on {a}, {b}, and how lexical matching differs from embeddings.",
        ],
    },
    "tagging": {
        "titles": [
            "Automatic tagging with embeddings",
            "Top-k tag recommendation",
            "Organizing notes with labels",
            "Tag similarity experiment",
        ],
        "terms": ["automatic tagging", "labels", "top-k tags", "tag embeddings", "classification"],
        "templates": [
            "The system can suggest tags by comparing a note embedding with tag embeddings. This note focuses on {a}, {b}, and lightweight categorization.",
            "Automatic organization is useful when notes accumulate quickly. The approach uses {a}, {b}, and a small predefined vocabulary.",
            "Tagging should remain explainable in a research prototype. The note discusses {a}, {b}, and ranking tags by similarity.",
        ],
    },
    "databases": {
        "titles": [
            "SQLite storage for notes",
            "Database schema for PKM prototype",
            "Vector index storage idea",
            "Metadata table design",
        ],
        "terms": ["SQLite", "schema", "indexes", "storage", "metadata"],
        "templates": [
            "A simple data layer can store note id, title, content, category, and tags. This note mentions {a}, {b}, and future vector storage.",
            "For the research prototype, CSV is enough, but database storage could support larger experiments. The note covers {a}, {b}, and retrieval metadata.",
            "A later version can persist embeddings and search results. This note is about {a}, {b}, and practical storage decisions.",
        ],
    },
    "research_methods": {
        "titles": [
            "Evaluation plan for search quality",
            "Manual relevance labels",
            "Metrics for retrieval experiments",
            "Research methodology note",
        ],
        "terms": ["Precision@K", "Recall@K", "nDCG@K", "MRR", "relevance labels"],
        "templates": [
            "The experiment compares two retrieval methods using {a}, {b}, and a set of labeled information needs.",
            "Manual relevance labels make the search evaluation reproducible. This note tracks {a}, {b}, and limitations of the dataset.",
            "The methodology chapter should explain how {a} and {b} reflect retrieval quality in a PKM context.",
        ],
    },
    "productivity": {
        "titles": [
            "Weekly review workflow",
            "Personal knowledge review",
            "Task capture inside notes",
            "Reducing friction in PKM",
        ],
        "terms": ["weekly review", "task capture", "workflow", "focus", "personal productivity"],
        "templates": [
            "A useful PKM system helps the user rediscover old ideas during a weekly review. This note covers {a}, {b}, and lightweight routines.",
            "Productivity notes should not become a heavy project management system. The focus is on {a}, {b}, and simple review habits.",
            "The prototype should support thinking and recall. This note discusses {a}, {b}, and connecting tasks with research ideas.",
        ],
    },
    "software_architecture": {
        "titles": [
            "Simple module structure",
            "Keeping the prototype modular",
            "Search API boundary",
            "Architecture without overengineering",
        ],
        "terms": ["modules", "architecture", "API boundary", "Streamlit", "clean code"],
        "templates": [
            "The prototype should separate data loading, search, tagging, and evaluation. This note discusses {a}, {b}, and keeping code readable.",
            "Research code benefits from simple boundaries. The note focuses on {a}, {b}, and avoiding unnecessary backend complexity.",
            "The application can stay small while still being reusable for experiments. This note mentions {a}, {b}, and module design.",
        ],
    },
    "summarization": {
        "titles": [
            "Automatic summary idea",
            "Summaries for search results",
            "Note preview generation",
            "Condensing long research notes",
        ],
        "terms": ["summarization", "abstract", "preview", "LLM", "short summary"],
        "templates": [
            "Search results are easier to inspect when each note has a short summary. This note explores {a}, {b}, and future LLM support.",
            "Summarization is not required for the first prototype, but it can improve result browsing. The note discusses {a}, {b}, and note previews.",
            "A generated abstract can help users decide whether to open a note. This note records ideas about {a}, {b}, and interface design.",
        ],
    },
}


def make_note(note_id: int, topic: str, index: int, random: Random) -> dict[str, str | int]:
    config = TOPICS[topic]
    title = f"{random.choice(config['titles'])} #{index + 1}"
    terms = random.sample(config["terms"], 2)
    template = random.choice(config["templates"])
    content = template.format(a=terms[0], b=terms[1])
    content += " " + random.choice(
        [
            "It is written as a realistic working note rather than a polished article.",
            "The note can be used as evidence in a search-quality experiment.",
            "This entry is intentionally short, like a personal knowledge base fragment.",
            "It includes enough context to be useful for semantic retrieval tests.",
        ]
    )
    return {
        "id": note_id,
        "title": title,
        "content": content,
        "topic": topic,
        "source": "synthetic_pkm",
    }


def main() -> None:
    random = Random(42)
    rows: list[dict[str, str | int]] = []
    note_id = 1

    for topic in TOPICS:
        for index in range(30):
            rows.append(make_note(note_id, topic, index, random))
            note_id += 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "title", "content", "topic", "source"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} notes at {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

