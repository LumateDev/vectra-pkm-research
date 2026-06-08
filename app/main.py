from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.data.loader import DEFAULT_DATASET_PATH, load_notes
from app.search.keyword import TfidfKeywordSearch
from app.search.semantic import MODEL_NAME, SemanticSearch
from app.tagging.classifier import EmbeddingTagger
from app.ui.components import render_note_card, render_results


@st.cache_resource(show_spinner="Загружаю embedding-модель...")
def get_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


@st.cache_data(show_spinner="Загружаю заметки...")
def get_notes(dataset_version: float):
    return load_notes()


@st.cache_resource(show_spinner="Строю keyword-индекс...")
def get_keyword_search(_notes, dataset_version: float):
    return TfidfKeywordSearch(_notes)


@st.cache_resource(show_spinner="Строю semantic-индекс...")
def get_semantic_search(_notes, _model, dataset_version: float):
    return SemanticSearch(_notes, model=_model)


@st.cache_resource(show_spinner="Готовлю автоматические теги...")
def get_tagger(_model):
    return EmbeddingTagger(_model)


@st.cache_resource(show_spinner="Определяю категории заметок...")
def get_classified_notes(_notes, _tagger, dataset_version: float):
    notes_with_metadata = _notes.copy()
    classifications = [_tagger.classify(text) for text in notes_with_metadata["text"]]
    notes_with_metadata["tags"] = [item.tags for item in classifications]
    notes_with_metadata["category"] = [item.category for item in classifications]
    return notes_with_metadata


def main() -> None:
    st.set_page_config(page_title="Vectra Research", layout="wide")
    st.title("Vectra Research")
    st.caption("Исследовательский прототип PKM-системы: сравнение keyword search и semantic search.")

    with st.expander("Как пользоваться прототипом", expanded=True):
        st.markdown(
            """
            1. Слева выберите заметку из датасета, чтобы посмотреть ее текст, авто-теги и категорию.
            2. Справа введите запрос обычными словами. Можно писать по-русски или по-английски.
            3. Выберите режим поиска:
               - **Семантический поиск** ищет по смыслу через embeddings.
               - **Поиск по ключевым словам** ищет совпадения слов через TF-IDF.
               - **Сравнение** показывает оба метода рядом.
            4. Блок **Похожие заметки** показывает semantic-соседей выбранной заметки.
            """
        )

    dataset_version = DEFAULT_DATASET_PATH.stat().st_mtime
    notes = get_notes(dataset_version)
    model = get_model()
    keyword_search = get_keyword_search(notes, dataset_version)
    semantic_search = get_semantic_search(notes, model, dataset_version)
    tagger = get_tagger(model)
    notes_with_metadata = get_classified_notes(notes, tagger, dataset_version)

    with st.sidebar:
        st.header("Заметки")
        st.caption("Выберите заметку для просмотра и поиска похожих материалов.")
        selected_id = st.selectbox(
            "Открыть заметку",
            notes_with_metadata["id"].tolist(),
            format_func=lambda note_id: notes_with_metadata.loc[
                notes_with_metadata["id"] == note_id, "title"
            ].iloc[0],
        )
        st.divider()
        st.metric("Размер датасета", len(notes_with_metadata))
        st.caption("Синтетические PKM-style заметки для демонстрации и экспериментов.")

    selected_note = notes_with_metadata.loc[notes_with_metadata["id"] == selected_id].iloc[0]

    left, right = st.columns([1, 1])

    with left:
        st.markdown("## Просмотр заметки")
        render_note_card(
            note_id=int(selected_note["id"]),
            title=str(selected_note["title"]),
            content=str(selected_note["content"]),
            tags=list(selected_note["tags"]),
            category=str(selected_note["category"]),
        )

        related = semantic_search.related_notes(int(selected_note["id"]), top_k=3)
        render_results(related, "Похожие заметки")

    with right:
        st.markdown("## Поиск")
        query = st.text_input(
            "Введите поисковый запрос",
            placeholder="Например: как автоматически организовать мои заметки",
            help="Семантический поиск понимает смысл запроса, даже если слова не совпадают с текстом заметки.",
        )
        search_mode = st.radio(
            "Режим поиска",
            ["Семантический поиск", "Поиск по ключевым словам", "Сравнение"],
            horizontal=True,
            help="Для курсовой особенно полезен режим сравнения: он показывает разницу между TF-IDF и embeddings.",
        )
        top_k = st.slider(
            "Сколько результатов показать",
            min_value=3,
            max_value=10,
            value=5,
            help="Top-K: количество верхних результатов в выдаче.",
        )

        st.caption(
            "Подсказка: попробуйте запросы 'как найти похожие идеи без точных слов', "
            "'где хранить векторы заметок' или 'как доказать что один поиск лучше другого'."
        )

        if search_mode == "Поиск по ключевым словам":
            render_results(
                keyword_search.search(query, top_k=top_k),
                "Результаты keyword search",
                "Введите запрос. TF-IDF лучше всего работает, когда слова запроса есть в заметках.",
            )
        elif search_mode == "Семантический поиск":
            render_results(
                semantic_search.search(query, top_k=top_k),
                "Результаты semantic search",
                "Введите запрос. Semantic search ищет заметки по смысловой близости.",
            )
        else:
            keyword_col, semantic_col = st.columns(2)
            with keyword_col:
                render_results(
                    keyword_search.search(query, top_k=top_k),
                    "Keyword search",
                    "Введите запрос, чтобы увидеть результаты TF-IDF.",
                )
            with semantic_col:
                render_results(
                    semantic_search.search(query, top_k=top_k),
                    "Semantic search",
                    "Введите запрос, чтобы увидеть результаты embeddings-поиска.",
                )


if __name__ == "__main__":
    main()
