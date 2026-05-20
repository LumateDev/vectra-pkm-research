from __future__ import annotations

from typing import Iterable

import streamlit as st


TAG_LABELS = {
    "python": "Python",
    "machine learning": "машинное обучение",
    "databases": "базы данных",
    "frontend": "frontend",
    "backend": "backend",
    "ai": "ИИ",
    "productivity": "продуктивность",
    "architecture": "архитектура",
    "research": "исследование",
    "embeddings": "эмбеддинги",
}

CATEGORY_LABELS = {
    "Programming": "Программирование",
    "AI": "Искусственный интеллект",
    "Databases": "Базы данных",
    "Research": "Исследование",
    "Productivity": "Продуктивность",
    "Personal": "Личное",
    "Projects": "Проекты",
}


def format_tag(tag: str) -> str:
    return TAG_LABELS.get(tag, tag)


def format_category(category: str) -> str:
    return CATEGORY_LABELS.get(category, category)


def render_note_card(note_id: int, title: str, content: str, tags: list[str], category: str) -> None:
    with st.container(border=True):
        st.caption(f"Заметка #{note_id} - категория: {format_category(category)}")
        st.subheader(title)
        st.write(content)
        st.markdown("**Автоматические теги:**")
        st.write(" ".join(f"`{format_tag(tag)}`" for tag in tags))


def render_results(results: Iterable, title: str, empty_message: str | None = None) -> None:
    st.markdown(f"### {title}")
    results = list(results)
    if not results:
        st.info(
            empty_message
            or "Введите запрос выше. Например: 'автоматически организовать заметки' или 'database indexes'."
        )
        return

    for rank, result in enumerate(results, start=1):
        with st.container(border=True):
            st.caption(f"Место {rank} - релевантность {result.score:.3f} - заметка #{result.note_id}")
            st.markdown(f"**{result.title}**")
            st.write(result.content[:420] + ("..." if len(result.content) > 420 else ""))
