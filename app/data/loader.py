from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[2] / "datasets" / "pkm_notes_ru.csv"


def load_notes(path: Path | str = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    notes = pd.read_csv(path)
    required_columns = {"id", "title", "content"}
    missing_columns = required_columns.difference(notes.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    notes["text"] = notes["title"].fillna("") + "\n\n" + notes["content"].fillna("")
    return notes
