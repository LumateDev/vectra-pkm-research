from __future__ import annotations

import os


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_MODEL = os.getenv("VECTRA_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

