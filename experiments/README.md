# Experiments

Автоматические сценарии для проверки исследовательской гипотезы:

> Semantic search based on embeddings works better than keyword search for personal notes.

## Files

- `generate_dataset.py` - генератор английского synthetic PKM dataset.
- `generate_russian_dataset.py` - генератор русского synthetic PKM dataset.
- `search_scenarios.csv` - английские тестовые сценарии.
- `search_scenarios_ru.csv` - русские тестовые сценарии.
- `metrics.py` - Precision@K, Recall@K, nDCG@K, MRR.
- `evaluate_search.py` - сравнение TF-IDF keyword search и semantic search.
- `reports/` - сгенерированные CSV-таблицы и markdown-выводы.

## Run

Русский датасет и русский эксперимент:

```powershell
.\.venv\Scripts\python.exe experiments\generate_russian_dataset.py
.\.venv\Scripts\python.exe experiments\evaluate_search.py --language ru
```

Английский датасет и английский эксперимент:

```powershell
.\.venv\Scripts\python.exe experiments\generate_dataset.py
.\.venv\Scripts\python.exe experiments\evaluate_search.py --language en
```

Generated outputs:

- `experiments/reports/search_metrics_ru.csv`
- `experiments/reports/search_top_results_ru.csv`
- `experiments/reports/search_evaluation_summary_ru.md`
- `experiments/reports/search_metrics_en.csv`
- `experiments/reports/search_top_results_en.csv`
- `experiments/reports/search_evaluation_summary_en.md`

