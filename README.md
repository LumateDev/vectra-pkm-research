# Vectra PKM Research

Исследовательский прототип AI-powered PKM-системы для магистерской работы:

> Разработка интеллектуальной системы для управления персональными знаниями на основе методов искусственного интеллекта.

Проект не является production SaaS или полноценным аналогом Notion/Obsidian. Это небольшой прототип для демонстрации semantic search, automatic tagging, categorization и экспериментального сравнения с keyword search.

## Возможности

- просмотр PKM-style заметок;
- keyword search на базе TF-IDF;
- semantic search на базе sentence-transformers embeddings;
- сравнение keyword search и semantic search в одном интерфейсе;
- автоматические теги через similarity между заметкой и тегами;
- автоматическая категория заметки;
- похожие заметки через cosine similarity;
- автоматические экспериментальные сценарии с Precision@K, Recall@K, nDCG@K и MRR.

## Быстрый Запуск

```powershell
cd C:\Users\Alex\Desktop\Vectra-Research
.\.venv\Scripts\python.exe -m streamlit run app/main.py
```

После запуска открыть:

```text
http://127.0.0.1:8501
```

Если `.venv` еще нет:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Первый запуск semantic search может быть медленным: модель embeddings скачивается в локальный cache Hugging Face.

## Embedding Model

По умолчанию используется multilingual-модель:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Она лучше подходит для русского датасета и русских запросов. При необходимости можно переключить модель через переменную окружения:

```powershell
$env:VECTRA_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
```

## Структура

```text
app/
  data/          загрузка датасетов
  search/        keyword и semantic search
  tagging/       авто-теги и категории
  ui/            компоненты Streamlit UI
  utils/         конфигурация
datasets/        synthetic PKM-style datasets
experiments/     сценарии, метрики, отчеты
notebooks/       место для exploratory notebooks
```

## Датасеты

- `datasets/pkm_notes_ru.csv` - основной русский датасет для UI, 240 заметок.
- `datasets/pkm_notes_large.csv` - английский synthetic dataset, 240 заметок.
- `datasets/notes.csv` - маленький стартовый dataset, 60 заметок.

Русский датасет можно пересоздать:

```powershell
.\.venv\Scripts\python.exe experiments\generate_russian_dataset.py
```

Английский датасет:

```powershell
.\.venv\Scripts\python.exe experiments\generate_dataset.py
```

## Эксперименты

Русский прогон:

```powershell
.\.venv\Scripts\python.exe experiments\evaluate_search.py --language ru
```

Английский прогон:

```powershell
.\.venv\Scripts\python.exe experiments\evaluate_search.py --language en
```

Отчеты сохраняются в `experiments/reports/`:

- `search_metrics_ru.csv`
- `search_top_results_ru.csv`
- `search_evaluation_summary_ru.md`
- `search_metrics_en.csv`
- `search_top_results_en.csv`
- `search_evaluation_summary_en.md`

## Исследовательская Гипотеза

> Семантический поиск на основе embeddings работает лучше keyword-search для персональных заметок, особенно когда пользователь формулирует запрос не теми же словами, что использованы в заметке.

Метрики:

- Precision@K;
- Recall@K;
- nDCG@K;
- MRR.

Важно: датасеты синтетические, поэтому выводы следует подавать как экспериментальную демонстрацию прототипа, а не как универсальное доказательство.

