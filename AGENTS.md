# AGENTS.md

## Project Context

This repository is a research prototype for an AI-powered Personal Knowledge Management system. It supports a master's research project about using AI methods for personal knowledge management.

The project should stay small, readable, and experiment-oriented. Do not turn it into a production SaaS, enterprise backend, microservice architecture, or complex frontend application.

## Core Stack

- Python
- Streamlit
- sentence-transformers
- scikit-learn
- pandas
- numpy
- matplotlib / plotly for later visualizations

## Main Workflows

Run the app:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/main.py
```

Run Russian experiments:

```powershell
.\.venv\Scripts\python.exe experiments\evaluate_search.py --language ru
```

Run English experiments:

```powershell
.\.venv\Scripts\python.exe experiments\evaluate_search.py --language en
```

## Coding Guidance

- Prefer simple modules over abstractions.
- Keep UI minimal and research-focused.
- Keep search logic independent from Streamlit so experiments can reuse it.
- Preserve dataset columns: `id`, `title`, `content`, `topic`, `source`.
- Do not commit `.venv`, model caches, Python cache files, or local secrets.
- If changing evaluation logic, regenerate reports in `experiments/reports/`.

