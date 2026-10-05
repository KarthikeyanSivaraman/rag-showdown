# rag-showdown frontend

React + Vite single-page demo UI for the rag-showdown FastAPI backend
(`../app.py`). See the app-level `../README.md` for setup, run instructions,
ports, and CORS notes -- this file is intentionally minimal to avoid
duplicating that doc.

Quick start:

```bash
npm install
npm run dev
```

Requires the backend already running on `http://127.0.0.1:8000`
(`uvicorn app:app --reload` from the `rag-showdown/` root).
