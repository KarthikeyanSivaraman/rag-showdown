"""
app.py

FastAPI server exposing the rag-showdown pipeline (services/vector_store.py +
services/claude_client.py) as a single HTTP endpoint for the React frontend
(frontend/) to call. This is a thin wrapper -- all retrieval/generation logic
lives in services/, matching what run_test.py already does; nothing here
duplicates that logic.

Only one question is supported (this is a demo of a specific known-good/known-
bad RAG result, not a general Q&A app) -- see run_test.py's QUESTION and
data/answer_key.md for why.

Idempotent: build_collection() rebuilds the Chroma collection from scratch on
every call to /ask (same as run_test.py), so re-running never double-indexes.
This is a demo app (single fixed question, small local corpus) -- rebuilding
per request is deliberately simple rather than caching, since the whole
pipeline (embed 29 tickets + one Claude CLI call) is what's being demoed.

Run (port 8001 matches API_URL in frontend/src/App.jsx):
    uvicorn app:app --reload --port 8001
"""
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from services import vector_store, claude_client  # noqa: E402
from run_test import (  # noqa: E402
    QUESTION,
    TOP_K,
    GROUND_TRUTH_COUNT,
    GROUND_TRUTH_IDS,
    TICKETS_PATH,
    load_tickets,
    build_prompt,
)

load_dotenv(APP_DIR / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_showdown.app")

app = FastAPI(title="rag-showdown", description="Hand-built top-k RAG demo API")

# Local dev only -- React (Vite) runs on a different origin than this API.
# Vite's default port is 5173, but it auto-increments (5174, 5175, ...) if
# that port is already taken by another process -- observed 2026-09-26 when
# a leftover dev server was still holding 5173. Allow a small range rather
# than hardcoding one port, since which one Vite actually picks isn't fixed.
# No auth/production hardening needed for a local demo app.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):517[0-9]",
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = QUESTION
    k: int = TOP_K


class AskResponse(BaseModel):
    question: str
    retrieved_ids: list[str]
    claude_answer: str
    ground_truth_count: int
    ground_truth_ids: list[str]
    true_positives: list[str]
    missed: list[str]
    recall: float


@app.get("/")
def root():
    return {
        "app": "rag-showdown",
        "endpoint": "POST /ask",
        "question": QUESTION,
        "default_k": TOP_K,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest = AskRequest()):
    if req.question != QUESTION:
        raise HTTPException(
            status_code=400,
            detail=(
                "This demo only supports the fixed question it was built to test "
                f"retrieval against: {QUESTION!r}. See data/answer_key.md."
            ),
        )
    if req.k < 1 or req.k > 29:
        raise HTTPException(status_code=400, detail="k must be between 1 and 29 (corpus size).")

    tickets = load_tickets()
    logger.info("Loaded %d tickets from %s", len(tickets), TICKETS_PATH.name)

    collection = vector_store.build_collection(tickets)
    retrieved = vector_store.retrieve(collection, req.question, k=req.k)
    retrieved_ids = [ticket_id for ticket_id, _doc, _dist in retrieved]

    prompt = build_prompt(req.question, retrieved)

    try:
        answer = claude_client.ask_claude(prompt, cwd=str(APP_DIR))
    except claude_client.ClaudeCliError as exc:
        logger.error("claude CLI call failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"claude CLI call failed: {exc}") from exc

    retrieved_set = set(retrieved_ids)
    true_positives = sorted(retrieved_set & GROUND_TRUTH_IDS)
    missed = sorted(GROUND_TRUTH_IDS - retrieved_set)
    recall = len(true_positives) / GROUND_TRUTH_COUNT

    return AskResponse(
        question=req.question,
        retrieved_ids=retrieved_ids,
        claude_answer=answer,
        ground_truth_count=GROUND_TRUTH_COUNT,
        ground_truth_ids=sorted(GROUND_TRUTH_IDS),
        true_positives=true_positives,
        missed=missed,
        recall=recall,
    )
