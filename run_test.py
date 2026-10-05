"""
run_test.py

CLI test harness for the rag-showdown demo. Loads the 29 synthetic support
tickets (data/tickets.json), embeds + indexes them with the local Chroma /
sentence-transformers vector store (services/vector_store.py), retrieves the
top-k most relevant tickets for a single counting question, and asks Claude
(via the headless CLI, services/claude_client.py) to answer using ONLY the
retrieved chunks.

This is the "hand-built RAG" side of the showdown: plain top-k retrieval over
a question that requires a corpus-wide count. The point of the demo is to see
whether top-k (k=5 here) actually surfaces enough of the true matches to let
the LLM count correctly -- per data/answer_key.md, the ground truth is 7
matching tickets out of 29, deliberately more than k, so it's expected (not a
bug) if this pipeline undercounts.

Idempotent: build_collection() always rebuilds the Chroma collection from
scratch (see its docstring), so re-running this script never double-indexes
or accumulates duplicate tickets across runs.

Usage:
    python run_test.py
"""
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from services import vector_store, claude_client  # noqa: E402

load_dotenv(APP_DIR / ".env")

QUESTION = "How many tickets mention shipping delays?"
TOP_K = 5
GROUND_TRUTH_COUNT = 7  # data/answer_key.md
GROUND_TRUTH_IDS = {"T002", "T004", "T007", "T010", "T013", "T018", "T029"}  # data/answer_key.md
TICKETS_PATH = APP_DIR / "data" / "tickets.json"

PROMPT_TEMPLATE = """You are answering a question using ONLY the support tickets provided below. \
Do not use any outside knowledge and do not assume tickets exist beyond what is shown here.

Question: {question}

Here are the {n} support tickets retrieved as potentially relevant:

{chunks}

Instructions:
- Answer using ONLY the tickets shown above. If a ticket doesn't actually describe a shipping \
delay (e.g. it's about a wrong item, an address change, or an on-time delivery), don't count it.
- Give a specific numeric count of how many of the tickets shown above are genuinely about a \
shipping delay, followed by a list of their ticket IDs.
- Do not claim there are more matching tickets than were shown to you above.
"""


def load_tickets():
    with open(TICKETS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["tickets"]


def build_prompt(question, retrieved):
    chunks = "\n\n".join(
        f"[{ticket_id}]\n{document}" for ticket_id, document, _distance in retrieved
    )
    return PROMPT_TEMPLATE.format(question=question, n=len(retrieved), chunks=chunks)


def main():
    tickets = load_tickets()
    print(f"Loaded {len(tickets)} tickets from {TICKETS_PATH.name}")

    print("Embedding + indexing tickets (first run downloads the ~80MB embedding "
          "model -- normal, not a hang)...")
    collection = vector_store.build_collection(tickets)

    retrieved = vector_store.retrieve(collection, QUESTION, k=TOP_K)
    retrieved_ids = [ticket_id for ticket_id, _doc, _dist in retrieved]

    print(f"\nQuestion: {QUESTION}")
    print(f"Retrieved top-{TOP_K} ticket IDs: {retrieved_ids}")

    prompt = build_prompt(QUESTION, retrieved)

    print("\nCalling Claude (headless CLI, no tools -- answering only from retrieved chunks)...")
    try:
        answer = claude_client.ask_claude(prompt, cwd=str(APP_DIR))
    except claude_client.ClaudeCliError as exc:
        print(f"\nERROR calling claude CLI: {exc}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("CLAUDE'S ANSWER")
    print("=" * 70)
    print(answer)

    print("\n" + "=" * 70)
    print("GRADING")
    print("=" * 70)
    retrieved_set = set(retrieved_ids)
    true_positives = sorted(retrieved_set & GROUND_TRUTH_IDS)
    missed = sorted(GROUND_TRUTH_IDS - retrieved_set)
    false_positives = sorted(retrieved_set - GROUND_TRUTH_IDS)

    print(f"Retrieved ticket IDs (top-{TOP_K}): {retrieved_ids}")
    print(f"Ground-truth matching tickets ({GROUND_TRUTH_COUNT} total, data/answer_key.md): "
          f"{sorted(GROUND_TRUTH_IDS)}")
    print(f"\nOf the {GROUND_TRUTH_COUNT} true matches, retrieval surfaced "
          f"{len(true_positives)}: {true_positives}")
    if missed:
        print(f"Retrieval MISSED {len(missed)} true match(es), so no matter how well "
              f"Claude reasons, it cannot report the true total of {GROUND_TRUTH_COUNT}: {missed}")
    if false_positives:
        print(f"Retrieval also surfaced {len(false_positives)} non-matching ticket(s): "
              f"{false_positives}")
    print(f"\nRecall: {len(true_positives)}/{GROUND_TRUTH_COUNT} "
          f"({100 * len(true_positives) / GROUND_TRUTH_COUNT:.0f}%)")
    print("\n(This grades retrieval completeness against the known answer key -- it does not "
          "parse Claude's prose for a number, since a naive digit-substring match on the answer "
          "text is unreliable, e.g. it can false-match on a ticket ID like \"T007\". Read Claude's "
          "answer above for what it actually concluded from the tickets it was given.)")


if __name__ == "__main__":
    main()
