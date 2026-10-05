"""
inspect_embeddings.py

Educational script: builds the same Chroma collection run_test.py uses, then
prints the raw embedding vectors so you can actually see what "an embedding"
is -- a list of ~384 floating point numbers per ticket (all-MiniLM-L6-v2's
output dimension). Not part of the demo pipeline itself, just for looking.

Usage:
    python inspect_embeddings.py [ticket_id ...]

With no arguments, shows the first 3 tickets. Pass specific ticket IDs
(e.g. "python inspect_embeddings.py T002 T004") to inspect those.
"""
import json
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from services import vector_store  # noqa: E402

TICKETS_PATH = APP_DIR / "data" / "tickets.json"


def main():
    with open(TICKETS_PATH, "r", encoding="utf-8") as f:
        tickets = json.load(f)["tickets"]

    print(f"Building collection from {len(tickets)} tickets (in-memory, ephemeral)...\n")
    collection = vector_store.build_collection(tickets)

    requested_ids = sys.argv[1:] or [t["id"] for t in tickets[:3]]

    result = collection.get(ids=requested_ids, include=["embeddings", "documents"])

    for ticket_id, document, embedding in zip(result["ids"], result["documents"], result["embeddings"]):
        print("=" * 70)
        print(f"Ticket: {ticket_id}")
        print(f"Text: {document[:100]}{'...' if len(document) > 100 else ''}")
        print(f"Embedding dimensions: {len(embedding)}")
        print(f"First 10 of {len(embedding)} numbers: {[round(x, 4) for x in embedding[:10]]}")
        print()

    print(
        "That list of numbers IS the embedding -- a point in ~384-dimensional space.\n"
        "Tickets whose meaning is similar (e.g. two different shipping-delay\n"
        "complaints) end up as points that are close together in that space, even\n"
        "though they share no exact words. Retrieval (collection.query in\n"
        "services/vector_store.py) just finds the k closest points to the\n"
        "question's own embedding -- that's the entire mechanism."
    )


if __name__ == "__main__":
    main()
