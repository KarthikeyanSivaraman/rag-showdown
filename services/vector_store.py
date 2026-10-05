"""
services/vector_store.py

Local, self-hosted embedding + vector store for the RAG pipeline --
sentence-transformers (all-MiniLM-L6-v2, a small ~80MB model) for
embeddings, Chroma (embedded/local, no server process) for storage. No
paid embedding API, no hosted vector DB.
"""
import os

import chromadb
from chromadb.utils import embedding_functions

DEFAULT_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
COLLECTION_NAME = "support_tickets"


def build_collection(tickets, persist_dir=None, model_name=None):
    """
    Embeds `tickets` (list of {id, subject, body}) and loads them into a
    fresh Chroma collection.

    Idempotent: always deletes any existing collection with the same name
    first, then recreates it and inserts once, so re-running this (e.g. via
    run_test.py) never double-inserts the same tickets or accumulates
    duplicates across runs.
    """
    model_name = model_name or DEFAULT_MODEL
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

    client = chromadb.PersistentClient(path=persist_dir) if persist_dir else chromadb.Client()

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # collection didn't exist yet on this client -- fine

    collection = client.create_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)

    ids = [t["id"] for t in tickets]
    documents = [f"Subject: {t['subject']}\n{t['body']}" for t in tickets]
    metadatas = [{"subject": t["subject"]} for t in tickets]

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return collection


def retrieve(collection, query, k=5):
    """Returns top-k (ticket_id, document_text, distance) tuples for `query`."""
    result = collection.query(query_texts=[query], n_results=k)
    ids = result["ids"][0]
    documents = result["documents"][0]
    distances = result["distances"][0]
    return list(zip(ids, documents, distances))
