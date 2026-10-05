# rag-showdown

**Your RAG pipeline isn't hallucinating. It's blind.**

A small, reproducible demo of the failure mode where top-k RAG gets a
*counting* question wrong without ever hallucinating. Companion code for the
YouTube video on **AI Data Simplified** (link added on publish).

Ask a plain top-k RAG pipeline *"How many tickets mention shipping delays?"*
over 29 support tickets and it answers **5**. The real answer is **7**. Every
ticket the model saw was described correctly; it just never saw the other
two. Top-k retrieval hands the LLM the *k most similar* chunks, and a "how
many" question needs *all* of them.

This mirrors the finding in Snowflake's engineering blog post
[Beyond RAG: Enterprise Search Meets Analytics Over Unstructured Data](https://www.snowflake.com/en/blog/engineering/beyond-rag-enterprise-analytical-search/),
where a standard RAG agent scored 5.6% on analytical questions over
unstructured data.

## The dataset

`data/tickets.json`: 29 synthetic support tickets for a fictional
subscription-box company.

- **7** genuinely describe a shipping delay, each phrased differently and
  none containing the words "shipping delay", so keyword search can't cheat.
- **3** are near-miss distractors that mention shipping but aren't delay
  complaints (an address change, a wrong item, an item that arrived on time
  but damaged).
- **19** are unrelated.

`data/answer_key.md` holds the hand-checked ground truth. The pipeline never
reads it; it's only there for grading.

## The pipeline

1. **Embed + index**: `services/vector_store.py` embeds each ticket (subject +
   body) locally with sentence-transformers (`all-MiniLM-L6-v2`, no API key)
   into an in-memory Chroma collection.
2. **Retrieve**: the top 5 tickets most similar to the question.
3. **Generate**: the question plus the 5 retrieved tickets (and nothing else)
   go to Claude through the headless Claude Code CLI (`claude -p`,
   `services/claude_client.py`). **All tools are disabled**, so Claude can't
   go read the dataset or the answer key itself. It can only answer from
   what retrieval gave it.
4. **Grade**: compare the retrieved IDs against the answer key.

Result from a real run: retrieved `T029, T007, T018, T010, T002`, missed
`T004` and `T013`, recall **5/7 (71%)**. Claude correctly reports that all 5
tickets it was shown are shipping delays, so there's no hallucination, just
an incomplete view.

## Run it

**Requirements:** Python 3.10+, Node 18+ (for the optional UI), and the
[Claude Code](https://claude.com/claude-code) CLI on your `PATH`, logged in.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

The first run downloads the embedding model (~80 MB) from Hugging Face. That
can take a few minutes; later runs use the cache.

### Command line

```bash
python run_test.py
```

Prints the retrieved ticket IDs, Claude's full answer, and the ground-truth
comparison. Each run rebuilds the index from scratch.

### Web UI (FastAPI + React)

```bash
# terminal 1: backend
uvicorn app:app --port 8001

# terminal 2: frontend
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 and click **Run**. You'll see the 5/7 stat card,
the retrieved (green) vs. missed (red) ticket IDs, and Claude's answer.

`POST /ask` on the backend returns the same result as JSON:
`retrieved_ids`, `claude_answer`, `true_positives`, `missed`, `recall`.

### Peek at the embeddings

```bash
python inspect_embeddings.py T002 T004
```

Prints the raw 384-dimensional vectors so you can see what an embedding
actually is.

## Notes

- Each `/ask` call or `run_test.py` run makes **one real Claude call**
  through your own Claude Code login.
- **Running this from inside a Claude Code session?** Start the backend from
  a normal terminal. Nested `claude -p` calls can hang when they inherit a
  parent session's `CLAUDECODE` environment variables.
- Only the one fixed question is supported. It's a demo of a specific
  failure mode, not a general Q&A app.
- Not built (yet): a second approach, like full-context or aggregation-aware
  retrieval, to compare against this top-k baseline.

## License

MIT, see [LICENSE](LICENSE).
