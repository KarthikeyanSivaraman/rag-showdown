import { useState } from "react";
import "./App.css";

// Backend URL -- FastAPI (app.py). Default port is 8000, but 8000 can get
// stuck with a stale/orphaned socket on Windows (observed 2026-09-26 --
// no owning process, but the OS still reports it LISTENING); pass a
// different --port to uvicorn and update this constant to match if so.
const API_URL = "http://127.0.0.1:8001/ask";

const QUESTION = "How many tickets mention shipping delays?";

function App() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  async function runQuery() {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Request failed (${res.status})`);
      }
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message || "Something went wrong calling the backend.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <header className="header">
        <h1>rag-showdown</h1>
        <p className="subtitle">
          Hand-built top-k RAG vs. a corpus-wide counting question -- does
          plain retrieval surface enough of the truth?
        </p>
      </header>

      <section className="question-card">
        <div className="question-label">Question</div>
        <div className="question-text">&ldquo;{QUESTION}&rdquo;</div>
        <button className="run-button" onClick={runQuery} disabled={loading}>
          {loading ? "Running..." : "Run"}
        </button>
      </section>

      {loading && (
        <div className="loading">
          <div className="spinner" />
          <p>Embedding tickets + calling Claude -- this takes a few seconds...</p>
        </div>
      )}

      {error && (
        <div className="error-card">
          <strong>Error:</strong> {error}
        </div>
      )}

      {result && !loading && <Results result={result} />}
    </div>
  );
}

function Results({ result }) {
  const {
    retrieved_ids,
    claude_answer,
    ground_truth_count,
    ground_truth_ids,
    true_positives,
    missed,
    recall,
  } = result;

  const recallPct = Math.round(recall * 100);

  return (
    <section className="results">
      <div className="stat-card">
        <div className="stat-header">RETRIEVAL RESULT</div>
        <div className="stat-number">
          <span className="stat-good">{true_positives.length}</span>
          <span className="stat-divider"> / </span>
          <span>{ground_truth_count}</span>
        </div>
        <div className="stat-label">true shipping-delay tickets retrieved</div>
        <div className={`recall-badge ${recallPct === 100 ? "recall-full" : "recall-partial"}`}>
          Recall: {recallPct}%
        </div>
      </div>

      <div className="id-grid">
        <div className="id-column">
          <div className="id-column-title">Retrieved (top-k)</div>
          <div className="chip-list">
            {retrieved_ids.map((id) => (
              <span
                key={id}
                className={`chip ${true_positives.includes(id) ? "chip-hit" : "chip-neutral"}`}
              >
                {id}
              </span>
            ))}
          </div>
        </div>

        <div className="id-column">
          <div className="id-column-title">
            Missed ({missed.length} of {ground_truth_count} true matches)
          </div>
          <div className="chip-list">
            {missed.length === 0 ? (
              <span className="chip chip-hit">None -- all found</span>
            ) : (
              missed.map((id) => (
                <span key={id} className="chip chip-miss">
                  {id}
                </span>
              ))
            )}
          </div>
        </div>
      </div>

      <div className="ground-truth-note">
        Ground truth ({ground_truth_count} tickets, see{" "}
        <code>data/answer_key.md</code>): {ground_truth_ids.join(", ")}
      </div>

      <div className="answer-card">
        <div className="answer-header">Claude&apos;s answer</div>
        <pre className="answer-text">{claude_answer}</pre>
      </div>
    </section>
  );
}

export default App;
