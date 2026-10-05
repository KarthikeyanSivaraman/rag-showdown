"""
services/claude_client.py

Generation layer for the RAG pipeline: calls Claude via the headless
Claude Code CLI (`claude -p`), authenticated with your existing Claude
login -- no ANTHROPIC_API_KEY needed.

All tools are disabled for this call. Claude must answer only from the
retrieved chunks already inlined in the prompt -- if it could still
Read/Glob/Grep/WebFetch, it could go find the real answer itself and the
test would no longer be measuring the retrieval pipeline.
"""
import json
import logging
import subprocess

logger = logging.getLogger("rag_showdown.claude_client")

DISALLOWED_TOOLS = "Bash Edit MultiEdit NotebookEdit Write Read Glob Grep WebFetch WebSearch"
DEFAULT_TIMEOUT_SECONDS = 120


class ClaudeCliError(Exception):
    pass


def ask_claude(prompt: str, cwd: str, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> str:
    """Runs a headless `claude -p` call and returns its final text reply."""
    try:
        result = subprocess.run(
            [
                "claude",
                "-p",
                prompt,
                "--output-format",
                "json",
                "--disallowedTools",
                DISALLOWED_TOOLS,
            ],
            cwd=cwd,
            capture_output=True,
            encoding="utf-8",  # Windows gotcha: plain text=True
            timeout=timeout,   # decodes with the system locale (cp1252 here) and mangles/crashes on em-dashes etc.
        )
    except subprocess.TimeoutExpired as exc:
        logger.exception("claude CLI call timed out")
        raise ClaudeCliError("claude CLI call timed out") from exc
    except OSError as exc:
        logger.exception("Could not launch claude CLI")
        raise ClaudeCliError(f"Could not launch claude CLI: {exc}") from exc
    except UnicodeDecodeError as exc:
        logger.exception("Could not decode claude CLI output as UTF-8")
        raise ClaudeCliError("claude CLI output had invalid encoding") from exc

    if result.returncode != 0:
        logger.error("claude CLI exited %d: %s", result.returncode, result.stderr[:2000])
        raise ClaudeCliError(f"claude CLI failed (exit {result.returncode}): {result.stderr[:500]}")

    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        logger.exception("Could not parse claude CLI output: %r", result.stdout[:2000])
        raise ClaudeCliError(f"Could not parse claude CLI output: {result.stdout[:500]}") from exc

    if parsed.get("is_error"):
        logger.error("claude CLI reported error: %s", parsed)
        raise ClaudeCliError(parsed.get("result") or "claude CLI call failed")

    logger.info(
        "claude call complete cost_usd=%s turns=%s",
        parsed.get("total_cost_usd"),
        parsed.get("num_turns"),
    )
    return parsed.get("result", "") or "(no text returned)"
