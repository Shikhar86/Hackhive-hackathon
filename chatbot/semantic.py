"""Optional, local embedding search over stored event summaries."""

import json
import math
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


OLLAMA_EMBED_URL = "http://127.0.0.1:11434/api/embed"
EMBED_MODEL = "nomic-embed-text"


def _embed(texts: list[str]) -> list[list[float]]:
    payload = json.dumps({"model": EMBED_MODEL, "input": texts}).encode("utf-8")
    request = Request(
        OLLAMA_EMBED_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            vectors = json.load(response)["embeddings"]
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(
            "Local semantic search needs Ollama running with the nomic-embed-text model"
        ) from exc
    if len(vectors) != len(texts) or not all(vectors):
        raise RuntimeError("Ollama returned incomplete embeddings")
    return vectors


def rank_events(question: str, events: list[dict], limit: int) -> list[dict]:
    """Rank existing event rows by cosine similarity without changing the DB."""
    if not events:
        return []
    texts = [question, *(f"{event['category'].replace('_', ' ')}: {event['summary']}" for event in events)]
    vectors = _embed(texts)
    query = vectors[0]
    query_norm = math.sqrt(sum(value * value for value in query))

    def score(vector: list[float]) -> float:
        norm = math.sqrt(sum(value * value for value in vector))
        if not query_norm or not norm or len(vector) != len(query):
            return float("-inf")
        return sum(left * right for left, right in zip(query, vector)) / (query_norm * norm)

    return [event for _, event in sorted(
        zip((score(vector) for vector in vectors[1:]), events),
        key=lambda item: item[0], reverse=True,
    )[:limit]]
