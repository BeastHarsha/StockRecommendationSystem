"""Fetch, sanitize, chunk and embed company news into Chroma.

Run with: python -m rag.ingest   (ingests news for every holding in the portfolio)
"""

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import requests

from agent.guardrails import sanitize_text

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
COLLECTION_NAME = "news"
CHUNK_SIZE = 2000  # characters, roughly 500 tokens
CHUNK_OVERLAP = 200


@dataclass
class NewsItem:
    title: str
    url: str
    source: str
    published: datetime
    summary: str


@dataclass
class IngestResult:
    symbol: str
    fetched: int
    stored: int
    removed_sentences: int


def build_news_query(symbol: str, company_name: str | None) -> str:
    name = company_name if company_name and company_name != symbol else symbol
    return f'"{name}" share NSE'


def fetch_news(query: str, days: int = 180, max_items: int = 25, timeout: int = 15) -> list[NewsItem]:
    params = {"q": f"{query} when:{days}d", "hl": "en-IN", "gl": "IN", "ceid": "IN:en"}
    response = requests.get(
        GOOGLE_NEWS_RSS, params=params, timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0 (PortfolioPulse news ingest)"},
    )
    response.raise_for_status()
    feed = feedparser.parse(response.content)

    items = []
    for entry in feed.entries[:max_items]:
        if not entry.get("published_parsed") or not entry.get("link"):
            continue
        items.append(NewsItem(
            title=entry.get("title", ""),
            url=entry["link"],
            source=(entry.get("source") or {}).get("title", ""),
            published=datetime(*entry.published_parsed[:6], tzinfo=timezone.utc),
            summary=entry.get("summary", ""),
        ))
    return items


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if not text:
        return []
    chunks, start = [], 0
    while True:
        chunks.append(text[start:start + size])
        if start + size >= len(text):
            return chunks
        start += size - overlap


class SentenceTransformerEmbedder:
    """Loads the model on first use; importing sentence-transformers takes tens of seconds."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def __call__(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model.encode(list(texts), normalize_embeddings=True).tolist()


def open_collection(persist_dir: Path):
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir), settings=ChromaSettings(anonymized_telemetry=False))
    return client.get_or_create_collection(COLLECTION_NAME, embedding_function=None, metadata={"hnsw:space": "cosine"})


def _document_text(item: NewsItem) -> tuple[str, str, int]:
    title = sanitize_text(item.title, max_chars=500)
    body = sanitize_text(item.summary)
    removed = len(title.removed) + len(body.removed)
    if body.text and title.text and title.text in body.text:
        text = body.text
    else:
        text = ". ".join(part for part in (title.text, body.text) if part)
    return title.text, text, removed


def ingest_news(collection, embedder, symbol: str, company_name: str | None, fetch=fetch_news, days: int = 180) -> IngestResult:
    items = fetch(build_news_query(symbol, company_name), days=days)
    records: dict[str, tuple[str, dict]] = {}
    removed_total = 0

    for item in items:
        title, text, removed = _document_text(item)
        removed_total += removed
        for index, chunk in enumerate(chunk_text(text)):
            doc_id = hashlib.sha1(f"{symbol}|{item.url}|{index}".encode()).hexdigest()
            records[doc_id] = (chunk, {
                "symbol": symbol,
                "title": title[:300],
                "source": item.source,
                "source_url": item.url,
                "published_ts": int(item.published.timestamp()),
                "published_date": item.published.date().isoformat(),
            })

    if records:
        ids = list(records)
        documents = [records[i][0] for i in ids]
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=[records[i][1] for i in ids],
            embeddings=embedder(documents),
        )
    return IngestResult(symbol=symbol, fetched=len(items), stored=len(records), removed_sentences=removed_total)


async def _ingest_portfolio() -> None:
    from agent.guardrails import GuardedToolbox
    from agent.mcp_client import groww_tools
    from config import load_settings

    settings = load_settings()
    async with groww_tools() as tools:
        portfolio = await GuardedToolbox(tools).call("get_portfolio")

    collection = open_collection(settings.chroma_persist_dir)
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    for holding in portfolio["holdings"]:
        result = ingest_news(collection, embedder, holding["symbol"], holding.get("company_name"))
        print(f"{result.symbol}: fetched {result.fetched}, stored {result.stored}, "
              f"removed {result.removed_sentences} suspicious sentences")


if __name__ == "__main__":
    import asyncio

    asyncio.run(_ingest_portfolio())
