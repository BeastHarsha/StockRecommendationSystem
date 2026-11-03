"""Retrieve recent news chunks per holding from Chroma."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass
class RetrievedDoc:
    id: str
    text: str
    title: str
    source: str
    url: str
    published_date: str
    distance: float


class NewsRetriever:
    def __init__(self, collection, embedder):
        self.collection = collection
        self.embedder = embedder

    def retrieve(self, symbol: str, query: str, k: int = 5, days: int = 180, now: datetime | None = None) -> list[RetrievedDoc]:
        total = self.collection.count()
        if total == 0:
            return []
        cutoff = int(((now or datetime.now(timezone.utc)) - timedelta(days=days)).timestamp())
        result = self.collection.query(
            query_embeddings=self.embedder([query]),
            n_results=min(k, total),
            where={"$and": [{"symbol": {"$eq": symbol}}, {"published_ts": {"$gte": cutoff}}]},
            include=["documents", "metadatas", "distances"],
        )
        return [
            RetrievedDoc(
                id=doc_id,
                text=text,
                title=meta.get("title", ""),
                source=meta.get("source", ""),
                url=meta.get("source_url", ""),
                published_date=meta.get("published_date", ""),
                distance=float(distance),
            )
            for doc_id, text, meta, distance in zip(
                result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]
