from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .gemini_client import GeminiClient, GeminiConnectionError
from .models import Owner
from .scheduler import Scheduler


@dataclass
class RetrievedChunk:
    source: str
    title: str
    text: str
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


class BaseRetriever:
    def __init__(self, kb_path: str) -> None:
        self.kb_path = Path(kb_path)
        payload = json.loads(self.kb_path.read_text(encoding="utf-8"))
        self.kb_docs = payload["documents"]

    def _build_schedule_docs(self, owner: Owner) -> list[dict]:
        scheduler = Scheduler(owner)
        docs: list[dict] = []
        for pet in owner.pets:
            docs.append(
                {
                    "source": "schedule",
                    "title": f"Pet profile: {pet.name}",
                    "text": f"{pet.name} is a {pet.age} year old {pet.species}. Notes: {pet.notes or 'none'}",
                }
            )
            for task in pet.tasks:
                docs.append(
                    {
                        "source": "schedule",
                        "title": f"Task for {pet.name}",
                        "text": (
                            f"{pet.name} has task {task.description} on {task.due_date.isoformat()} "
                            f"at {task.due_time.strftime('%H:%M')} priority {task.priority} "
                            f"frequency {task.frequency} duration {task.duration_minutes} minutes "
                            f"type {task.task_type} status {'complete' if task.completed else 'pending'}"
                        ),
                    }
                )
        summary = scheduler.summarize_schedule()
        docs.append(
            {
                "source": "schedule",
                "title": "Schedule summary",
                "text": (
                    f"Pending tasks {summary.pending_count}. Completed tasks {summary.completed_count}. "
                    f"High priority pending {summary.high_priority_count}. "
                    f"Conflicts {'; '.join(summary.conflicts) if summary.conflicts else 'none'}."
                ),
            }
        )
        return docs

    def build_corpus(self, owner: Owner) -> list[dict]:
        return self.kb_docs + self._build_schedule_docs(owner)


class TfidfRetriever(BaseRetriever):
    def __init__(self, kb_path: str) -> None:
        super().__init__(kb_path)
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))

    def retrieve(self, query: str, owner: Owner, top_k: int = 5) -> List[RetrievedChunk]:
        corpus = self.build_corpus(owner)
        texts = [doc["text"] for doc in corpus]
        matrix = self.vectorizer.fit_transform(texts + [query])
        doc_matrix = matrix[:-1]
        query_vector = matrix[-1]
        scores = cosine_similarity(query_vector, doc_matrix).flatten()
        ranked_indices = scores.argsort()[::-1][:top_k]
        return [
            RetrievedChunk(
                source=corpus[index]["source"],
                title=corpus[index]["title"],
                text=corpus[index]["text"],
                score=float(scores[index]),
            )
            for index in ranked_indices
        ]


class GeminiEmbeddingRetriever(BaseRetriever):
    def __init__(
        self,
        kb_path: str,
        *,
        client: Optional[GeminiClient] = None,
        embedding_model: str = "gemini-embedding-001",
        output_dimensionality: int = 768,
    ) -> None:
        super().__init__(kb_path)
        self.client = client or GeminiClient()
        self.embedding_model = embedding_model
        self.output_dimensionality = output_dimensionality

    def retrieve(self, query: str, owner: Owner, top_k: int = 5) -> List[RetrievedChunk]:
        corpus = self.build_corpus(owner)
        texts = [doc["text"] for doc in corpus]
        titles = [doc["title"] for doc in corpus]
        doc_vectors = self.client.embed_texts(
            texts=texts,
            task_type="RETRIEVAL_DOCUMENT",
            titles=titles,
            output_dimensionality=self.output_dimensionality,
            model=self.embedding_model,
        )
        query_vector = self.client.embed_texts(
            texts=[query],
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=self.output_dimensionality,
            model=self.embedding_model,
        )[0]
        doc_matrix = np.array(doc_vectors)
        query_matrix = np.array([query_vector])
        scores = cosine_similarity(query_matrix, doc_matrix).flatten()
        ranked_indices = scores.argsort()[::-1][:top_k]
        return [
            RetrievedChunk(
                source=corpus[index]["source"],
                title=corpus[index]["title"],
                text=corpus[index]["text"],
                score=float(scores[index]),
            )
            for index in ranked_indices
        ]


class RAGRetriever:
    """Facade that selects the configured retrieval backend."""

    def __init__(
        self,
        kb_path: str,
        *,
        backend: str = "tfidf",
        gemini_client: Optional[GeminiClient] = None,
        gemini_embedding_model: str = "gemini-embedding-001",
    ) -> None:
        self.kb_path = kb_path
        self.backend = backend
        self.gemini_client = gemini_client
        self.gemini_embedding_model = gemini_embedding_model
        self.tfidf = TfidfRetriever(kb_path)
        self._gemini: Optional[GeminiEmbeddingRetriever] = None

    def set_backend(self, backend: str) -> None:
        self.backend = backend

    def retrieve(self, query: str, owner: Owner, top_k: int = 5) -> List[RetrievedChunk]:
        if self.backend == "gemini":
            if self._gemini is None:
                self._gemini = GeminiEmbeddingRetriever(
                    self.kb_path,
                    client=self.gemini_client,
                    embedding_model=self.gemini_embedding_model,
                )
            try:
                return self._gemini.retrieve(query, owner, top_k=top_k)
            except GeminiConnectionError:
                raise
        return self.tfidf.retrieve(query, owner, top_k=top_k)
