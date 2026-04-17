"""
memory_store.py — Incident Memory Store

Maintains a persistent knowledge base of past incidents, root causes,
and fixes.  During retrieval the memory is searched first, providing
contextual priors before querying the main vector store.

This simulates a learning system that accumulates institutional
knowledge across analysis sessions.
"""

import json
import os
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


# ─── Incident record ──────────────────────────────────────────────────────────

@dataclass
class Incident:
    """A stored incident from a past analysis session."""
    incident_id: str
    timestamp: str                      # when the incident was recorded
    query: str                          # original user query
    root_cause: str                     # identified root cause
    severity: str                       # CRITICAL / HIGH / MEDIUM / LOW
    confidence: float                   # confidence score [0, 1]
    supporting_logs: List[str]          # key log lines
    fix_recommendation: str             # recommended fix
    error_codes: List[str] = field(default_factory=list)
    modules_involved: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_document(self) -> Document:
        """Convert to a LangChain Document for vector storage."""
        text = (
            f"Root Cause: {self.root_cause}\n"
            f"Severity: {self.severity}\n"
            f"Query: {self.query}\n"
            f"Fix: {self.fix_recommendation}\n"
            f"Error Codes: {', '.join(self.error_codes)}\n"
            f"Modules: {', '.join(self.modules_involved)}\n"
            f"Supporting Logs:\n" + "\n".join(self.supporting_logs[:5])
        )
        return Document(
            page_content=text,
            metadata={
                "incident_id": self.incident_id,
                "timestamp": self.timestamp,
                "severity": self.severity,
                "confidence": self.confidence,
                "type": "memory_incident",
            },
        )

    @classmethod
    def from_dict(cls, d: dict) -> "Incident":
        return cls(**d)


# ─── Memory Store ──────────────────────────────────────────────────────────────

class MemoryStore:
    """
    Persistent incident memory with vector-based retrieval.

    Stores past root cause analyses and enables similarity search
    to find relevant prior incidents before querying the main log
    vector store.  This provides contextual priors that improve
    analysis quality and reduce redundant investigation.

    Storage:
        - JSON file for persistence (human-readable)
        - In-memory ChromaDB collection for fast similarity search
    """

    def __init__(
        self,
        storage_path: str = "data/memory_store.json",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.storage_path = storage_path
        self.incidents: List[Incident] = []
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vectorstore: Optional[Chroma] = None

        # Load existing incidents
        self._load()
        self._rebuild_index()

    # ── Public API ─────────────────────────────────────────────────────────

    def add_incident(self, incident: Incident) -> None:
        """Store a new incident and update the search index."""
        self.incidents.append(incident)
        self._save()
        self._rebuild_index()

    def search(self, query: str, top_k: int = 3) -> List[Incident]:
        """
        Find past incidents relevant to the current query.

        Returns incidents sorted by similarity to the query.
        """
        if not self.vectorstore or not self.incidents:
            return []

        results = self.vectorstore.similarity_search_with_relevance_scores(
            query, k=min(top_k, len(self.incidents))
        )

        matched = []
        for doc, score in results:
            inc_id = doc.metadata.get("incident_id")
            for inc in self.incidents:
                if inc.incident_id == inc_id:
                    matched.append(inc)
                    break

        return matched

    def get_context_for_query(self, query: str, top_k: int = 3) -> str:
        """
        Get formatted context from past incidents for LLM injection.

        Returns a string block describing relevant prior incidents
        that the LLM can use as additional context.
        """
        incidents = self.search(query, top_k)
        if not incidents:
            return ""

        lines = ["RELEVANT PAST INCIDENTS (from memory):"]
        for i, inc in enumerate(incidents, 1):
            lines.append(f"\n  Incident {i}: [{inc.severity}] {inc.root_cause}")
            lines.append(f"    Confidence: {inc.confidence:.2f}")
            lines.append(f"    Fix: {inc.fix_recommendation}")
            if inc.error_codes:
                lines.append(f"    Error codes: {', '.join(inc.error_codes)}")
            if inc.modules_involved:
                lines.append(f"    Modules: {', '.join(inc.modules_involved)}")

        return "\n".join(lines)

    def create_incident_from_result(
        self,
        query: str,
        analysis_result: dict,
    ) -> Incident:
        """
        Create an Incident object from an AdaptiveRAG analysis result.

        The analysis_result should contain keys like root_cause,
        confidence, supporting_logs, reasoning_steps, etc.
        """
        incident = Incident(
            incident_id=f"INC-{int(time.time())}",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            query=query,
            root_cause=analysis_result.get("root_cause", "Unknown"),
            severity=analysis_result.get("severity", "MEDIUM"),
            confidence=analysis_result.get("confidence", 0.0),
            supporting_logs=analysis_result.get("supporting_logs", []),
            fix_recommendation=analysis_result.get("recommendation", ""),
            error_codes=analysis_result.get("error_codes", []),
            modules_involved=analysis_result.get("modules_involved", []),
            tags=analysis_result.get("tags", []),
        )
        return incident

    def clear(self) -> None:
        """Remove all stored incidents."""
        self.incidents = []
        self._save()
        self.vectorstore = None

    @property
    def size(self) -> int:
        return len(self.incidents)

    # ── Persistence ────────────────────────────────────────────────────────

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.storage_path) or ".", exist_ok=True)
        data = [inc.to_dict() for inc in self.incidents]
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def _load(self) -> None:
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.incidents = [Incident.from_dict(d) for d in data]
            except (json.JSONDecodeError, TypeError, KeyError):
                self.incidents = []

    def _rebuild_index(self) -> None:
        if not self.incidents:
            self.vectorstore = None
            return

        docs = [inc.to_document() for inc in self.incidents]

        import chromadb
        client = chromadb.Client()
        try:
            client.delete_collection("memory_incidents")
        except Exception:
            pass

        self.vectorstore = Chroma.from_documents(
            docs, self.embeddings,
            collection_name="memory_incidents",
            client=client,
        )
