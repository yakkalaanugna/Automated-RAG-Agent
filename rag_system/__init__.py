# Analyzing the Retrieval–Reasoning Gap in Adaptive RAG Systems
# Modular package for research-grade RAG system
#
# Heavy imports (retriever, adaptive_agent) are lazy to keep test collection
# fast — they pull in sentence-transformers and CrossEncoder which take
# several seconds to load.

from .parser import TelecomLogParser
from .evaluator import RAGEvaluator
from . import config


def __getattr__(name):
    """Lazy-load heavy modules on first access."""
    if name == "HybridRetriever":
        from .retriever import HybridRetriever
        return HybridRetriever
    if name == "QueryRefiner":
        from .query_refiner import QueryRefiner
        return QueryRefiner
    if name == "MemoryStore":
        from .memory_store import MemoryStore
        return MemoryStore
    if name == "AdaptiveIterativeRAGAgent":
        from .adaptive_agent import AdaptiveIterativeRAGAgent
        return AdaptiveIterativeRAGAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TelecomLogParser",
    "HybridRetriever",
    "QueryRefiner",
    "MemoryStore",
    "RAGEvaluator",
    "AdaptiveIterativeRAGAgent",
    "config",
]
