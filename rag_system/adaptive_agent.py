"""
adaptive_agent.py — Adaptive Iterative RAG Agent

Core contribution: replaces naive fixed-loop iteration with an adaptive
confidence-gated approach.  Each iteration:
    1. Retrieves documents (hybrid: vector + BM25)
    2. Runs LLM analysis on retrieved context
    3. Computes a multi-signal confidence score
    4. Compares improvement against a convergence threshold
    5. Refines the query if improvement is above threshold
    6. Stops early when confidence plateaus or reaches maximum

The agent maintains full iteration history and returns an explainable
structured result including reasoning steps, retrieval scores, and
per-iteration confidence.
"""

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from .parser import TelecomLogParser, LogRecord
from .retriever import HybridRetriever, ScoredDocument
from .query_refiner import QueryRefiner
from .memory_store import MemoryStore


# ─── Iteration snapshot ────────────────────────────────────────────────────────

@dataclass
class IterationResult:
    """Snapshot of a single iteration's state."""
    iteration: int
    query: str
    confidence: float
    improvement: float
    analysis: str
    retrieval_scores: List[float]
    num_docs_retrieved: int
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "query": self.query,
            "confidence": round(self.confidence, 4),
            "improvement": round(self.improvement, 4),
            "analysis_preview": self.analysis[:200] + "..." if len(self.analysis) > 200 else self.analysis,
            "retrieval_scores": [round(s, 4) for s in self.retrieval_scores],
            "num_docs_retrieved": self.num_docs_retrieved,
        }


# ─── Adaptive Iterative RAG Agent ─────────────────────────────────────────────

class AdaptiveIterativeRAGAgent:
    """
    Adaptive Iterative Retrieval-Augmented Generation agent for
    telecom log root cause analysis.

    Architecture:
        ┌─────────────────────────────────────────────────┐
        │               User Query                         │
        │                    │                              │
        │         ┌──────────▼──────────┐                  │
        │         │  Memory Lookup      │                  │
        │         └──────────┬──────────┘                  │
        │                    │                              │
        │  ┌─────────────────▼─────────────────────────┐   │
        │  │         ADAPTIVE LOOP (max N)              │   │
        │  │  ┌──────────────────────────────────────┐  │   │
        │  │  │ 1. Query Refinement (LLM)            │  │   │
        │  │  │ 2. Hybrid Retrieval (Vector + BM25)  │  │   │
        │  │  │ 3. Cross-Encoder Reranking            │  │   │
        │  │  │ 4. LLM Analysis                      │  │   │
        │  │  │ 5. Confidence Scoring                │  │   │
        │  │  │ 6. Convergence Check                 │  │   │
        │  │  └──────────────────────────────────────┘  │   │
        │  └─────────────────┬─────────────────────────┘   │
        │                    │                              │
        │         ┌──────────▼──────────┐                  │
        │         │  Best Result Select │                  │
        │         └──────────┬──────────┘                  │
        │                    │                              │
        │         ┌──────────▼──────────┐                  │
        │         │  Memory Store       │                  │
        │         └─────────────────────┘                  │
        └─────────────────────────────────────────────────┘

    Parameters:
        max_iterations:        upper bound on iteration count (default 5)
        convergence_threshold: minimum improvement to continue (default 0.01)
        alpha:                 hybrid retrieval vector weight (default 0.7)
        top_k:                 documents per retrieval pass (default 12)
    """

    ANALYSIS_PROMPT = ChatPromptTemplate.from_template(
        """You are an expert telecom log analyst specializing in 5G gNB, UE testing, and end-to-end call flow debugging.
You understand eGate console logs, uec_1.log, rain runtime logs, btslog/syslog, RLC stats, and bearer stats.

CRITICAL: Use TIMESTAMPS to correlate events across different log files.
Each log entry shows [SEVERITY] [file:Line_number] @timestamp message.
ALWAYS reference the exact file name, line number, and timestamp when citing evidence.

{memory_context}

Iteration {iteration} of {max_iterations}.
{previous_findings}

Format your response EXACTLY as:
## Root Cause
(one-line summary)

## Severity
CRITICAL / HIGH / MEDIUM / LOW

## Error Timeline
(events in chronological order with file:line @timestamp)

## Details
(deep explanation of the failure chain referencing specific lines and timestamps)

## Reasoning Steps
1. (step-by-step reasoning about how you arrived at the root cause)
2. ...

## Recommendation
(what to do)

Context (retrieved logs):
{context}

Question: {question}"""
    )

    def __init__(
        self,
        groq_api_key: str,
        model_name: str = "llama-3.3-70b-versatile",
        max_iterations: int = 3,
        convergence_threshold: float = 0.01,
        alpha: float = 0.7,
        top_k: int = 6,
        use_memory: bool = True,
        memory_path: str = "data/memory_store.json",
    ):
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.top_k = top_k
        self.use_memory = use_memory

        # LLM
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model_name,
            max_tokens=2048,
            temperature=0,
        )

        # Components
        self.parser = TelecomLogParser()
        self.retriever = HybridRetriever(alpha=alpha, top_k=top_k)
        self.query_refiner = QueryRefiner(self.llm)
        self.memory = MemoryStore(storage_path=memory_path) if use_memory else None

        # State
        self.records: List[LogRecord] = []
        self.documents: List[Document] = []
        self._indexed = False

    # ── Data loading ───────────────────────────────────────────────────────

    def load_logs(self, folder: str) -> List[LogRecord]:
        """Parse and index all logs from a folder."""
        self.records = self.parser.parse_folder(folder)
        self._build_documents()
        self._index()
        return self.records

    def load_records(self, records: List[LogRecord]) -> None:
        """Load pre-parsed LogRecord objects."""
        self.records = records
        self._build_documents()
        self._index()

    def load_texts(self, texts: List[str], filenames: List[str]) -> List[LogRecord]:
        """Parse raw text strings and index them."""
        self.records = []
        for text, fname in zip(texts, filenames):
            self.records.extend(self.parser.parse_text(text, fname))
        self._build_documents()
        self._index()
        return self.records

    def _build_documents(self) -> None:
        """Convert LogRecords to LangChain Documents with enriched content.
        
        Uses natural-language style enrichment rather than bracket-formatted
        metadata, because:
          - all-MiniLM-L6-v2 was trained on natural text; structured prefixes
            like [ERROR] [file:L4] degrade embedding quality
          - BM25 tokenizes on whitespace, so natural words like "error" and
            "log1.txt" match queries better than "[error]" or "[log1.txt:l4]"
          - The source/line/timestamp remain in metadata for format_retrieved()
        """
        self.documents = [
            Document(
                page_content=(
                    f"{r.log_level} {r.source} line {r.line_number} "
                    f"{'at ' + r.timestamp + ' ' if r.timestamp else ''}"
                    f"{r.module + ' ' if r.module else ''}"
                    f"{r.error_code + ' ' if r.error_code else ''}"
                    f"{r.message}"
                ),
                metadata={
                    "source": r.source,
                    "line": r.line_number,
                    "timestamp": r.timestamp,
                    "severity": r.log_level,
                    "module": r.module,
                    "error_code": r.error_code,
                },
            )
            for r in self.records
        ]

    def _index(self) -> None:
        """Build retriever indices."""
        if self.documents:
            self.retriever.index(self.documents)
            self._indexed = True

    # ── Core analysis ──────────────────────────────────────────────────────

    def analyze(self, query: str) -> Dict[str, Any]:
        """
        Run the adaptive iterative RAG pipeline.

        Returns a structured result dictionary:
            {
                "root_cause": str,
                "severity": str,
                "confidence": float,
                "supporting_logs": [str, ...],
                "reasoning_steps": [str, ...],
                "retrieval_scores": [float, ...],
                "iterations": [IterationResult, ...],
                "best_iteration": int,
                "converged": bool,
                "memory_context": str,
                "recommendation": str,
            }
        """
        if not self._indexed:
            raise RuntimeError("No logs indexed. Call load_logs() or load_records() first.")

        start_time = time.time()

        # Memory lookup
        memory_context = ""
        if self.memory and self.use_memory:
            memory_context = self.memory.get_context_for_query(query)

        # Iteration state
        iterations: List[IterationResult] = []
        current_query = query
        previous_analysis = ""
        previous_confidence = 0.0
        best_iteration_idx = 0
        best_confidence = 0.0
        best_scored_docs: List[ScoredDocument] = []
        consecutive_drops = 0
        seen_doc_contents: set = set()  # Track docs across iterations to avoid redundancy

        for i in range(1, self.max_iterations + 1):
            iter_start = time.time()

            # 1. Retrieve documents with cross-encoder reranking
            #    On iteration 2+, over-fetch to compensate for dedup filtering
            n_cand = 20 if i == 1 else 30
            raw_scored_docs = self.retriever.retrieve_and_rerank(
                current_query, top_k=self.top_k, n_candidates=n_cand
            )

            # Deduplicate: on iteration 2+, prefer NEW documents not seen before
            if i > 1 and seen_doc_contents:
                new_docs = [sd for sd in raw_scored_docs if sd.content not in seen_doc_contents]
                old_docs = [sd for sd in raw_scored_docs if sd.content in seen_doc_contents]
                # Keep some old high-scoring docs for context continuity, but prioritize new
                scored_docs = (new_docs + old_docs)[:self.top_k]
            else:
                scored_docs = raw_scored_docs[:self.top_k]

            # Track seen documents
            for sd in scored_docs:
                seen_doc_contents.add(sd.content)

            context_text = self.retriever.format_retrieved(scored_docs)
            retrieval_scores = [sd.final_score for sd in scored_docs]

            # 2. Run LLM analysis
            previous_findings = ""
            if previous_analysis:
                findings = QueryRefiner.extract_key_findings(previous_analysis)
                previous_findings = f"Previous iteration findings: {findings}"

            chain = self.ANALYSIS_PROMPT | self.llm | StrOutputParser()
            analysis = chain.invoke({
                "context": context_text,
                "question": current_query,
                "memory_context": memory_context if memory_context else "No prior incidents in memory.",
                "iteration": i,
                "max_iterations": self.max_iterations,
                "previous_findings": previous_findings,
            })

            # 3. Compute confidence score
            confidence = self._compute_confidence(
                analysis, scored_docs, previous_analysis
            )

            # 4. Record iteration
            improvement = confidence - previous_confidence
            iteration_result = IterationResult(
                iteration=i,
                query=current_query,
                confidence=confidence,
                improvement=improvement,
                analysis=analysis,
                retrieval_scores=retrieval_scores,
                num_docs_retrieved=len(scored_docs),
                timestamp=time.time() - iter_start,
            )
            iterations.append(iteration_result)

            # Track best iteration AND its scored_docs
            if confidence > best_confidence:
                best_confidence = confidence
                best_iteration_idx = i - 1  # 0-indexed
                best_scored_docs = scored_docs
                consecutive_drops = 0
            else:
                consecutive_drops += 1

            # 5. Stopping criteria (refined after expert review):
            #    - Allow ONE confidence dip (natural variance from query narrowing)
            #    - Stop on 2 consecutive non-improvements (true plateau/degradation)
            #    - High-confidence early exit at 0.85
            if consecutive_drops >= 2:
                break

            if i > 1 and improvement < self.convergence_threshold and consecutive_drops >= 1:
                break

            # High-confidence early exit: no need to iterate further
            if confidence >= 0.85:
                break

            # 6. Refine query for next iteration (if not last)
            if i < self.max_iterations:
                current_query = self.query_refiner.refine(
                    original_query=query,
                    previous_query=current_query,
                    findings_summary=QueryRefiner.extract_key_findings(analysis),
                    retrieved_docs_text=context_text,
                    iteration=i,
                    max_iterations=self.max_iterations,
                )

            previous_analysis = analysis
            previous_confidence = confidence

        # Select best iteration result
        best = iterations[best_iteration_idx]
        total_time = time.time() - start_time

        # Parse structured fields from best analysis
        parsed = self._parse_analysis(best.analysis)

        # Use best iteration's scored_docs (not last iteration's)
        final_docs = best_scored_docs if best_scored_docs else scored_docs

        result = {
            "root_cause": parsed.get("root_cause", ""),
            "severity": parsed.get("severity", ""),
            "confidence": best.confidence,
            "supporting_logs": [f"[{sd.metadata.get('source', '')}] {sd.content}" for sd in final_docs],
            "reasoning_steps": parsed.get("reasoning_steps", []),
            "retrieval_scores": best.retrieval_scores,
            "iterations": [it.to_dict() for it in iterations],
            "best_iteration": best_iteration_idx + 1,
            "converged": len(iterations) < self.max_iterations,
            "total_iterations": len(iterations),
            "memory_context": memory_context,
            "recommendation": parsed.get("recommendation", ""),
            "error_timeline": parsed.get("error_timeline", ""),
            "details": parsed.get("details", ""),
            "full_analysis": best.analysis,
            "latency_seconds": total_time,
            "confidence_trajectory": [it.confidence for it in iterations],
            "error_codes": list(set(r.error_code for r in self.records if r.error_code)),
            "modules_involved": list(set(r.module for r in self.records if r.module)),
        }

        # Store in memory
        if self.memory and self.use_memory:
            incident = self.memory.create_incident_from_result(query, result)
            self.memory.add_incident(incident)

        return result

    # ── Baseline methods for comparison ────────────────────────────────────

    def analyze_baseline(self, query: str) -> Dict[str, Any]:
        """
        Single-pass baseline RAG (no iteration, no query refinement).
        For evaluation comparison.
        """
        if not self._indexed:
            raise RuntimeError("No logs indexed.")

        start = time.time()
        scored_docs = self.retriever.retrieve_vector_only(query, self.top_k)
        context_text = self.retriever.format_retrieved(scored_docs)

        chain = self.ANALYSIS_PROMPT | self.llm | StrOutputParser()
        analysis = chain.invoke({
            "context": context_text,
            "question": query,
            "memory_context": "No prior incidents.",
            "iteration": 1,
            "max_iterations": 1,
            "previous_findings": "",
        })

        confidence = self._compute_confidence(analysis, scored_docs, "")
        parsed = self._parse_analysis(analysis)
        latency = time.time() - start

        return {
            "root_cause": parsed.get("root_cause", ""),
            "severity": parsed.get("severity", ""),
            "confidence": confidence,
            "supporting_logs": [f"[{sd.metadata.get('source', '')}] {sd.content}" for sd in scored_docs],
            "reasoning_steps": parsed.get("reasoning_steps", []),
            "retrieval_scores": [sd.final_score for sd in scored_docs],
            "iterations": [{"iteration": 1, "confidence": confidence}],
            "best_iteration": 1,
            "converged": True,
            "total_iterations": 1,
            "full_analysis": analysis,
            "latency_seconds": latency,
            "confidence_trajectory": [confidence],
        }

    def analyze_dense_rerank(self, query: str) -> Dict[str, Any]:
        """
        Dense + cross-encoder reranking, single pass (no iteration, no BM25).

        Isolates the contribution of cross-encoder reranking from the
        contribution of iterative refinement.
        """
        if not self._indexed:
            raise RuntimeError("No logs indexed.")

        start = time.time()

        # Stage 1: dense-only candidate pool (no BM25)
        candidates = self.retriever.retrieve_vector_only(query, top_k=max(20, self.top_k * 3))
        if not candidates:
            return {
                "root_cause": "", "severity": "", "confidence": 0.0,
                "supporting_logs": [], "reasoning_steps": [], "retrieval_scores": [],
                "iterations": [{"iteration": 1, "confidence": 0.0}],
                "best_iteration": 1, "converged": True, "total_iterations": 1,
                "full_analysis": "", "latency_seconds": time.time() - start,
                "confidence_trajectory": [0.0],
            }

        # Stage 2: cross-encoder reranking
        import numpy as np
        pairs = [(query, sd.content) for sd in candidates]
        ce_raw = self.retriever.reranker.predict(pairs)
        ce_scores = 1.0 / (1.0 + np.exp(-np.array(ce_raw)))
        for sd, ce_score in zip(candidates, ce_scores):
            sd.reranker_score = float(ce_score)
            sd.final_score = float(ce_score)
        candidates.sort(key=lambda x: x.final_score, reverse=True)
        scored_docs = candidates[:self.top_k]

        context_text = self.retriever.format_retrieved(scored_docs)

        chain = self.ANALYSIS_PROMPT | self.llm | StrOutputParser()
        analysis = chain.invoke({
            "context": context_text,
            "question": query,
            "memory_context": "No prior incidents.",
            "iteration": 1,
            "max_iterations": 1,
            "previous_findings": "",
        })

        confidence = self._compute_confidence(analysis, scored_docs, "")
        parsed = self._parse_analysis(analysis)
        latency = time.time() - start

        return {
            "root_cause": parsed.get("root_cause", ""),
            "severity": parsed.get("severity", ""),
            "confidence": confidence,
            "supporting_logs": [f"[{sd.metadata.get('source', '')}] {sd.content}" for sd in scored_docs],
            "reasoning_steps": parsed.get("reasoning_steps", []),
            "retrieval_scores": [sd.final_score for sd in scored_docs],
            "iterations": [{"iteration": 1, "confidence": confidence}],
            "best_iteration": 1,
            "converged": True,
            "total_iterations": 1,
            "full_analysis": analysis,
            "latency_seconds": latency,
            "confidence_trajectory": [confidence],
        }

    def analyze_fixed_iterative(self, query: str, num_iterations: int = 3) -> Dict[str, Any]:
        """
        Fixed-iteration RAG (iterates N times regardless of confidence).
        For evaluation comparison.
        """
        if not self._indexed:
            raise RuntimeError("No logs indexed.")

        start = time.time()
        current_query = query
        previous_analysis = ""
        iterations = []

        for i in range(1, num_iterations + 1):
            scored_docs = self.retriever.retrieve(current_query, self.top_k)
            context_text = self.retriever.format_retrieved(scored_docs)

            previous_findings = ""
            if previous_analysis:
                findings = QueryRefiner.extract_key_findings(previous_analysis)
                previous_findings = f"Previous iteration findings: {findings}"

            chain = self.ANALYSIS_PROMPT | self.llm | StrOutputParser()
            analysis = chain.invoke({
                "context": context_text,
                "question": current_query,
                "memory_context": "No prior incidents.",
                "iteration": i,
                "max_iterations": num_iterations,
                "previous_findings": previous_findings,
            })

            confidence = self._compute_confidence(analysis, scored_docs, previous_analysis)
            iterations.append({
                "iteration": i,
                "query": current_query,
                "confidence": round(confidence, 4),
            })

            if i < num_iterations:
                current_query = self.query_refiner.refine(
                    original_query=query,
                    previous_query=current_query,
                    findings_summary=QueryRefiner.extract_key_findings(analysis),
                    retrieved_docs_text=context_text,
                    iteration=i,
                    max_iterations=num_iterations,
                )

            previous_analysis = analysis

        parsed = self._parse_analysis(analysis)
        latency = time.time() - start

        return {
            "root_cause": parsed.get("root_cause", ""),
            "severity": parsed.get("severity", ""),
            "confidence": confidence,
            "supporting_logs": [f"[{sd.metadata.get('source', '')}] {sd.content}" for sd in scored_docs],
            "reasoning_steps": parsed.get("reasoning_steps", []),
            "retrieval_scores": [sd.final_score for sd in scored_docs],
            "iterations": iterations,
            "best_iteration": num_iterations,
            "converged": False,
            "total_iterations": num_iterations,
            "full_analysis": analysis,
            "latency_seconds": latency,
            "confidence_trajectory": [it["confidence"] for it in iterations],
        }

    # ── Confidence scoring ─────────────────────────────────────────────────

    def _compute_confidence(
        self,
        analysis: str,
        scored_docs: List[ScoredDocument],
        previous_analysis: str,
    ) -> float:
        """
        Multi-signal confidence scoring.

        Components:
            1. Retrieval quality (mean of top-k retrieval scores)   — weight 0.3
            2. Analysis completeness (section coverage heuristic)   — weight 0.3
            3. Evidence density (specific references in analysis)   — weight 0.2
            4. Consistency with previous iteration                  — weight 0.2
        """
        # 1. Retrieval quality: average final_score of retrieved docs
        if scored_docs:
            retrieval_quality = sum(sd.final_score for sd in scored_docs) / len(scored_docs)
        else:
            retrieval_quality = 0.0

        # 2. Analysis completeness: check for expected sections
        expected_sections = [
            "root cause", "severity", "error timeline",
            "details", "recommendation",
        ]
        analysis_lower = analysis.lower()
        sections_found = sum(1 for s in expected_sections if s in analysis_lower)
        completeness = sections_found / len(expected_sections)

        # 3. Evidence density: count specific references (file:line, timestamps)
        file_refs = len(re.findall(r'\w+\.\w+:L?\d+', analysis))
        timestamp_refs = len(re.findall(r'\d{2}:\d{2}:\d{2}', analysis))
        ue_refs = len(re.findall(r'UE\d+', analysis))
        evidence_count = file_refs + timestamp_refs + ue_refs
        # Normalize: expect ~10 references for a thorough analysis
        evidence_density = min(evidence_count / 10.0, 1.0)

        # 4. Consistency with previous analysis
        consistency = 0.5  # neutral if no previous
        if previous_analysis:
            # Extract root causes and compare
            curr_rc = self._extract_section(analysis, "Root Cause")
            prev_rc = self._extract_section(previous_analysis, "Root Cause")
            if curr_rc and prev_rc:
                # Simple word overlap
                curr_words = set(curr_rc.lower().split())
                prev_words = set(prev_rc.lower().split())
                if curr_words | prev_words:
                    overlap = len(curr_words & prev_words) / len(curr_words | prev_words)
                    consistency = overlap

        # Weighted combination
        confidence = (
            0.3 * retrieval_quality
            + 0.3 * completeness
            + 0.2 * evidence_density
            + 0.2 * consistency
        )

        return min(max(confidence, 0.0), 1.0)

    # ── Analysis parsing ───────────────────────────────────────────────────

    @staticmethod
    def _extract_section(text: str, section_name: str) -> str:
        """Extract content under a markdown ## section heading."""
        pattern = rf'##\s*{re.escape(section_name)}\s*\n(.*?)(?=\n##|\Z)'
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    @classmethod
    def _parse_analysis(cls, analysis: str) -> Dict[str, Any]:
        """Parse structured sections from LLM analysis output."""
        result: Dict[str, Any] = {}

        result["root_cause"] = cls._extract_section(analysis, "Root Cause")
        result["severity"] = cls._extract_section(analysis, "Severity").strip().upper()
        result["error_timeline"] = cls._extract_section(analysis, "Error Timeline")
        result["details"] = cls._extract_section(analysis, "Details")
        result["recommendation"] = cls._extract_section(analysis, "Recommendation")

        # Parse reasoning steps as a list
        reasoning_text = cls._extract_section(analysis, "Reasoning Steps")
        steps = []
        for line in reasoning_text.split("\n"):
            line = line.strip()
            if line and re.match(r'\d+[\.\)]', line):
                steps.append(re.sub(r'^\d+[\.\)]\s*', '', line))
        result["reasoning_steps"] = steps if steps else [reasoning_text] if reasoning_text else []

        return result

    # ── Utility ────────────────────────────────────────────────────────────

    @property
    def num_records(self) -> int:
        return len(self.records)

    @property
    def severity_summary(self) -> Dict[str, int]:
        return TelecomLogParser.severity_counts(self.records)
