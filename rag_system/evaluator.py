"""
evaluator.py — RAG Evaluation Framework

Implements standard IR and RAG-specific evaluation metrics for
comparing baseline, iterative, and adaptive iterative RAG approaches.

Metrics:
    - precision@k       — fraction of retrieved docs that are relevant
    - recall@k          — fraction of relevant docs that were retrieved
    - root_cause_accuracy — whether the correct root cause was identified
    - iteration_count   — number of iterations until convergence
    - latency           — wall-clock time for full pipeline execution
    - confidence_trajectory — confidence score across iterations

Returns results as pandas DataFrames for analysis and visualization.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd


# ─── Evaluation record ─────────────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    """Results from a single evaluation run."""
    method: str                             # "baseline" | "iterative" | "adaptive"
    query: str
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    root_cause_accuracy: float = 0.0        # 1.0 if correct, 0.0 otherwise
    num_iterations: int = 1
    latency_seconds: float = 0.0
    final_confidence: float = 0.0
    confidence_trajectory: List[float] = field(default_factory=list)
    retrieval_scores: List[float] = field(default_factory=list)
    predicted_root_cause: str = ""
    ground_truth_root_cause: str = ""


# ─── Ground truth definition ──────────────────────────────────────────────────

@dataclass
class GroundTruth:
    """Ground truth for a test query."""
    query: str
    relevant_doc_ids: List[str]       # identifiers of relevant documents
    root_cause: str                    # expected root cause string
    root_cause_keywords: List[str]     # keywords that must appear in a correct answer
    severity: str = ""


# ─── RAG Evaluator ─────────────────────────────────────────────────────────────

class RAGEvaluator:
    """
    Evaluation framework for comparing RAG system variants.

    Supports three comparison modes:
        1. Baseline RAG     — single-pass retrieval + LLM
        2. Iterative RAG    — fixed N iterations
        3. Adaptive RAG     — confidence-gated adaptive iteration

    Usage:
        evaluator = RAGEvaluator()
        evaluator.add_ground_truth(GroundTruth(...))
        results = evaluator.evaluate_all(agent, queries)
        df = evaluator.to_dataframe(results)
    """

    def __init__(self):
        self.ground_truths: Dict[str, GroundTruth] = {}
        self.results: List[EvaluationResult] = []

    # ── Ground truth management ────────────────────────────────────────────

    def add_ground_truth(self, gt: GroundTruth) -> None:
        self.ground_truths[gt.query] = gt

    def add_ground_truths(self, gts: List[GroundTruth]) -> None:
        for gt in gts:
            self.add_ground_truth(gt)

    # ── Metric computation ─────────────────────────────────────────────────

    @staticmethod
    def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """
        Precision@k: fraction of top-k retrieved documents that are relevant.

        Args:
            retrieved_ids: ordered list of retrieved document identifiers
            relevant_ids:  set of ground-truth relevant document identifiers
            k:             cutoff rank
        """
        if k == 0:
            return 0.0
        top_k = retrieved_ids[:k]
        relevant_set = set(relevant_ids)
        hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
        return hits / k

    @staticmethod
    def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """
        Recall@k: fraction of relevant documents appearing in top-k results.
        """
        if not relevant_ids:
            return 0.0
        top_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        hits = len(top_k & relevant_set)
        return hits / len(relevant_set)

    @staticmethod
    def root_cause_match(predicted: str, ground_truth_keywords: List[str]) -> float:
        """
        Root cause accuracy: 1.0 if all keywords present in prediction, else partial.

        Uses keyword overlap to allow flexible matching of root cause descriptions.
        """
        if not ground_truth_keywords:
            return 0.0
        predicted_lower = predicted.lower()
        matches = sum(1 for kw in ground_truth_keywords if kw.lower() in predicted_lower)
        return matches / len(ground_truth_keywords)

    # ── Single evaluation ──────────────────────────────────────────────────

    def evaluate_single(
        self,
        method: str,
        query: str,
        retrieved_doc_contents: List[str],
        predicted_root_cause: str,
        confidence: float,
        num_iterations: int,
        latency: float,
        confidence_trajectory: List[float],
        retrieval_scores: List[float],
    ) -> EvaluationResult:
        """
        Evaluate a single query result against ground truth.
        """
        gt = self.ground_truths.get(query)

        # Compute metrics against ground truth if available
        precision = 0.0
        recall = 0.0
        rc_accuracy = 0.0

        if gt:
            # For precision/recall, match by content substring
            retrieved_ids = []
            for content in retrieved_doc_contents:
                for rel_id in gt.relevant_doc_ids:
                    if rel_id.lower() in content.lower():
                        retrieved_ids.append(rel_id)
                        break
                else:
                    retrieved_ids.append(f"irrelevant_{len(retrieved_ids)}")

            k = len(retrieved_doc_contents)
            precision = self.precision_at_k(retrieved_ids, gt.relevant_doc_ids, k)
            recall = self.recall_at_k(retrieved_ids, gt.relevant_doc_ids, k)
            rc_accuracy = self.root_cause_match(predicted_root_cause, gt.root_cause_keywords)

        result = EvaluationResult(
            method=method,
            query=query,
            precision_at_k=precision,
            recall_at_k=recall,
            root_cause_accuracy=rc_accuracy,
            num_iterations=num_iterations,
            latency_seconds=latency,
            final_confidence=confidence,
            confidence_trajectory=confidence_trajectory,
            retrieval_scores=retrieval_scores,
            predicted_root_cause=predicted_root_cause,
            ground_truth_root_cause=gt.root_cause if gt else "",
        )

        self.results.append(result)
        return result

    # ── Batch evaluation with agent ────────────────────────────────────────

    def evaluate_agent(
        self,
        agent,
        queries: List[str],
        method_name: str = "adaptive",
    ) -> List[EvaluationResult]:
        """
        Run an AdaptiveIterativeRAGAgent on a list of queries and evaluate.

        Args:
            agent:       An AdaptiveIterativeRAGAgent instance (must have .analyze())
            queries:     List of query strings
            method_name: Label for this method in results

        Returns:
            List of EvaluationResult objects
        """
        results = []
        for query in queries:
            start = time.time()
            analysis = agent.analyze(query)
            latency = time.time() - start

            result = self.evaluate_single(
                method=method_name,
                query=query,
                retrieved_doc_contents=analysis.get("supporting_logs", []),
                predicted_root_cause=analysis.get("root_cause", ""),
                confidence=analysis.get("confidence", 0.0),
                num_iterations=len(analysis.get("iterations", [1])),
                latency=latency,
                confidence_trajectory=[
                    it.get("confidence", 0.0)
                    for it in analysis.get("iterations", [])
                ],
                retrieval_scores=analysis.get("retrieval_scores", []),
            )
            results.append(result)

        return results

    # ── Results aggregation ────────────────────────────────────────────────

    def to_dataframe(self, results: Optional[List[EvaluationResult]] = None) -> pd.DataFrame:
        """Convert evaluation results to a pandas DataFrame."""
        data = results or self.results
        rows = []
        for r in data:
            rows.append({
                "Method": r.method,
                "Query": r.query,
                "Precision@K": round(r.precision_at_k, 4),
                "Recall@K": round(r.recall_at_k, 4),
                "Root Cause Accuracy": round(r.root_cause_accuracy, 4),
                "Iterations": r.num_iterations,
                "Latency (s)": round(r.latency_seconds, 2),
                "Final Confidence": round(r.final_confidence, 4),
                "Predicted Root Cause": r.predicted_root_cause[:100],
            })
        return pd.DataFrame(rows)

    def summary_by_method(self, results: Optional[List[EvaluationResult]] = None) -> pd.DataFrame:
        """Aggregate metrics by method (baseline / iterative / adaptive)."""
        df = self.to_dataframe(results)
        if df.empty:
            return df
        return df.groupby("Method").agg({
            "Precision@K": "mean",
            "Recall@K": "mean",
            "Root Cause Accuracy": "mean",
            "Iterations": "mean",
            "Latency (s)": "mean",
            "Final Confidence": "mean",
        }).round(4).reset_index()

    def clear_results(self) -> None:
        self.results = []

    # ── Visualization data helpers ─────────────────────────────────────────

    def get_confidence_trajectories(
        self, results: Optional[List[EvaluationResult]] = None,
    ) -> Dict[str, List[List[float]]]:
        """
        Get confidence trajectories grouped by method.

        Returns: {method_name: [[conf_iter1, conf_iter2, ...], ...]}
        """
        data = results or self.results
        trajectories: Dict[str, List[List[float]]] = {}
        for r in data:
            trajectories.setdefault(r.method, []).append(r.confidence_trajectory)
        return trajectories

    def get_comparison_data(
        self, results: Optional[List[EvaluationResult]] = None,
    ) -> pd.DataFrame:
        """
        Get per-query comparison across methods for plotting.
        """
        data = results or self.results
        rows = []
        for r in data:
            rows.append({
                "Method": r.method,
                "Query": r.query[:50],
                "Precision@K": r.precision_at_k,
                "Recall@K": r.recall_at_k,
                "Root Cause Accuracy": r.root_cause_accuracy,
                "Iterations": r.num_iterations,
                "Confidence": r.final_confidence,
                "Latency": r.latency_seconds,
            })
        return pd.DataFrame(rows)
