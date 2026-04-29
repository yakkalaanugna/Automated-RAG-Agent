#!/usr/bin/env python3
"""
run_ablation.py — Full ablation study across 6 RAG configurations.

Configurations:
    1. Dense-only retrieval (no BM25, no reranker, no iteration)
    2. BM25-only retrieval (no dense, no reranker, no iteration)
    3. Hybrid retrieval (BM25 + dense, no reranker, no iteration)
    4. Hybrid + cross-encoder reranking (no iteration)
    5. Hybrid + iteration (no reranking)
    6. Full system (hybrid + reranking + iteration)

Metrics per configuration:
    - Precision@K
    - Recall@K
    - MRR (Mean Reciprocal Rank)
    - Root Cause Accuracy (keyword overlap)

Output:
    - results/ablation_results.csv
    - results/ablation_summary.csv
    - results/ablation_detailed.json

Usage:
    python run_ablation.py [--logs data/synthetic_logs] [--skip-judge]
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

import functools
print = functools.partial(print, flush=True)


# ─── Metrics ─────────────────────────────────────────────────────────────────

def precision_at_k(retrieved_ids, relevant_ids, k):
    """Fraction of top-k retrieved that are relevant."""
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return hits / k


def recall_at_k(retrieved_ids, relevant_ids, k):
    """Fraction of relevant documents appearing in top-k."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    hits = len(top_k & relevant_set)
    return hits / len(relevant_set)


def mrr(retrieved_ids, relevant_ids):
    """Mean Reciprocal Rank — 1/rank of first relevant document."""
    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def root_cause_match(predicted, keywords):
    """Fractional keyword overlap score."""
    if not keywords:
        return 0.0
    predicted_lower = predicted.lower()
    matches = sum(1 for kw in keywords if kw.lower() in predicted_lower)
    return matches / len(keywords)


# ─── Configuration Runners ───────────────────────────────────────────────────

class AblationRunner:
    """Runs all 6 ablation configurations on the same indexed corpus."""
    
    def __init__(self, agent):
        self.agent = agent
    
    def run_dense_only(self, query):
        """Config 1: Dense-only retrieval, single pass."""
        return self.agent.analyze_baseline(query)
    
    def run_bm25_only(self, query):
        """Config 2: BM25-only retrieval, single pass."""
        if not self.agent._indexed:
            raise RuntimeError("No logs indexed.")
        
        start = time.time()
        scored_docs = self.agent.retriever.retrieve_bm25_only(query, self.agent.top_k)
        context_text = self.agent.retriever.format_retrieved(scored_docs)
        
        from langchain_core.output_parsers import StrOutputParser
        chain = self.agent.ANALYSIS_PROMPT | self.agent.llm | StrOutputParser()
        analysis = chain.invoke({
            "context": context_text,
            "question": query,
            "memory_context": "No prior incidents.",
            "iteration": 1,
            "max_iterations": 1,
            "previous_findings": "",
        })
        
        confidence = self.agent._compute_confidence(analysis, scored_docs, "")
        parsed = self.agent._parse_analysis(analysis)
        latency = time.time() - start
        
        return {
            "root_cause": parsed.get("root_cause", ""),
            "severity": parsed.get("severity", ""),
            "confidence": confidence,
            "supporting_logs": [f"[{sd.metadata.get('source', '')}] {sd.content}" for sd in scored_docs],
            "reasoning_steps": parsed.get("reasoning_steps", []),
            "retrieval_scores": [sd.final_score for sd in scored_docs],
            "total_iterations": 1,
            "latency_seconds": latency,
            "confidence_trajectory": [confidence],
        }
    
    def run_hybrid_no_rerank(self, query):
        """Config 3: Hybrid (BM25 + dense), no reranking, single pass."""
        if not self.agent._indexed:
            raise RuntimeError("No logs indexed.")
        
        start = time.time()
        scored_docs = self.agent.retriever.retrieve(query, self.agent.top_k)
        context_text = self.agent.retriever.format_retrieved(scored_docs)
        
        from langchain_core.output_parsers import StrOutputParser
        chain = self.agent.ANALYSIS_PROMPT | self.agent.llm | StrOutputParser()
        analysis = chain.invoke({
            "context": context_text,
            "question": query,
            "memory_context": "No prior incidents.",
            "iteration": 1,
            "max_iterations": 1,
            "previous_findings": "",
        })
        
        confidence = self.agent._compute_confidence(analysis, scored_docs, "")
        parsed = self.agent._parse_analysis(analysis)
        latency = time.time() - start
        
        return {
            "root_cause": parsed.get("root_cause", ""),
            "severity": parsed.get("severity", ""),
            "confidence": confidence,
            "supporting_logs": [f"[{sd.metadata.get('source', '')}] {sd.content}" for sd in scored_docs],
            "reasoning_steps": parsed.get("reasoning_steps", []),
            "retrieval_scores": [sd.final_score for sd in scored_docs],
            "total_iterations": 1,
            "latency_seconds": latency,
            "confidence_trajectory": [confidence],
        }
    
    def run_hybrid_rerank(self, query):
        """Config 4: Hybrid + cross-encoder reranking, single pass."""
        return self.agent.analyze_dense_rerank(query)
    
    def run_hybrid_iterative(self, query):
        """Config 5: Hybrid + iteration (no reranking)."""
        return self.agent.analyze_fixed_iterative(query, num_iterations=3)
    
    def run_full_system(self, query):
        """Config 6: Full system (hybrid + reranking + iteration)."""
        return self.agent.analyze(query)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run ablation study on RAG configurations")
    parser.add_argument("--logs", default="data/synthetic_logs", help="Path to log files")
    parser.add_argument("--queries", default="data/synthetic_eval_queries.json", help="Path to queries JSON")
    parser.add_argument("--out", default="results", help="Output directory")
    parser.add_argument("--max-queries", type=int, default=None, help="Limit number of queries (for testing)")
    parser.add_argument("--skip-judge", action="store_true", help="Skip LLM-as-judge evaluation")
    args = parser.parse_args()
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: Set GROQ_API_KEY in .env")
        sys.exit(1)
    
    from rag_system.adaptive_agent import AdaptiveIterativeRAGAgent
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load queries
    with open(args.queries, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    queries = dataset["queries"]
    if args.max_queries:
        queries = queries[:args.max_queries]
    
    print(f"Loaded {len(queries)} queries from {args.queries}")
    
    # Setup agent
    agent = AdaptiveIterativeRAGAgent(
        groq_api_key=api_key,
        max_iterations=3,
        top_k=6,
    )
    
    print(f"Loading logs from {args.logs}...")
    records = agent.load_logs(args.logs)
    print(f"Parsed {len(records)} log records.\n")
    
    runner = AblationRunner(agent)
    
    # Define configurations
    configs = [
        ("Dense-Only", runner.run_dense_only),
        ("BM25-Only", runner.run_bm25_only),
        ("Hybrid", runner.run_hybrid_no_rerank),
        ("Hybrid+Rerank", runner.run_hybrid_rerank),
        ("Hybrid+Iteration", runner.run_hybrid_iterative),
        ("Full-System", runner.run_full_system),
    ]
    
    # Run evaluation
    all_results = []
    
    for config_name, config_fn in configs:
        print("=" * 70)
        print(f"Configuration: {config_name}")
        print("=" * 70)
        
        for qi, q in enumerate(queries):
            query_text = q["query"]
            gt = q["ground_truth"]
            
            start = time.time()
            try:
                result = config_fn(query_text)
            except Exception as e:
                print(f"  [{qi+1:3d}] ERROR: {e}")
                result = {"supporting_logs": [], "root_cause": "", "confidence": 0.0,
                          "retrieval_scores": [], "total_iterations": 1, "latency_seconds": time.time() - start}
            latency = time.time() - start
            
            # Compute retrieval metrics
            retrieved_files = []
            for log_text in result.get("supporting_logs", []):
                matched = False
                for rel_file in gt["relevant_files"]:
                    if rel_file.lower() in log_text.lower():
                        retrieved_files.append(rel_file)
                        matched = True
                        break
                if not matched:
                    retrieved_files.append(f"irrelevant_{len(retrieved_files)}")
            
            k = len(result.get("supporting_logs", []))
            p_at_k = precision_at_k(retrieved_files, gt["relevant_files"], k) if k > 0 else 0.0
            r_at_k = recall_at_k(retrieved_files, gt["relevant_files"], k) if k > 0 else 0.0
            mrr_score = mrr(retrieved_files, gt["relevant_files"])
            rc_acc = root_cause_match(result.get("root_cause", ""), q.get("keywords", []))
            
            row = {
                "config": config_name,
                "query_id": q["id"],
                "query": query_text,
                "query_type": q["type"],
                "difficulty": q["difficulty"],
                "scenario_id": q["scenario_id"],
                "precision_at_k": round(p_at_k, 4),
                "recall_at_k": round(r_at_k, 4),
                "mrr": round(mrr_score, 4),
                "root_cause_accuracy": round(rc_acc, 4),
                "confidence": round(result.get("confidence", 0.0), 4),
                "iterations": result.get("total_iterations", 1),
                "latency_s": round(latency, 2),
                "predicted_root_cause": result.get("root_cause", ""),
            }
            all_results.append(row)
            
            if (qi + 1) % 5 == 0 or qi == len(queries) - 1:
                print(f"  [{qi+1:3d}/{len(queries)}] P@K={p_at_k:.3f} R@K={r_at_k:.3f} MRR={mrr_score:.3f} RCA={rc_acc:.3f} [{latency:.1f}s]")
        
        print()
    
    # Save detailed results
    import pandas as pd
    
    df = pd.DataFrame(all_results)
    df.to_csv(out_dir / "ablation_results.csv", index=False)
    
    # Compute summary
    summary = df.groupby("config").agg({
        "precision_at_k": "mean",
        "recall_at_k": "mean",
        "mrr": "mean",
        "root_cause_accuracy": "mean",
        "confidence": "mean",
        "iterations": "mean",
        "latency_s": "mean",
    }).round(4).reset_index()
    
    summary.columns = ["Configuration", "Avg P@K", "Avg R@K", "Avg MRR",
                       "Avg RCA", "Avg Confidence", "Avg Iterations", "Avg Latency (s)"]
    
    summary.to_csv(out_dir / "ablation_summary.csv", index=False)
    
    # Summary by query type
    type_summary = df.groupby(["config", "query_type"]).agg({
        "precision_at_k": "mean",
        "recall_at_k": "mean",
        "mrr": "mean",
        "root_cause_accuracy": "mean",
    }).round(4).reset_index()
    type_summary.to_csv(out_dir / "ablation_by_query_type.csv", index=False)
    
    # Save detailed JSON
    with open(out_dir / "ablation_detailed.json", "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "num_queries": len(queries),
                "num_configs": len(configs),
                "configs": [c[0] for c in configs],
            },
            "results": all_results,
        }, f, indent=2)
    
    # Print summary table
    print("\n" + "=" * 70)
    print("ABLATION STUDY RESULTS SUMMARY")
    print("=" * 70)
    print(summary.to_string(index=False))
    print()
    
    # Print comparison insights
    print("\nKEY COMPARISONS:")
    print("-" * 50)
    for metric in ["Avg P@K", "Avg R@K", "Avg MRR", "Avg RCA"]:
        best = summary.loc[summary[metric].idxmax()]
        worst = summary.loc[summary[metric].idxmin()]
        print(f"  {metric:12s}: Best={best['Configuration']:20s} ({best[metric]:.4f})")
        print(f"  {'':12s}  Worst={worst['Configuration']:20s} ({worst[metric]:.4f})")
        print(f"  {'':12s}  Delta={best[metric] - worst[metric]:.4f}")
        print()
    
    print(f"\nWrote: {out_dir / 'ablation_results.csv'}")
    print(f"Wrote: {out_dir / 'ablation_summary.csv'}")
    print(f"Wrote: {out_dir / 'ablation_by_query_type.csv'}")
    print(f"Wrote: {out_dir / 'ablation_detailed.json'}")


if __name__ == "__main__":
    main()
