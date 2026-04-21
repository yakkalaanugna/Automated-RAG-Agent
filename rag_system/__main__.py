"""
__main__.py — CLI entry point for the RAG system.

Usage:
    python -m rag_system "Why did UE4 fail?"
    python -m rag_system "What caused the packet loss?" --logs data/logs/
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        prog="rag_system",
        description="Analyzing the Retrieval–Reasoning Gap in Adaptive RAG Systems",
    )
    parser.add_argument("query", help="Analysis query (e.g., 'Why did UE4 fail?')")
    parser.add_argument("--logs", default="data/logs", help="Path to log files directory")
    parser.add_argument("--top-k", type=int, default=6, help="Number of documents to retrieve per iteration")
    parser.add_argument("--max-iter", type=int, default=3, help="Maximum iterations")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: Set GROQ_API_KEY in .env or as environment variable", file=sys.stderr)
        sys.exit(1)

    from .adaptive_agent import AdaptiveIterativeRAGAgent

    agent = AdaptiveIterativeRAGAgent(
        groq_api_key=api_key,
        max_iterations=args.max_iter,
        top_k=args.top_k,
    )

    records = agent.load_logs(args.logs)
    print(f"Parsed {len(records)} log records from {args.logs}", file=sys.stderr)

    result = agent.analyze(args.query)

    if args.json:
        # Remove non-serializable fields
        output = {k: v for k, v in result.items() if k != "full_analysis"}
        print(json.dumps(output, indent=2, default=str))
    else:
        print()
        print(f"Root Cause: {result.get('root_cause', 'N/A')}")
        print(f"Severity:   {result.get('severity', 'N/A')}")
        print(f"Confidence: {result.get('confidence', 0):.3f}")
        print(f"Iterations: {result.get('total_iterations', 1)} (best: {result.get('best_iteration', 1)})")
        print(f"Latency:    {result.get('latency_seconds', 0):.1f}s")
        print()
        print("Supporting Evidence:")
        for i, log in enumerate(result.get("supporting_logs", [])[:5], 1):
            print(f"  {i}. {log[:120]}")
        print()
        if result.get("recommendation"):
            print(f"Recommendation: {result['recommendation']}")


if __name__ == "__main__":
    main()
