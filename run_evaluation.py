#!/usr/bin/env python3
"""
run_evaluation.py — End-to-end evaluation of four RAG configurations
on an expanded telecom-log query set.

Configurations:
    1. Baseline             — dense-only retrieval, single pass
    2. Dense + Rerank       — dense retrieval + cross-encoder, single pass
    3. Fixed Iterative      — hybrid retrieval, 3 fixed iterations, no rerank
    4. Adaptive (ours)      — hybrid + cross-encoder + confidence-gated iter.

Outputs (written to results/):
    - eval_per_query.csv            per-query, per-method metrics
    - eval_summary.csv              aggregated metrics by method
    - eval_cases.json               top-k docs, predictions, GT per run
                                    (used for qualitative case analysis)
    - eval_confidence.csv           (confidence, correctness) pairs
                                    (used for correlation analysis)

Usage:
    python run_evaluation.py                    # requires GROQ_API_KEY in .env
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

# Make print default to flushing so we see progress under redirection.
import functools
print = functools.partial(print, flush=True)  # type: ignore


# ─── Expanded query set (12 queries derived from the same failure chain) ─────
#
# Ground truth is derived directly from the logs in data/logs/*.txt.
# No new data is introduced; we only ask more questions about the same
# underlying failure scenario (UE4 RRC reconfiguration failure -> CU-CP
# UE context release -> packet forward-jump). Queries vary by
# specificity (broad / targeted / causal-chain / time-anchored / contrastive).

def build_ground_truths():
    from rag_system.evaluator import GroundTruth
    return [
        # ── Broad queries ────────────────────────────────────────────────
        GroundTruth(
            query="Why did UE4 fail?",
            relevant_doc_ids=["log1.txt", "log2.txt", "log3.txt"],
            root_cause="RRC Reconfiguration failure code 4 in rfma_impl.cpp caused UE4 release",
            root_cause_keywords=["rrcreconfiguration", "failure", "code 4", "rfma_impl", "ue4", "release"],
            severity="CRITICAL",
        ),
        GroundTruth(
            query="Summarize the failure scenario in this session.",
            relevant_doc_ids=["log1.txt", "log2.txt", "log3.txt"],
            root_cause="UE4 RRC reconfiguration failure followed by CU-CP-initiated context release and packet loss",
            root_cause_keywords=["ue4", "rrc", "reconfiguration", "release", "cu-cp"],
            severity="CRITICAL",
        ),
        GroundTruth(
            query="What is the root cause of the incident?",
            relevant_doc_ids=["log1.txt", "log2.txt"],
            root_cause="Failure while applying RRCReconfiguration (code 4) in rfma_impl.cpp",
            root_cause_keywords=["rrcreconfiguration", "code 4", "rfma_impl", "failure"],
            severity="CRITICAL",
        ),
        # ── Targeted queries ─────────────────────────────────────────────
        GroundTruth(
            query="What caused the RRC reconfiguration failure?",
            relevant_doc_ids=["log1.txt", "log2.txt"],
            root_cause="Failure (code 4) while applying RRCReconfiguration message in rfma_impl.cpp for UE4",
            root_cause_keywords=["rrcreconfiguration", "code 4", "rfma_impl", "ue4"],
            severity="CRITICAL",
        ),
        GroundTruth(
            query="Which source file and line reported the critical error?",
            relevant_doc_ids=["log2.txt"],
            root_cause="rfma_impl.cpp line 80 reported the RRCReconfiguration failure",
            root_cause_keywords=["rfma_impl", "80"],
            severity="CRITICAL",
        ),
        GroundTruth(
            query="What error code was reported for the RRC failure?",
            relevant_doc_ids=["log1.txt", "log2.txt"],
            root_cause="Error code 4 was reported while applying RRCReconfiguration",
            root_cause_keywords=["code 4", "rrcreconfiguration"],
            severity="CRITICAL",
        ),
        # ── Causal-chain queries ─────────────────────────────────────────
        GroundTruth(
            query="What caused the UE context release?",
            relevant_doc_ids=["log1.txt", "log3.txt"],
            root_cause="UE Context Release triggered by CU-CP after RRC reconfiguration failure",
            root_cause_keywords=["ue context release", "cu-cp", "rrc", "failure"],
            severity="CRITICAL",
        ),
        GroundTruth(
            query="What event led the CU-CP to trigger UE release?",
            relevant_doc_ids=["log1.txt", "log3.txt"],
            root_cause="CU-CP triggered UE release after UE4 RRC reconfiguration failure (code 4)",
            root_cause_keywords=["cu-cp", "trigger", "release", "rrc", "ue4"],
            severity="CRITICAL",
        ),
        GroundTruth(
            query="Why were packets lost for UE4?",
            relevant_doc_ids=["log3.txt", "log1.txt"],
            root_cause="Packet loss due to long forward jump after UE4 RRC failure and release",
            root_cause_keywords=["forward jump", "packet", "lost", "ue4"],
            severity="CRITICAL",
        ),
        GroundTruth(
            query="What chain of events led to packet loss after registration?",
            relevant_doc_ids=["log1.txt", "log2.txt", "log3.txt"],
            root_cause="Registration -> PDU session establishment -> RRC reconfiguration failure (code 4) -> CU-CP UE release -> long forward jump / packet loss",
            root_cause_keywords=["rrc", "failure", "release", "forward jump", "packet"],
            severity="CRITICAL",
        ),
        # ── Time-anchored query ──────────────────────────────────────────
        GroundTruth(
            query="What happened around 18:34:08?",
            relevant_doc_ids=["log1.txt", "log2.txt", "log3.txt"],
            root_cause="At 18:34:08 UE4 RRC reconfiguration failed (code 4) and CU-CP began triggering UE release",
            root_cause_keywords=["18:34:08", "rrcreconfiguration", "failure", "ue4"],
            severity="CRITICAL",
        ),
        # ── Contrastive query (tests distractor handling) ────────────────
        GroundTruth(
            query="Did UE1 experience the same failure as UE4?",
            relevant_doc_ids=["log1.txt", "log2.txt"],
            root_cause="No. UE1 received a normal RRC Release while registered; UE4 suffered an RRCReconfiguration failure (code 4).",
            root_cause_keywords=["ue1", "rrc release", "ue4", "reconfiguration"],
            severity="INFO",
        ),
    ]


# ─── LLM-as-judge scoring (binary correctness) ───────────────────────────────

JUDGE_PROMPT = """You are evaluating whether an automated root-cause diagnosis
is correct for a telecom-log failure analysis task.

QUERY:
{query}

GROUND TRUTH ROOT CAUSE:
{ground_truth}

PREDICTED ROOT CAUSE:
{predicted}

Decide whether the PREDICTED root cause correctly identifies the same
underlying cause as the GROUND TRUTH. Minor wording differences are fine;
the prediction must agree on the *mechanism* (which component failed, what
error, and its downstream effect where applicable). Partial or wrong
attributions count as Incorrect.

Respond with exactly one word on the first line: "Correct" or "Incorrect".
On the next line give a one-sentence justification.
"""


def llm_judge(llm, query: str, ground_truth: str, predicted: str) -> dict:
    """Binary Yes/No correctness judgment via the same LLM used in the pipeline."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    if not predicted.strip():
        return {"judgment": "Incorrect", "reason": "Empty prediction."}

    prompt = ChatPromptTemplate.from_template(JUDGE_PROMPT)
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({
        "query": query,
        "ground_truth": ground_truth,
        "predicted": predicted,
    }).strip()
    first_line = raw.splitlines()[0].strip() if raw else ""
    verdict = "Correct" if first_line.lower().startswith("correct") else "Incorrect"
    reason = "\n".join(raw.splitlines()[1:]).strip() if "\n" in raw else ""
    return {"judgment": verdict, "reason": reason, "raw": raw}


# ─── Correlation helpers ─────────────────────────────────────────────────────

def pearson(xs, ys):
    import statistics as st
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return float("nan")
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx * dy else float("nan")


def spearman(xs, ys):
    def ranks(a):
        idx = sorted(range(len(a)), key=lambda i: a[i])
        r = [0] * len(a)
        for rank, i in enumerate(idx, start=1):
            r[i] = rank
        return r
    if len(xs) < 2:
        return float("nan")
    return pearson(ranks(xs), ranks(ys))


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG methods on telecom logs")
    parser.add_argument("--logs", default="data/logs", help="Path to log files directory")
    parser.add_argument("--api-key", default=None, help="Groq API key (or set GROQ_API_KEY env var)")
    parser.add_argument("--out", default="results", help="Output directory")
    parser.add_argument("--skip-judge", action="store_true", help="Skip LLM-as-judge pass")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: Set GROQ_API_KEY in .env or pass --api-key")
        sys.exit(1)

    from rag_system.adaptive_agent import AdaptiveIterativeRAGAgent
    from rag_system.evaluator import RAGEvaluator

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ground_truths = build_ground_truths()
    queries = [gt.query for gt in ground_truths]
    gt_by_query = {gt.query: gt for gt in ground_truths}

    # ── Setup ──────────────────────────────────────────────────────────────
    evaluator = RAGEvaluator()
    evaluator.add_ground_truths(ground_truths)

    agent = AdaptiveIterativeRAGAgent(
        groq_api_key=api_key,
        max_iterations=3,
        top_k=6,
    )

    print(f"Loading logs from {args.logs}...")
    records = agent.load_logs(args.logs)
    print(f"Parsed {len(records)} log records.")
    print(f"Evaluating {len(queries)} queries across 4 methods.\n")

    # Artifact store: per (method, query) -> run details for case analysis
    cases = []

    def flush_partial():
        """Write whatever we have so far so a mid-run crash still yields data."""
        try:
            with (out_dir / "eval_cases.json").open("w", encoding="utf-8") as f:
                json.dump(cases, f, indent=2)
            evaluator.to_dataframe().to_csv(out_dir / "eval_per_query.csv", index=False)
            evaluator.summary_by_method().to_csv(out_dir / "eval_summary.csv", index=False)
        except Exception as e:
            print(f"  (partial flush failed: {e})")

    def run_method(label, fn, iters_fn=lambda r: r.get("total_iterations", 1)):
        print("=" * 64)
        print(f"Running {label}...")
        print("=" * 64)
        for query in queries:
            start = time.time()
            try:
                result = fn(query)
            except Exception as e:
                latency = time.time() - start
                print(f"  [{latency:5.1f}s] {query[:55]:55s} ERROR: {e}")
                result = {"supporting_logs": [], "root_cause": "", "confidence": 0.0}
            latency = time.time() - start
            evaluator.evaluate_single(
                method=label,
                query=query,
                retrieved_doc_contents=result.get("supporting_logs", []),
                predicted_root_cause=result.get("root_cause", ""),
                confidence=result.get("confidence", 0.0),
                num_iterations=iters_fn(result),
                latency=latency,
                confidence_trajectory=result.get("confidence_trajectory", []),
                retrieval_scores=result.get("retrieval_scores", []),
            )
            cases.append({
                "method": label,
                "query": query,
                "ground_truth": gt_by_query[query].root_cause,
                "predicted_root_cause": result.get("root_cause", ""),
                "supporting_logs": result.get("supporting_logs", [])[:6],
                "retrieval_scores": result.get("retrieval_scores", []),
                "confidence": result.get("confidence", 0.0),
                "confidence_trajectory": result.get("confidence_trajectory", []),
                "iterations": iters_fn(result),
                "latency_s": latency,
            })
            print(f"  [{latency:5.1f}s] {query[:55]:55s} conf={result.get('confidence', 0):.3f}")
        flush_partial()
        print(f"  (wrote partial results after {label})")
        print()

    # ── Run all four methods ───────────────────────────────────────────────
    run_method("Baseline",        agent.analyze_baseline,       lambda r: 1)
    run_method("Dense+Rerank",    agent.analyze_dense_rerank,   lambda r: 1)
    run_method("FixedIterative",  lambda q: agent.analyze_fixed_iterative(q, num_iterations=3))
    run_method("Adaptive",        agent.analyze)

    # ── Aggregate metrics ──────────────────────────────────────────────────
    summary = evaluator.summary_by_method()
    per_query = evaluator.to_dataframe()

    print("=" * 64)
    print("SUMMARY BY METHOD")
    print("=" * 64)
    print(summary.to_string(index=False))
    print()

    # ── LLM-as-judge pass ──────────────────────────────────────────────────
    if not args.skip_judge:
        print("=" * 64)
        print("LLM-as-judge (binary correctness)")
        print("=" * 64)
        judgments = []
        import csv as _csv
        conf_path = out_dir / "eval_confidence.csv"
        with conf_path.open("w", newline="", encoding="utf-8") as cf:
            cw = _csv.DictWriter(cf, fieldnames=["method", "query", "judge_correct", "keyword_accuracy", "confidence"])
            cw.writeheader()
            for c in cases:
                try:
                    j = llm_judge(agent.llm, c["query"], c["ground_truth"], c["predicted_root_cause"])
                except Exception as e:
                    j = {"judgment": "Incorrect", "reason": f"judge-error: {e}", "raw": ""}
                c["judge_verdict"] = j["judgment"]
                c["judge_reason"] = j["reason"]
                row = {
                    "method": c["method"],
                    "query": c["query"],
                    "judge_correct": 1 if j["judgment"] == "Correct" else 0,
                    "keyword_accuracy": next(
                        (r.root_cause_accuracy for r in evaluator.results
                         if r.method == c["method"] and r.query == c["query"]), 0.0),
                    "confidence": c["confidence"],
                }
                judgments.append(row)
                cw.writerow(row)
                cf.flush()
                # Also re-flush cases.json so judge verdicts are preserved incrementally.
                try:
                    with (out_dir / "eval_cases.json").open("w", encoding="utf-8") as f:
                        json.dump(cases, f, indent=2)
                except Exception:
                    pass
                print(f"  [{j['judgment']:9s}] {c['method']:15s} {c['query'][:45]}")

        # Judge accuracy by method
        print()
        print("Judge correctness by method:")
        by_method = {}
        for j in judgments:
            by_method.setdefault(j["method"], []).append(j["judge_correct"])
        for m, vs in by_method.items():
            acc = sum(vs) / len(vs) if vs else 0.0
            print(f"  {m:15s} {acc:.3f}  ({sum(vs)}/{len(vs)})")

        # Confidence vs correctness correlation (pooled across all runs)
        xs = [j["confidence"] for j in judgments]
        ys = [j["judge_correct"] for j in judgments]
        print()
        print("Confidence vs judge-correctness:")
        print(f"  Pearson  r = {pearson(xs, ys):.3f}")
        print(f"  Spearman rho = {spearman(xs, ys):.3f}")

        # Save
        import csv
        with (out_dir / "eval_confidence.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["method", "query", "judge_correct", "keyword_accuracy", "confidence"])
            w.writeheader()
            w.writerows(judgments)    # ── Save everything ────────────────────────────────────────────────────
    per_query.to_csv(out_dir / "eval_per_query.csv", index=False)
    summary.to_csv(out_dir / "eval_summary.csv", index=False)
    with (out_dir / "eval_cases.json").open("w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2)

    print()
    print(f"Wrote {out_dir / 'eval_per_query.csv'}")
    print(f"Wrote {out_dir / 'eval_summary.csv'}")
    print(f"Wrote {out_dir / 'eval_cases.json'}")
    if not args.skip_judge:
        print(f"Wrote {out_dir / 'eval_confidence.csv'}")


if __name__ == "__main__":
    main()
