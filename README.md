# Adaptive Iterative RAG for Telecom Log Root Cause Analysis
A research project exploring how **iterative retrieval improves evidence quality but does not necessarily improve reasoning accuracy** in RAG systems.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LLM](https://img.shields.io/badge/LLM-Llama%203.3%2070B-orange.svg)](https://groq.com/)
[![Framework](https://img.shields.io/badge/Framework-LangChain-green.svg)](https://langchain.com/)
[![Vector DB](https://img.shields.io/badge/VectorDB-ChromaDB-purple.svg)](https://www.trychroma.com/)

## TL;DR

- **What:** Automatically finds root causes in telecom network logs using iterative retrieval + LLM reasoning
- **Why:** Single-pass RAG misses multi-file causal chains; fixed iteration wastes compute on easy queries
- **How:** `Query → Hybrid Retrieval (BM25+Dense) → Cross-Encoder Reranking → Confidence-Gated Iteration → LLM Analysis`

---

## Motivation

- **Noisy logs** — vendor-specific formats, inconsistent timestamps, error codes buried in free text
- **Implicit causal chains** — one failure spans multiple files (RRC failure → bearer cancellation → UE release) with no explicit links
- **Multi-step reasoning** — tracing the full chain needs cross-file correlation that a single retrieval pass misses

---

## System Overview

```
Query → Log Parser → Hybrid Retrieval (BM25 + Dense)
                         ↓
                  Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2)
                         ↓
                  LLM Root Cause Analysis (Llama 3.3 70B)
                         ↓
                  Confidence Scoring (4-signal weighted)
                         ↓
                  ┌─ High confidence? → Return best result
                  └─ Low confidence?  → Refine query, iterate (max 3)
```

| Component | What it does |
|-----------|-------------|
| **Hybrid Retrieval** | BM25 + dense vectors → 20–30 candidates. BM25 catches error codes; dense captures semantics |
| **Cross-Encoder Reranking** | `ms-marco-MiniLM-L-6-v2` (22M params) rescores (query, doc) pairs. Reranks only — doesn't affect recall |
| **Iterative Refinement** | LLM rewrites query using discovered error codes/timestamps → retrieves corroborating evidence |
| **Confidence Stopping** | 4-signal score (retrieval quality, completeness, evidence density, consistency) → stops early or iterates |

---

## Key Features

- Confidence-gated adaptive iteration (max 3 rounds, early stopping)
- Hybrid retrieval with cross-encoder reranking (no domain fine-tuning needed)
- LLM-based query refinement between iterations
- Structured output: root cause, severity, supporting logs with file/line/timestamp, reasoning chain
- Persistent incident memory for cross-session learning
- Telecom log parser (eGate, UEC, RAIN, syslog, archives)
- Built-in evaluation framework (Precision@K, Recall@K, root cause accuracy)
- Streamlit dashboard for interactive analysis

---

## Results

Preliminary results on a small corpus (3 log files, ~30 records, 3 queries). **Not statistically significant.**

| Method | Precision@K | Root Cause Acc. | Latency |
|--------|:-----------:|:---------------:|:-------:|
| Baseline RAG | ~0.80 | ~0.67 | ~2.5s |
| Fixed Iterative | ~0.69 | ~0.67 | ~20s |
| **Adaptive (ours)** | **0.80–0.85** | ~0.67 | ~16s |

- **Improved:** Precision@K — cross-encoder reranking surfaces more relevant docs. Most gain from reranking; iteration adds robustness.
- **Unchanged:** Root Cause Accuracy — same frozen LLM across all methods. Better retrieval ≠ better reasoning.
- **Tradeoff:** ~16s vs ~2.5s latency (cross-encoder + iteration overhead).

<details>
<summary>Example output</summary>

```json
{
  "root_cause": "UE4 released due to RRCReconfiguration failure (code 4) in rfma_impl.cpp",
  "confidence": 0.762,
  "severity": "CRITICAL",
  "supporting_logs": ["[log1.txt] ERROR ... Failure (code 4)", "[log2.txt] ERROR ... rfma_impl.cpp"],
  "iterations": [{"iteration": 1, "confidence": 0.71}, {"iteration": 2, "confidence": 0.76}],
  "converged": true
}
```

</details>

---

## Repository Structure

```
rag_system/
├── adaptive_agent.py      # Core: iterative RAG with confidence-gated stopping
├── retriever.py           # Hybrid retrieval (BM25+Dense) + cross-encoder reranking
├── evaluator.py           # Precision@K, Recall@K, root cause accuracy metrics
├── parser.py              # Telecom log parser (multi-format)
├── query_refiner.py       # LLM-based query rewriting
├── memory_store.py        # Persistent incident memory
└── config.py              # Domain constants and configuration

data/logs/                 # Sample telecom logs (eGate, UEC, RAIN)
results/                   # Plots and sample outputs
tests/                     # Unit tests (parser, evaluator)
notebooks/                 # Step-by-step pipeline walkthrough
run_evaluation.py          # End-to-end evaluation script
streamlit_app.py           # Web dashboard
```

---

## Setup

```bash
git clone https://github.com/yakkalaanugna/Automated-RAG-Agent.git
cd Automated-RAG-Agent
python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows
pip install -r requirements.txt
cp .env.example .env       # Add your Groq API key
```

### Run

```bash
# Interactive dashboard
streamlit run streamlit_app.py

# CLI analysis
python -m rag_system "Why did UE4 fail?" --logs data/logs/

# Run evaluation
python run_evaluation.py

# Run tests
pytest tests/ -v
```

---

## Limitations

- **Small dataset** — 3 queries, ~30 records. Indicative, not statistically significant.
- **Latency** — cross-encoder + iteration adds ~13s over baseline. Not optimized for real-time.
- **Fixed LLM ceiling** — root cause accuracy bounded by frozen Llama-3.3-70b. Better retrieval ≠ better reasoning.
- **No domain fine-tuning** — off-the-shelf embedding models. Telecom-adapted models could help.
- **Incomplete ablation** — reranking dominates Precision@K gains, but full ±reranker × ±iteration study not done.

---

## Future Work

- Larger telecom datasets with statistical significance testing
- Ablation study: reranker vs. iteration contributions
- Domain-adapted embeddings for telecom vocabulary
- Latency optimization (distillation, candidate pruning)
- Graph-based retrieval for explicit causal chain modeling

---

## 📄 Paper

Read the full paper here: [PDF](paper/adaptive_iterative_rag.pdf)

---

## Citation

```bibtex
@misc{yakkala2025adaptiverag,
  title   = {Adaptive Iterative RAG for Telecom Log Root Cause Analysis},
  author  = {Yakkala, Anugna},
  year    = {2025},
  url     = {https://github.com/yakkalaanugna/Automated-RAG-Agent}
}
```

---

## Author

**Anugna Yakkala** 
— Integrated M.Tech in Computer Science and Engineering (Specialization: Data Science)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Anugna%20Yakkala-blue?logo=linkedin)](https://www.linkedin.com/in/anugna-yakkala-b6383a24b/)
[![GitHub](https://img.shields.io/badge/GitHub-yakkalaanugna-black?logo=github)](https://github.com/yakkalaanugna/Automated-RAG-Agent)

---

## License

This project is for academic and research purposes.
