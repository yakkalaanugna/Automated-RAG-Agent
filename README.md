# Iterative RAG Agent — Telecom Log Analyzer

AI-powered telecom log analyzer using **RAG (Retrieval-Augmented Generation)** to automatically investigate logs, trace errors, and identify root causes.

Upload your telecom test logs → AI retrieves relevant entries via vector search → LLM analyzes and gives structured root cause diagnosis.

## Screenshots

### Single Log Analysis
![Single Analysis](screenshots/single_analysis.png)

### Pass vs Fail Comparison
![Pass vs Fail](screenshots/pass_vs_fail.png)

### Batch Analysis
![Batch Analysis](screenshots/batch_analysis.png)

## Tech Stack

| Component | Library |
|-----------|---------|
| Embeddings | HuggingFace (`all-MiniLM-L6-v2`) |
| Vector Store | ChromaDB |
| LLM | Groq (`llama-3.3-70b-versatile`) |
| Framework | LangChain |
| Web UI | Streamlit |

## How It Works

1. **Upload log files** — supports `.txt` `.log` `.json` `.csv` `.xml` `.html` `.cfg` `.tgz` `.zip`
2. **Log parsing** — cleans ANSI codes, deduplicates, filters important lines using telecom-specific keywords
3. **Severity detection** — classifies each entry as ERROR, FAIL, WARNING, or INFO
4. **Vector embedding** — HuggingFace `all-MiniLM-L6-v2` embeds log entries into 384d vectors
5. **ChromaDB storage** — vector store enables fast semantic retrieval
6. **RAG chain** — retriever fetches top-k relevant logs → prompt template → Groq LLM generates analysis
7. **Structured output** — Root Cause, Severity, Details, Recommendation

## Features

- **Vector retrieval** — ChromaDB stores embeddings, retriever finds semantically relevant entries for any query
- **Pass vs Fail comparison** — upload both logs, finds errors unique to the failure (ignoring common noise)
- **Batch analysis** — upload multiple files, all analyzed together
- **Cached embeddings** — HuggingFace model loaded once via `@st.cache_resource`, fast on reruns
- **Cached vectorstore** — same file content = same vectorstore, no re-embedding
- **Archive support** — auto-extracts `.tgz` `.zip` archives, filters relevant files inside
- **Telecom-specific** — keywords for eGate, RRC, NGAP, RACH, handover, beam failure, PDU session, etc.

## Project Structure

```
├── rag_notebook.ipynb    # Full pipeline (step-by-step Jupyter notebook)
├── streamlit_app.py      # Web UI (Streamlit dashboard)
├── requirements.txt      # Python dependencies
├── data/logs/            # Sample log files
├── screenshots/          # App screenshots for README
└── .gitignore
```

## Getting Started

**1. Clone:**
```bash
git clone https://github.com/yakkalaanugna/Automated-RAG-Agent.git
cd Automated-RAG-Agent
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run the notebook:**
Open `rag_notebook.ipynb` in VS Code or Jupyter and run all cells step by step.

**4. Or run the Streamlit app directly:**
```bash
streamlit run streamlit_app.py
```
Enter your **Groq API key** in the sidebar and upload log files.

## Groq API Key

Get a free key at https://console.groq.com/keys — no credit card needed.

Uses **Llama 3.3 70B** model via Groq's fast inference API.
