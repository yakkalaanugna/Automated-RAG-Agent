
import streamlit as st
import os, re, io, tarfile, zipfile, hashlib
from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ─── Constants ───
IMPORTANT_KEYWORDS = [
    "warn", "error", "err", "fail", "timeout", "loss",
    "latency", "delay", "congestion", "critical", "fatal",
    "refused", "rejected", "denied", "abort", "crash",
    "exception", "unreachable", "invalid", "mismatch",
    "drop", "retry", "disconnect", "panic", "oom",
    "not found", "degraded", "down", "offline",
    "rrc release", "ue context release", "crc nok",
    "cell setup", "nok", "ue release", "ngap", "rach",
    "handover", "ho failure", "rlf", "radio link failure",
    "beam failure", "pdu session", "registration reject"
]
ARCHIVE_EXT = (".tgz", ".tar.gz", ".zip")
ARCHIVE_PATTERNS = [
    "syslog", "messages", "dmesg", "kern.log", "daemon.log",
    "worker", "egate", "alarm", "error", "uec_1", "uec_2",
    "btslog", "rain", "runtime", "gnb", "enb", "cu_cp", "cu_up", "firewall"
]
SEV_BADGE = {"ERROR": "\U0001f534", "FAIL": "\U0001f534", "WARNING": "\U0001f7e1", "INFO": "\U0001f535"}

# ─── Cached embedding model — loaded ONCE across all reruns ───
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ─── Cached vectorstore — keyed by content hash, not rebuilt on rerun ───
@st.cache_resource
def get_vectorstore(_emb, content_hash, entries_tuple, collection_name):
    docs = [Document(page_content=m, metadata={"source": f, "severity": detect_severity(m)}) for f, m in entries_tuple]
    return Chroma.from_documents(docs, _emb, collection_name=f"{collection_name}_{content_hash[:8]}")

# ─── Log Processing ───
def clean_line(line):
    line = re.sub(r'\x1b\[[0-9;]*m', '', line)
    line = re.sub(r'^\d{2}:\d{2}:\d{2}\.\d+ \d{2}:\d{2}:\d{2}\.\d+ ', '', line)
    return line.strip()

def is_important(line):
    low = line.lower()
    return any(kw in low for kw in IMPORTANT_KEYWORDS)

def detect_severity(msg):
    low = msg.lower()
    if any(k in low for k in ["error", "data loss", "timeout", "critical", "fatal"]): return "ERROR"
    if "failed" in low: return "FAIL"
    if any(k in low for k in ["warning", "latency", "delay", "congestion"]): return "WARNING"
    return "INFO"

def parse_content(content, filename):
    seen, results = set(), []
    for line in content.splitlines():
        c = clean_line(line)
        if c and len(c) >= 10 and is_important(c) and c not in seen:
            seen.add(c)
            results.append((filename, c))
    return results

def parse_archive_bytes(data, name):
    results = []
    try:
        if name.lower().endswith((".tgz", ".tar.gz")):
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
                for m in tar.getmembers():
                    if not m.isfile() or m.size > 50*1024*1024: continue
                    if not any(p in m.name.lower() for p in ARCHIVE_PATTERNS): continue
                    f = tar.extractfile(m)
                    if f: results.extend(parse_content(f.read().decode("utf-8", errors="ignore"), f"{name}/{os.path.basename(m.name)}"))
        elif name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
                for n in zf.namelist():
                    if not any(p in n.lower() for p in ARCHIVE_PATTERNS): continue
                    if zf.getinfo(n).file_size > 50*1024*1024: continue
                    results.extend(parse_content(zf.read(n).decode("utf-8", errors="ignore"), f"{name}/{os.path.basename(n)}"))
    except Exception:
        pass
    return results

def process_file(uploaded):
    raw = uploaded.read()
    if uploaded.name.lower().endswith(ARCHIVE_EXT):
        return parse_archive_bytes(raw, uploaded.name), raw
    return parse_content(raw.decode("utf-8", errors="ignore"), uploaded.name), raw

def format_docs(docs):
    return "\n".join(f"[{d.metadata.get('severity','?')}] [{d.metadata.get('source','?')}] {d.page_content}" for d in docs)

def run_rag(entries, question, api_key, collection_name="analysis"):
    emb = get_embeddings()
    content_hash = hashlib.md5(str(entries).encode()).hexdigest()
    vs = get_vectorstore(emb, content_hash, tuple(entries), collection_name)
    ret = vs.as_retriever(search_kwargs={"k": min(5, len(entries))})
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile", max_tokens=800)
    prompt = ChatPromptTemplate.from_template(
        "You are a telecom log analyst. Analyze the retrieved logs.\n\n"
        "Format:\n- Root Cause: (one line)\n- Severity: CRITICAL / HIGH / MEDIUM / LOW\n"
        "- Details: (explanation with evidence)\n- Recommendation: (what to do)\n\n"
        "Logs:\n{context}\n\nQuestion: {question}"
    )
    chain = ({"context": ret | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    return chain.invoke(question)

# ─── Page Setup ───
st.set_page_config(page_title="Iterative RAG Agent", page_icon="\U0001f4e1", layout="wide")
st.markdown("# \U0001f4e1 Iterative RAG Agent — Telecom Log Analyzer")
st.markdown("Upload logs \u2192 Vector retrieval \u2192 AI root cause analysis")
st.divider()

FILE_TYPES = ["txt","log","json","csv","xml","html","htm","cfg","tgz","gz","zip"]

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Groq API Key", type="password", help="https://console.groq.com/keys")
    st.divider()
    st.markdown("**Supported formats:**")
    st.markdown("`.txt` `.log` `.json` `.csv` `.xml` `.html` `.cfg` `.tgz` `.zip`")

if not api_key:
    st.info("Enter your Groq API key in the sidebar to start.")
    st.stop()

# Pre-warm embedding model on first load
get_embeddings()

tab1, tab2, tab3 = st.tabs(["Single Analysis", "Pass vs Fail", "Batch Analysis"])

# === TAB 1: Single Analysis ===
with tab1:
    st.subheader("Upload a log file for AI analysis")
    uploaded = st.file_uploader("Log file", type=FILE_TYPES, key="single")
    query = st.text_input("Question (optional)", placeholder="e.g., Why did UE connection fail?", key="q1")

    if uploaded and st.button("Analyze", type="primary", key="b1"):
        entries, _ = process_file(uploaded)
        if not entries:
            st.warning("No important entries found in this file.")
        else:
            sev = {"ERROR":0,"FAIL":0,"WARNING":0,"INFO":0}
            for _, m in entries: sev[detect_severity(m)] += 1
            cols = st.columns(4)
            for i,(s,c) in enumerate(sev.items()): cols[i].metric(f"{SEV_BADGE.get(s,'')} {s}", c)

            with st.expander(f"Log entries ({len(entries)} total)"):
                for fn, m in entries[:50]:
                    st.text(f"{SEV_BADGE.get(detect_severity(m),'')} [{fn}] {m}")

            st.divider()
            with st.spinner("Running RAG analysis..."):
                q = query if query else "Analyze all errors and find root cause"
                result = run_rag(entries, q, api_key, "single")
            st.markdown(result)

# === TAB 2: Pass vs Fail ===
with tab2:
    st.subheader("Compare PASS and FAIL logs")
    c1, c2 = st.columns(2)
    with c1: pass_file = st.file_uploader("PASS log", type=FILE_TYPES, key="pass")
    with c2: fail_file = st.file_uploader("FAIL log", type=FILE_TYPES, key="fail")

    if pass_file and fail_file and st.button("Compare", type="primary", key="b2"):
        pass_entries, _ = process_file(pass_file)
        fail_entries, _ = process_file(fail_file)

        pass_msgs = set(m for _, m in pass_entries)
        common = [(f, m) for f, m in fail_entries if m in pass_msgs]
        fail_only = [(f, m) for f, m in fail_entries if m not in pass_msgs]

        mc = st.columns(3)
        mc[0].metric("PASS entries", len(pass_entries))
        mc[1].metric("Common (ignorable)", len(common))
        mc[2].metric("FAIL-only (critical)", len(fail_only))

        if fail_only:
            with st.expander("Critical Errors (FAIL only)", expanded=True):
                for _, l in fail_only[:20]: st.text(f"\U0001f534 {l}")
        if common:
            with st.expander("Ignorable (both logs)"):
                for _, l in common[:20]: st.text(f"\u26aa {l}")

        if fail_only:
            st.divider()
            with st.spinner("Running RAG analysis on FAIL-only errors..."):
                result = run_rag(fail_only, "Analyze these FAIL-only errors. What caused the failure?", api_key, "comparison")
            st.markdown(result)
        else:
            st.success("No unique errors in FAIL log.")

# === TAB 3: Batch Analysis ===
with tab3:
    st.subheader("Upload multiple log files")
    batch = st.file_uploader("Log files", type=FILE_TYPES, accept_multiple_files=True, key="batch")
    bq = st.text_input("Question", placeholder="What caused the failure?", key="q3")

    if batch and st.button("Analyze All", type="primary", key="b3"):
        all_entries = []
        for f in batch:
            entries, _ = process_file(f)
            st.text(f"  {f.name}: {len(entries)} entries")
            all_entries.extend(entries)

        st.metric("Total Entries", len(all_entries))
        if all_entries:
            with st.spinner("Running RAG analysis on all files..."):
                q = bq if bq else "Analyze all errors and find root cause"
                result = run_rag(all_entries, q, api_key, "batch")
            st.markdown(result)

st.divider()
st.caption("Iterative RAG Agent \u2014 LangChain + ChromaDB + HuggingFace + Groq")
