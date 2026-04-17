"""
Streamlit Web UI — Adaptive Iterative RAG Agent for Telecom Log Analysis

Uses the modular architecture from rag_system/ for:
  - Structured log parsing (rag_system/parser.py)
  - Hybrid retrieval: Vector + BM25 (rag_system/retriever.py)
  - Adaptive iterative RAG with confidence scoring (rag_system/adaptive_agent.py)
  - Incident memory store (rag_system/memory_store.py)
"""

import streamlit as st
import os
import sys
import re

# Ensure modules/ is importable
sys.path.insert(0, os.path.abspath("."))

from rag_system.parser import TelecomLogParser, LogRecord
from rag_system.adaptive_agent import AdaptiveIterativeRAGAgent

# ═══════════════════════════════════════════════════════════════════════════════
# TELECOM DEBUG KNOWLEDGE BASE
# ═══════════════════════════════════════════════════════════════════════════════

FILE_ROLES = {
    "egate_console": "eGate simulator console — UE registration, RRC, NGAP, AMF events with timestamps.",
    "egate": "eGate simulator console — UE registration, RRC, NGAP, AMF events with timestamps.",
    "uec_1": "UE Controller log — confirms UE-level RRC events (release, reconfig, attach).",
    "uec_2": "UE Controller 2 log — secondary UE controller events.",
    "syslog": "System log from btslog — kernel/system events.",
    "btslog": "Base station log collection — contains syslog, rain, runtime logs.",
    "rain": "RAN Intelligence log — CU-CP/CU-UP UE release events, PCMD records.",
    "runtime": "RLC/MAC/PHY runtime stats — DL/UL RLC retransmissions, HARQ stats.",
    "log.html": "Robot Framework test result — test step pass/fail, DL/UL data loss.",
    "e2e_console": "End-to-end test console output.",
    "cpu_utilization": "CPU utilization log.",
    "bearer_stats": "Bearer statistics — DL/UL data rate, latency, packet loss.",
    "PacketReceiver": "Packet receiver — forward jumps indicate burst packet loss.",
}

DEBUG_WORKFLOWS = {
    "ue_release": {
        "name": "UE Release / RRC Release",
        "symptoms": ["rrc release", "ue context release", "ue release", "rrc reconfiguration failure",
                      "ctrl_del_ue", "rrcrelease_chosen"],
        "workflow": [
            "1. eGate Console: Find the exact timestamp of RRC Release",
            "2. uec_1.log: Confirm RRC release at same timestamp",
            "3. rain runtime log: Check CU-CP for UeRelease trigger reason",
            "4. btslog/syslog: Check for system-level events at that timestamp",
        ],
        "next_files": ["uec_1.log", "rain runtime log", "btslog/syslog"],
    },
    "data_loss": {
        "name": "DL/UL Data Loss / Traffic Failure",
        "symptoms": ["data loss", "bytes received.*not matching", "forward jump", "packets lost"],
        "workflow": [
            "1. log.html: Get exact UE IDs and data loss amounts",
            "2. eGate Console: Check bearer_stats for affected UEs",
            "3. runtime log: Check RLC stats — Out ReTx rate",
            "4. btslog/syslog: Check for CPU overload at failure timestamps",
        ],
        "next_files": ["egate_console.log", "runtime log", "btslog/syslog"],
    },
    "registration_failure": {
        "name": "UE Registration / Attach Failure",
        "symptoms": ["registration fail", "attach fail", "registration reject", "authentication fail"],
        "workflow": [
            "1. eGate Console: Find the registration failure and UE ID",
            "2. uec_1.log: Check NAS messages around that timestamp",
            "3. rain: Check AMF/NGAP events for rejection cause",
        ],
        "next_files": ["uec_1.log", "rain runtime log"],
    },
    "rrc_reconfiguration_failure": {
        "name": "RRC Reconfiguration Failure",
        "symptoms": ["rrcreconfiguration", "reconfiguration failure", "failure.*code.*while applying"],
        "workflow": [
            "1. eGate Console: Find RRCReconfiguration failure — note UE ID, failure code",
            "2. uec_1.log: Check NrRrcMsgHandler around that timestamp",
            "3. rain: Check if CU-CP sent valid configuration",
        ],
        "next_files": ["uec_1.log", "rain runtime log"],
    },
}

SEV_BADGE = {"ERROR": "\U0001f534", "FAIL": "\U0001f534", "WARNING": "\U0001f7e1", "INFO": "\U0001f535"}
FILE_TYPES = ["txt", "log", "json", "csv", "xml", "html", "htm", "cfg", "tgz", "gz", "zip"]

# ═══════════════════════════════════════════════════════════════════════════════
# CACHED RESOURCES
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_parser():
    return TelecomLogParser(filter_important=True)

@st.cache_resource
def get_agent(api_key):
    return AdaptiveIterativeRAGAgent(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        max_iterations=5,
        convergence_threshold=0.01,
        alpha=0.7,
        top_k=12,
        use_memory=True,
        memory_path="data/memory_store.json",
    )

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def detect_file_type(filename):
    fl = filename.lower()
    for key in FILE_ROLES:
        if key in fl:
            return key
    return "unknown"

def identify_symptoms(records):
    all_text = " ".join(r.message for r in records).lower()
    matched = []
    for wf_key, wf in DEBUG_WORKFLOWS.items():
        for symptom in wf["symptoms"]:
            if re.search(symptom, all_text, re.IGNORECASE):
                matched.append(wf_key)
                break
    return matched

def process_uploaded_file(uploaded, parser):
    raw = uploaded.read()
    records = parser.parse_bytes(raw, uploaded.name)
    return records, raw

def get_timestamp_clusters(records):
    clusters = {}
    for r in records:
        nts = TelecomLogParser.normalize_timestamp(r.timestamp)
        if nts:
            key = nts[:5]
            clusters.setdefault(key, []).append(r)
    return clusters

def display_result(result):
    """Display a structured analysis result."""
    st.markdown(f"### Root Cause")
    st.markdown(f"**{result['root_cause']}**")

    mc = st.columns(4)
    mc[0].metric("Confidence", f"{result['confidence']:.4f}")
    mc[1].metric("Severity", result.get('severity', 'N/A'))
    mc[2].metric("Iterations", f"{result['total_iterations']}")
    mc[3].metric("Converged", "Yes" if result['converged'] else "No")

    if result.get('confidence_trajectory'):
        st.markdown("**Confidence trajectory:**")
        for i, conf in enumerate(result['confidence_trajectory'], 1):
            bar = "\u2588" * int(conf * 40)
            st.text(f"  Iteration {i}: {conf:.4f} {bar}")

    if result.get('reasoning_steps'):
        with st.expander("Reasoning steps", expanded=True):
            for i, step in enumerate(result['reasoning_steps'], 1):
                st.markdown(f"{i}. {step}")

    st.divider()
    st.markdown(result['full_analysis'])

# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Adaptive RAG Agent", page_icon="\U0001f4e1", layout="wide")
st.markdown("# \U0001f4e1 Adaptive Iterative RAG Agent \u2014 Telecom Log Analyzer")
st.markdown(
    "Upload logs \u2192 Structured parsing \u2192 Hybrid retrieval (Vector + BM25) \u2192 "
    "**Adaptive iterative analysis** \u2192 Explainable root cause with confidence scoring"
)
st.divider()

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Groq API Key", type="password")
    st.markdown("[Get a free Groq API key](https://console.groq.com/keys)")
    st.divider()
    st.markdown("**System architecture:**")
    st.markdown(
        "1. **Parser**: Structured log parsing (timestamp, module, error code)\n"
        "2. **Retriever**: Hybrid (70% Vector + 30% BM25)\n"
        "3. **Query Refiner**: LLM-based iterative query rewriting\n"
        "4. **Agent**: Adaptive confidence-gated iteration (max 5)\n"
        "5. **Memory**: Persistent incident knowledge base"
    )
    st.divider()
    st.markdown("**Supported:** `.txt` `.log` `.json` `.csv` `.xml` `.html` `.cfg` `.tgz` `.zip`")

if not api_key:
    st.info("Enter your Groq API key in the sidebar to start. [Get one free here](https://console.groq.com/keys)")
    st.stop()

parser = get_parser()
agent = get_agent(api_key)

tab1, tab2, tab3 = st.tabs([
    "\U0001f50d Adaptive Analysis",
    "\u2194\ufe0f Pass vs Fail",
    "\U0001f4da Deep Debug (Multi-File)",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: Adaptive Iterative Analysis
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Upload log files for adaptive iterative analysis")
    st.caption(
        "The agent automatically iterates up to 5 times, refining queries and "
        "checking confidence until convergence. Returns structured, explainable output."
    )
    uploaded_files = st.file_uploader("Log files", type=FILE_TYPES, accept_multiple_files=True, key="single")
    query = st.text_input(
        "Question (optional)",
        placeholder="e.g., Why did UE4 get released? Trace the failure chain.",
        key="q1",
    )

    if uploaded_files and st.button("Analyze", type="primary", key="b1"):
        all_records = []
        files_info = []
        for uf in uploaded_files:
            records, _ = process_uploaded_file(uf, parser)
            ftype = detect_file_type(uf.name)
            role = FILE_ROLES.get(ftype, "General log file")
            files_info.append(f"  {uf.name} ({ftype}): {len(records)} entries \u2014 {role}")
            all_records.extend(records)

        if not all_records:
            st.warning("No important entries found in the uploaded files.")
        else:
            q = query if query else "Analyze all errors, find root cause, and recommend which additional files to check"

            # File type detection
            st.markdown("**Detected file types:**")
            for info in files_info:
                st.text(info)

            # Severity breakdown
            severity = parser.severity_counts(all_records)
            cols = st.columns(4)
            for i, (s, c) in enumerate(severity.items()):
                cols[i].metric(f"{SEV_BADGE.get(s, '')} {s}", c)

            # Detected workflows
            symptoms = identify_symptoms(all_records)
            if symptoms:
                with st.expander("Detected debug workflows", expanded=True):
                    for s in symptoms:
                        wf = DEBUG_WORKFLOWS[s]
                        st.markdown(f"**{wf['name']}**")
                        for step in wf["workflow"]:
                            st.markdown(f"  {step}")
                        st.markdown("---")

            # Log preview
            with st.expander(f"Parsed log entries ({len(all_records)} total)"):
                for r in all_records[:80]:
                    st.text(f"{SEV_BADGE.get(r.log_level, '')} [{r.source}:L{r.line_number}] @{r.timestamp} {r.message}")

            st.divider()

            # Run adaptive analysis
            with st.spinner("Running adaptive iterative analysis..."):
                agent.load_records(all_records)
                result = agent.analyze(q)

            display_result(result)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Pass vs Fail
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Compare PASS and FAIL logs")
    st.caption("Upload a passing run and a failing run. The AI finds where the flow diverged.")
    c1, c2 = st.columns(2)
    with c1:
        pass_file = st.file_uploader("PASS log", type=FILE_TYPES, key="pass")
    with c2:
        fail_file = st.file_uploader("FAIL log", type=FILE_TYPES, key="fail")

    if pass_file and fail_file and st.button("Compare", type="primary", key="b2"):
        pass_records, _ = process_uploaded_file(pass_file, parser)
        fail_records, _ = process_uploaded_file(fail_file, parser)

        pass_msgs = set(r.message for r in pass_records)
        fail_only = [r for r in fail_records if r.message not in pass_msgs]
        common = [r for r in fail_records if r.message in pass_msgs]

        mc = st.columns(4)
        mc[0].metric("PASS entries", len(pass_records))
        mc[1].metric("FAIL entries", len(fail_records))
        mc[2].metric("Common", len(common))
        mc[3].metric("FAIL-only", len(fail_only))

        if fail_only:
            with st.expander("FAIL-only errors", expanded=True):
                for r in fail_only[:30]:
                    st.text(f"\U0001f534 [{r.source}:L{r.line_number}] @{r.timestamp} {r.message}")

        analysis_records = fail_only if fail_only else fail_records
        if analysis_records:
            st.divider()
            question = (
                "Compare the PASS and FAIL logs. Use TIMESTAMPS to find the exact moment "
                "the FAIL log diverged from PASS. What event triggered the failure?"
            )

            with st.spinner("Running adaptive comparison analysis..."):
                agent.load_records(analysis_records)
                result = agent.analyze(question)

            display_result(result)
        else:
            st.success("No unique errors in FAIL log \u2014 logs match.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: Deep Debug
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Deep Debug \u2014 Cross-File Timestamp Correlation")
    st.caption(
        "Upload multiple files from the same test run. "
        "The agent correlates events by timestamp across files."
    )

    deep_files = st.file_uploader(
        "Upload all available log files", type=FILE_TYPES,
        accept_multiple_files=True, key="deep",
    )
    deep_q = st.text_input(
        "What are you debugging?",
        placeholder="e.g., UEs getting released, DL data loss, RRC Reconfiguration failure",
        key="q3",
    )

    if deep_files and st.button("Deep Analyze", type="primary", key="b3"):
        all_records = []
        per_file = {}
        files_info = []
        for uf in deep_files:
            records, _ = process_uploaded_file(uf, parser)
            ftype = detect_file_type(uf.name)
            role = FILE_ROLES.get(ftype, "General log file")
            per_file[uf.name] = records
            files_info.append(f"  {uf.name} ({ftype}): {len(records)} entries \u2014 {role}")
            all_records.extend(records)

        if not all_records:
            st.warning("No important entries found.")
        else:
            q = deep_q if deep_q else "Trace the failure chain across all uploaded files using timestamp correlation"

            st.markdown("**Files detected:**")
            for info in files_info:
                st.text(info)

            # Timestamp hotspots
            clusters = get_timestamp_clusters(all_records)
            if clusters:
                hotspots = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)[:5]
                with st.expander("Timestamp hotspots", expanded=True):
                    for ts_bucket, events in hotspots:
                        if len(events) >= 2:
                            files_in_bucket = set(r.source for r in events)
                            st.markdown(
                                f"**{ts_bucket}:xx** \u2014 {len(events)} events across "
                                f"{len(files_in_bucket)} file(s)"
                            )
                            for r in events[:8]:
                                st.text(f"  [{r.source}:L{r.line_number}] @{r.timestamp} {r.message}")

            # Debug workflows
            symptoms = identify_symptoms(all_records)
            if symptoms:
                with st.expander("Detected debug workflows"):
                    for s in symptoms:
                        wf = DEBUG_WORKFLOWS[s]
                        st.markdown(f"**{wf['name']}**")
                        for step in wf["workflow"]:
                            st.markdown(f"  {step}")

            st.divider()

            with st.spinner("Running adaptive deep analysis..."):
                agent.load_records(all_records)
                result = agent.analyze(q)

            display_result(result)

st.divider()
st.caption(
    "Adaptive Iterative RAG Agent \u2014 Hybrid Retrieval (Vector + BM25) \u2014 "
    "Confidence-Gated Iteration \u2014 LangChain + ChromaDB + HuggingFace + Groq"
)
import streamlit as st
import os, re, io, tarfile, zipfile, hashlib
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ═══════════════════════════════════════════════════════════════════════════════
# TELECOM DEBUG KNOWLEDGE BASE
# ═══════════════════════════════════════════════════════════════════════════════

FILE_ROLES = {
    "egate_console": "eGate simulator console — UE registration, RRC, NGAP, AMF events with timestamps. START HERE.",
    "egate": "eGate simulator console — UE registration, RRC, NGAP, AMF events with timestamps. START HERE.",
    "uec_1": "UE Controller log — confirms UE-level RRC events (release, reconfig, attach). Cross-check egate timestamps.",
    "uec_2": "UE Controller 2 log — secondary UE controller events.",
    "syslog": "System log from btslog — kernel/system events. Check for crashes, OOM, CPU spikes at failure timestamps.",
    "btslog": "Base station log collection — contains syslog, rain, runtime logs for gNB-side debugging.",
    "rain": "RAN Intelligence log — CU-CP/CU-UP UE release events, PCMD records, F1AP/E1AP/NGAP from gNB side.",
    "runtime": "RLC/MAC/PHY runtime stats — DL/UL RLC retransmissions (Out ReTx), HARQ stats, scheduling.",
    "log.html": "Robot Framework test result — test step pass/fail, DL/UL data loss measurements.",
    "e2e_console": "End-to-end test console output — test orchestration events.",
    "cpu_utilization": "CPU utilization log — check for CPU overload at failure timestamps.",
    "backup_output": "Configuration backup XML — gNB configuration state.",
    "sosreport": "SOS Report archive (tgz) — contains wts/log/uec_1.log and other diagnostics.",
    "bearer_stats": "Bearer statistics — DL/UL data rate, latency, packet loss, forward jumps.",
    "PacketReceiver": "Packet receiver — long forward jumps indicate burst packet loss.",
}

DEBUG_WORKFLOWS = {
    "ue_release": {
        "name": "UE Release / RRC Release",
        "symptoms": ["rrc release", "ue context release", "ue release", "rrc reconfiguration failure",
                      "ctrl_del_ue", "rrcrelease_chosen"],
        "workflow": [
            "1. eGate Console: Find the exact timestamp of RRC Release or UE Context Release",
            "2. uec_1.log (from sosreport -> wts -> log -> uec_1.log): Confirm RRC release at same timestamp",
            "3. rain runtime log: Check CU-CP for UeRelease trigger reason at that timestamp",
            "4. btslog -> syslog: Check for system-level events (crashes, OOM) at that timestamp",
            "5. If multiple UEs released simultaneously -> check for cell/beam failure or gNB restart",
        ],
        "next_files": ["uec_1.log", "rain runtime log", "btslog/syslog"],
    },
    "data_loss": {
        "name": "DL/UL Data Loss / Traffic Failure",
        "symptoms": ["data loss", "bytes received.*not matching", "bytes sent.*not matching",
                      "traffic.*nok", "forward jump", "packets lost"],
        "workflow": [
            "1. log.html / e2e_console: Get exact UE IDs and data loss amounts",
            "2. eGate Console: Run 'bearer_stats' for affected UEs — check packet loss, forward jumps, latency",
            "3. eGate Console: Look for 'long forward jump of pn' at PacketReceiver.cpp — burst packet loss",
            "4. runtime log: Check RLC stats — 'Out ReTx' rate indicates radio retransmissions",
            "5. btslog/syslog: Check for CPU overload, scheduling delays at forward jump timestamps",
            "6. rain: Check for beam failure, handover, or UE context release around same time",
        ],
        "next_files": ["egate_console.log (bearer_stats)", "runtime log (RLC stats)", "btslog/syslog"],
    },
    "registration_failure": {
        "name": "UE Registration / Attach Failure",
        "symptoms": ["registration fail", "attach fail", "registration reject", "ngap reject",
                      "authentication fail"],
        "workflow": [
            "1. eGate Console: Find the registration failure message and UE ID",
            "2. uec_1.log: Check NAS messages around that timestamp",
            "3. rain: Check AMF/NGAP events for the rejection cause",
            "4. syslog: Check if AMF or core network services were overloaded",
        ],
        "next_files": ["uec_1.log", "rain runtime log", "syslog"],
    },
    "handover_failure": {
        "name": "Handover Failure",
        "symptoms": ["handover fail", "ho failure", "x2ap.*timeout", "target cell.*not responding"],
        "workflow": [
            "1. eGate Console: Find handover failure event and source/target cell IDs",
            "2. rain: Check X2AP/Xn handover preparation and execution events",
            "3. btslog/syslog on target cell: Check if target gNB was operational",
            "4. runtime: Check for radio condition degradation before handover trigger",
        ],
        "next_files": ["rain runtime log", "target gNB syslog", "runtime log"],
    },
    "rrc_reconfiguration_failure": {
        "name": "RRC Reconfiguration Failure",
        "symptoms": ["rrcreconfiguration", "reconfiguration failure", "failure.*code.*while applying"],
        "workflow": [
            "1. eGate Console: Find RRCReconfiguration failure — note UE ID, failure code, timestamp",
            "2. uec_1.log: Check NrRrcMsgHandler around that timestamp for DL-DCCH-NR message details",
            "3. rain: Check if CU-CP sent valid configuration (bearerSetup, measConfig, etc.)",
            "4. runtime: Check PHY/MAC layer for radio condition at that time",
        ],
        "next_files": ["uec_1.log", "rain runtime log", "runtime log"],
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# PRE-COMPILED REGEX PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')

# Timestamp formats for real telecom logs
_TS_EGATE = re.compile(r'^(\d{2}:\d{2}:\d{2}\.\d+)\s+\d{2}:\d{2}:\d{2}\.\d+\s+')
_TS_UEC = re.compile(r"^[de]\s+(\d{2}:\d{2}:\d{2}'\d{3}\"\d{3})")
_TS_ERROR_UEC = re.compile(r"^ERROR!!\s+(\d{2}:\d{2}:\d{2}'\d{3}\"\d{3})")
_TS_RAIN = re.compile(r'<(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)Z?>')
_TS_ROBOT = re.compile(r'^(\d{8}\s+\d{2}:\d{2}:\d{2}\.\d+)')
_TS_GENERIC = re.compile(r'^(\d{2}:\d{2}:\d{2}[\.\:]\d+)')
_TS_ISO = re.compile(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})')

_IMPORTANT_RE = re.compile(
    r'warn|error|err\b|fail|timeout|loss|latency|delay|congestion|critical|fatal|'
    r'refused|rejected|denied|abort|crash|exception|unreachable|invalid|mismatch|'
    r'drop|retry|disconnect|panic|oom|not found|degraded|down|offline|'
    r'rrc release|rrc reconfiguration|ue context release|crc nok|cell setup|nok|ue release|ngap|rach|'
    r'handover|ho failure|rlf|radio link failure|beam failure|pdu session|registration reject|'
    r's1ap|x2ap|f1ap|e1ap|sctp.*fail|gtp.*error|overload|overflow|underflow|'
    r'segfault|core dump|stack trace|authentication fail|integrity fail|cipher fail|'
    r'drb release|srb fail|rlc retx|rlc retransmission|harq nack|pucch.*fail|prach.*fail|'
    r'forward jump|packets lost|data loss|bytes received.*not matching|bytes sent.*not matching|'
    r'long forward jump|out retx|bearer_stats|ctrl_del_ue|'
    r'rrcrelease|uecontextrelease|registrationreject|'
    r'nr_rrc::c1_rrcRelease|rfma_impl|pcmd record|trigger ue release',
    re.IGNORECASE,
)
_SEV_ERROR_RE = re.compile(
    r'error|data loss|timeout|critical|fatal|segfault|core dump|unreachable|abort|panic|'
    r'forward jump|packets lost|rrc reconfiguration.*failure|ctrl_del_ue',
    re.IGNORECASE,
)
_SEV_FAIL_RE = re.compile(r'fail|nok|reject', re.IGNORECASE)
_SEV_WARN_RE = re.compile(r'warn|latency|delay|congestion|degraded|retry|overload|retx|retransmission', re.IGNORECASE)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

ARCHIVE_EXT = (".tgz", ".tar.gz", ".zip")
ARCHIVE_PATTERNS = [
    "syslog", "messages", "dmesg", "kern.log", "daemon.log",
    "worker", "egate", "alarm", "error", "uec_1", "uec_2",
    "btslog", "rain", "runtime", "gnb", "enb", "cu_cp", "cu_up", "firewall",
    "log.html", "e2e_console", "cpu_utilization", "PacketReceiver",
]
SEV_BADGE = {"ERROR": "\U0001f534", "FAIL": "\U0001f534", "WARNING": "\U0001f7e1", "INFO": "\U0001f535"}
FILE_TYPES = ["txt", "log", "json", "csv", "xml", "html", "htm", "cfg", "tgz", "gz", "zip"]

# ═══════════════════════════════════════════════════════════════════════════════
# CACHED RESOURCES
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )

@st.cache_resource
def get_llm(_api_key):
    return ChatGroq(groq_api_key=_api_key, model_name="llama-3.3-70b-versatile", max_tokens=2048, temperature=0)

@st.cache_resource
def get_vectorstore(_emb, content_hash, entries_tuple, collection_name):
    docs = [
        Document(
            page_content=m,
            metadata={"source": f, "line": ln, "timestamp": ts, "severity": detect_severity(m)},
        )
        for f, ln, ts, m in entries_tuple
    ]
    return Chroma.from_documents(docs, _emb, collection_name=f"{collection_name}_{content_hash[:8]}")

# ═══════════════════════════════════════════════════════════════════════════════
# LOG PARSING — real telecom log formats
# ═══════════════════════════════════════════════════════════════════════════════

def clean_line(line):
    return _ANSI_RE.sub('', line).strip()

def extract_timestamp(line):
    for pattern in [_TS_EGATE, _TS_UEC, _TS_ERROR_UEC, _TS_ROBOT, _TS_GENERIC]:
        m = pattern.search(line)
        if m:
            return m.group(1)
    m = _TS_RAIN.search(line)
    if m:
        return m.group(1)
    m = _TS_ISO.search(line)
    if m:
        return m.group(1)
    return ""

def normalize_timestamp(ts):
    if not ts:
        return ""
    m = re.match(r'(\d{2}:\d{2}:\d{2})', ts)
    if m:
        return m.group(1)
    m = re.search(r'T(\d{2}:\d{2}:\d{2})', ts)
    if m:
        return m.group(1)
    return ts[:8] if len(ts) >= 8 else ts

def detect_file_type(filename):
    fl = filename.lower()
    for key in FILE_ROLES:
        if key in fl:
            return key
    return "unknown"

def is_important(line):
    return bool(_IMPORTANT_RE.search(line))

def detect_severity(msg):
    if _SEV_ERROR_RE.search(msg):
        return "ERROR"
    if _SEV_FAIL_RE.search(msg):
        return "FAIL"
    if _SEV_WARN_RE.search(msg):
        return "WARNING"
    return "INFO"

def identify_symptoms(entries):
    all_text = " ".join(m for _, _, _, m in entries).lower()
    matched = []
    for wf_key, wf in DEBUG_WORKFLOWS.items():
        for symptom in wf["symptoms"]:
            if re.search(symptom, all_text, re.IGNORECASE):
                matched.append(wf_key)
                break
    return matched

def get_timestamp_window(entries):
    clusters = {}
    for f, ln, ts, m in entries:
        nts = normalize_timestamp(ts)
        if nts:
            key = nts[:5]  # HH:MM bucket
            clusters.setdefault(key, []).append((f, ln, ts, m))
    return clusters

def parse_content(content, filename):
    seen, results = set(), []
    for line_no, line in enumerate(content.splitlines(), start=1):
        c = clean_line(line)
        if c and len(c) >= 10 and is_important(c) and c not in seen:
            seen.add(c)
            ts = extract_timestamp(c)
            results.append((filename, line_no, ts, c))
    return results

def parse_all_lines(content, filename):
    results = []
    for line_no, line in enumerate(content.splitlines(), start=1):
        c = clean_line(line)
        if c and len(c) >= 5:
            ts = extract_timestamp(c)
            results.append((filename, line_no, ts, c))
    return results

def parse_archive_bytes(data, name):
    results = []
    try:
        if name.lower().endswith((".tgz", ".tar.gz")):
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
                for m in tar.getmembers():
                    if not m.isfile() or m.size > 50 * 1024 * 1024:
                        continue
                    if not any(p in m.name.lower() for p in ARCHIVE_PATTERNS):
                        continue
                    f = tar.extractfile(m)
                    if f:
                        results.extend(
                            parse_content(f.read().decode("utf-8", errors="ignore"),
                                          f"{name}/{os.path.basename(m.name)}")
                        )
        elif name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
                for n in zf.namelist():
                    if not any(p in n.lower() for p in ARCHIVE_PATTERNS):
                        continue
                    if zf.getinfo(n).file_size > 50 * 1024 * 1024:
                        continue
                    results.extend(
                        parse_content(zf.read(n).decode("utf-8", errors="ignore"),
                                      f"{name}/{os.path.basename(n)}")
                    )
    except Exception:
        pass
    return results

def process_file(uploaded):
    raw = uploaded.read()
    if uploaded.name.lower().endswith(ARCHIVE_EXT):
        return parse_archive_bytes(raw, uploaded.name), raw
    return parse_content(raw.decode("utf-8", errors="ignore"), uploaded.name), raw

def process_file_all_lines(raw_bytes, filename):
    return parse_all_lines(raw_bytes.decode("utf-8", errors="ignore"), filename)

# ═══════════════════════════════════════════════════════════════════════════════
# RAG CHAIN
# ═══════════════════════════════════════════════════════════════════════════════

def format_docs(docs):
    return "\n".join(
        f"[{d.metadata.get('severity','?')}] [{d.metadata.get('source','?')}:L{d.metadata.get('line','?')}]"
        f" @{d.metadata.get('timestamp','')} {d.page_content}"
        for d in docs
    )

def _build_chain(entries, api_key, collection_name, prompt_text):
    emb = get_embeddings()
    content_hash = hashlib.md5(str(entries).encode()).hexdigest()
    vs = get_vectorstore(emb, content_hash, tuple(entries), collection_name)
    ret = vs.as_retriever(search_kwargs={"k": min(12, len(entries))})
    llm = get_llm(api_key)
    prompt = ChatPromptTemplate.from_template(prompt_text)
    return ({"context": ret | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())

def build_workflow_context(entries):
    symptoms = identify_symptoms(entries)
    file_types_seen = set()
    for f, _, _, _ in entries:
        ft = detect_file_type(f)
        if ft != "unknown":
            file_types_seen.add(ft)

    ctx_parts = []
    if file_types_seen:
        ctx_parts.append("FILES UPLOADED AND THEIR ROLES:")
        for ft in file_types_seen:
            ctx_parts.append(f"  - {ft}: {FILE_ROLES.get(ft, 'Unknown role')}")

    if symptoms:
        ctx_parts.append("\nDETECTED FAILURE PATTERNS AND DEBUG WORKFLOWS:")
        for s in symptoms:
            wf = DEBUG_WORKFLOWS[s]
            ctx_parts.append(f"\n  [{wf['name']}]")
            for step in wf["workflow"]:
                ctx_parts.append(f"    {step}")

    needed_files = set()
    for s in symptoms:
        for nf in DEBUG_WORKFLOWS[s].get("next_files", []):
            needed_files.add(nf)
    present_kw = " ".join(file_types_seen).lower()
    truly_missing = [nf for nf in needed_files if nf.split(" ")[0].split("/")[0].split(".")[0].lower() not in present_kw]
    if truly_missing:
        ctx_parts.append("\nFILES NOT YET UPLOADED (recommend uploading for deeper analysis):")
        for nf in truly_missing:
            ctx_parts.append(f"  - {nf}")

    clusters = get_timestamp_window(entries)
    if clusters:
        busiest = max(clusters.items(), key=lambda x: len(x[1]))
        if len(busiest[1]) > 2:
            ctx_parts.append(f"\nTIMESTAMP HOTSPOT: {len(busiest[1])} events clustered around {busiest[0]}:xx")
            ctx_parts.append("  Cross-correlate this timestamp across all uploaded files.")

    return "\n".join(ctx_parts)

TELECOM_SYSTEM_PROMPT = (
    "You are an expert telecom log analyst specializing in 5G gNB (RAN), UE testing, "
    "and end-to-end call flow debugging. You understand:\n"
    "- eGate console logs (UE simulator with RRC/NAS/NGAP messages)\n"
    "- uec_1.log (UE controller with NrRrcMsgHandler, RRC message handling)\n"
    "- rain runtime logs (CU-CP/CU-UP events, UE release triggers, PCMD records)\n"
    "- btslog/syslog (system-level events on gNB)\n"
    "- Runtime RLC stats (Out ReTx = retransmission bytes, lostPktsF1, scheduling)\n"
    "- Robot Framework log.html (test pass/fail, DL/UL data loss measurements)\n"
    "- Bearer stats (packet rates, latency, forward jumps = burst packet loss)\n\n"
    "CRITICAL RULES:\n"
    "1. ALWAYS use TIMESTAMPS to correlate events across different log files.\n"
    "2. When you see an error, explain the CHAIN OF EVENTS that led to it.\n"
    "3. If 'long forward jump' is seen, it means burst packet loss at that instant.\n"
    "4. If RLC 'Out ReTx' is non-zero, relate it to radio conditions.\n"
    "5. If logs are insufficient, ALWAYS tell the user EXACTLY which files to upload next and WHERE to find them "
    "(e.g., 'Upload uec_1.log from sosreport -> wts -> log -> uec_1.log').\n"
    "6. Reference EXACT file names, line numbers, and timestamps as evidence.\n"
    "7. For data loss: calculate the loss percentage (bytes_sent - bytes_received) / bytes_sent.\n"
    "8. For UE release: trace WHY the gNB initiated release — check the message BEFORE the release event.\n\n"
    "Each log entry shows [SEVERITY] [file:Line] @timestamp message.\n\n"
)

def run_rag(entries, question, api_key, collection_name="analysis", extra_context=""):
    workflow_ctx = build_workflow_context(entries)
    prompt_text = TELECOM_SYSTEM_PROMPT
    if workflow_ctx:
        prompt_text += workflow_ctx + "\n\n"
    if extra_context:
        prompt_text += extra_context + "\n\n"
    prompt_text += (
        "FORMAT YOUR RESPONSE AS:\n"
        "## Root Cause\n(one-line summary)\n\n"
        "## Severity\nCRITICAL / HIGH / MEDIUM / LOW\n\n"
        "## Error Timeline\n(list events in chronological order with file:line @timestamp)\n\n"
        "## Cross-File Correlation\n(how events in different files relate by timestamp)\n\n"
        "## Details\n(deep explanation of the failure chain, referencing specific lines and timestamps)\n\n"
        "## Next Steps / Missing Logs\n"
        "(if the uploaded logs are not enough to fully diagnose, recommend EXACTLY which files to upload next and WHERE to find them — "
        "e.g., 'Upload uec_1.log from sosreport -> wts -> log -> uec_1.log to confirm the RRC Release at 13:54:15')\n\n"
        "## Recommendation\n(what to do to fix/investigate further)\n\n"
        "Logs:\n{{context}}\n\nQuestion: {{question}}"
    )
    chain = _build_chain(entries, api_key, collection_name, prompt_text)
    return chain.stream(question)

def run_rag_iterative(entries, question, api_key, collection_name, prev_answer="", feedback="", extra_context=""):
    workflow_ctx = build_workflow_context(entries)
    prompt_text = (
        TELECOM_SYSTEM_PROMPT
        + "The user was NOT satisfied with the previous analysis.\n"
        "Review the previous answer and user feedback, then provide a DEEPER analysis.\n"
        "Look at different log entries, dig into timestamps, consider alternative root causes.\n"
        "If the user asks to check a specific file or event, focus on that.\n\n"
        "Previous answer:\n" + prev_answer + "\n\n"
        "User feedback: " + feedback + "\n\n"
    )
    if workflow_ctx:
        prompt_text += workflow_ctx + "\n\n"
    if extra_context:
        prompt_text += extra_context + "\n\n"
    prompt_text += (
        "FORMAT YOUR RESPONSE AS:\n"
        "## Root Cause\n(one-line summary)\n\n"
        "## Severity\nCRITICAL / HIGH / MEDIUM / LOW\n\n"
        "## Error Timeline\n(events in chronological order with file:line @timestamp)\n\n"
        "## Cross-File Correlation\n(timestamp-based correlation across files)\n\n"
        "## Details\n(deep explanation referencing specific lines and timestamps)\n\n"
        "## Next Steps / Missing Logs\n(which files to upload and where to find them)\n\n"
        "## Recommendation\n(what to do)\n\n"
        "Logs:\n{{context}}\n\nQuestion: {{question}}"
    )
    chain = _build_chain(entries, api_key, collection_name, prompt_text)
    return chain.stream(question)

# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Iterative RAG Agent", page_icon="\U0001f4e1", layout="wide")
st.markdown("# \U0001f4e1 Iterative RAG Agent \u2014 Telecom Log Analyzer")
st.markdown(
    "Upload logs \u2192 Timestamp extraction \u2192 Cross-file correlation \u2192 "
    "AI root cause analysis \u2192 **Debug workflow guidance** \u2192 **Iterative refinement**"
)
st.divider()

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Groq API Key", type="password")
    st.markdown("[Get a free Groq API key](https://console.groq.com/keys)")
    st.divider()
    st.markdown("**Telecom debug flow:**")
    st.markdown(
        "1. Upload **egate_console.log** first\n"
        "2. Get failure timestamps & symptoms\n"
        "3. Upload **uec_1.log** to confirm UE events\n"
        "4. Upload **btslog/syslog** for system events\n"
        "5. Upload **rain** for gNB-side RAN events\n"
        "6. Upload **runtime** for RLC/MAC stats"
    )
    st.divider()
    st.markdown("**Supported:** `.txt` `.log` `.json` `.csv` `.xml` `.html` `.cfg` `.tgz` `.zip`")
    st.divider()
    st.markdown("**Typical log collection:**")
    st.caption(
        "btslog/, custom_configs/, gnb_pm_counters/, egate_console.log, "
        "log.html, sosReport.tgz, cpu_utilization.log, runtime, rain"
    )

if not api_key:
    st.info("Enter your Groq API key in the sidebar to start. [Get one free here](https://console.groq.com/keys)")
    st.stop()

get_embeddings()

# Session state
for key in [
    "single_result", "single_entries", "single_query", "single_iteration",
    "single_files_info",
    "compare_result", "compare_entries", "compare_query", "compare_iteration", "compare_extra",
    "batch_result", "batch_entries", "batch_query", "batch_iteration", "batch_files_info",
]:
    if key not in st.session_state:
        st.session_state[key] = None if "iteration" not in key else 0

tab1, tab2, tab3 = st.tabs(
    ["\U0001f50d Single / Multi-File Analysis", "\u2194\ufe0f Pass vs Fail", "\U0001f4da Deep Debug (Correlated)"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: Single / Multi-File Analysis
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Upload one or more log files for AI analysis")
    st.caption("The AI detects file types, extracts timestamps, identifies failure patterns, and suggests next files to check.")
    uploaded_files = st.file_uploader("Log files", type=FILE_TYPES, accept_multiple_files=True, key="single")
    query = st.text_input("Question (optional)", placeholder="e.g., Why did UE4 get released? Check timestamp 18:34:08", key="q1")

    if uploaded_files and st.button("Analyze", type="primary", key="b1"):
        all_entries = []
        files_info = []
        for uf in uploaded_files:
            entries, _ = process_file(uf)
            ftype = detect_file_type(uf.name)
            role = FILE_ROLES.get(ftype, "General log file")
            files_info.append(f"  {uf.name} ({ftype}): {len(entries)} important entries \u2014 {role}")
            all_entries.extend(entries)

        if not all_entries:
            st.warning("No important entries found in the uploaded files.")
        else:
            st.session_state.single_entries = all_entries
            st.session_state.single_files_info = "\n".join(files_info)
            q = query if query else "Analyze all errors, find root cause, and recommend which additional files to check"
            st.session_state.single_query = q
            st.session_state.single_iteration = 1

            st.markdown("**Detected file types:**")
            for info in files_info:
                st.text(info)

            sev = {"ERROR": 0, "FAIL": 0, "WARNING": 0, "INFO": 0}
            for _, _, _, m in all_entries:
                sev[detect_severity(m)] += 1
            cols = st.columns(4)
            for i, (s, c) in enumerate(sev.items()):
                cols[i].metric(f"{SEV_BADGE.get(s, '')} {s}", c)

            symptoms = identify_symptoms(all_entries)
            if symptoms:
                with st.expander("Detected debug workflows", expanded=True):
                    for s in symptoms:
                        wf = DEBUG_WORKFLOWS[s]
                        st.markdown(f"**{wf['name']}**")
                        for step in wf["workflow"]:
                            st.markdown(f"  {step}")
                        if wf.get("next_files"):
                            st.markdown(f"  **Next files to upload:** {', '.join(wf['next_files'])}")
                        st.markdown("---")

            with st.expander(f"Log entries ({len(all_entries)} total) \u2014 with timestamps"):
                for fn, ln, ts, m in all_entries[:80]:
                    ts_display = f"@{ts}" if ts else ""
                    st.text(f"{SEV_BADGE.get(detect_severity(m), '')} [{fn}:L{ln}] {ts_display} {m}")

            st.divider()
            with st.spinner("Retrieving relevant logs..."):
                stream = run_rag(all_entries, q, api_key, "single")
            result = st.write_stream(stream)
            st.session_state.single_result = result

    if st.session_state.single_result and st.session_state.single_iteration > 0:
        st.divider()
        st.markdown(f"**Iteration {st.session_state.single_iteration}** \u2014 Are you satisfied?")
        fc1, fc2 = st.columns(2)
        with fc1:
            if st.button("\u2705 Yes, I'm satisfied", key=f"s1_yes_{st.session_state.single_iteration}"):
                st.success("Analysis complete.")
                st.session_state.single_result = None
                st.session_state.single_iteration = 0
        with fc2:
            feedback = st.text_input(
                "What's missing? (e.g., 'check uec_1 timestamps', 'why did gNB initiate release?')",
                key=f"s1_fb_{st.session_state.single_iteration}",
            )
            if st.button("\U0001f504 Re-analyze", key=f"s1_no_{st.session_state.single_iteration}"):
                fb = feedback if feedback else "Dig deeper \u2014 check timestamps, correlate across files, suggest specific debug steps"
                with st.spinner("Retrieving relevant logs..."):
                    stream = run_rag_iterative(
                        st.session_state.single_entries,
                        st.session_state.single_query,
                        api_key, "single",
                        st.session_state.single_result, fb,
                    )
                result = st.write_stream(stream)
                st.session_state.single_result = result
                st.session_state.single_iteration += 1

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Pass vs Fail
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Compare PASS and FAIL logs \u2014 Flow-Aware Analysis")
    st.caption("Upload a passing run and a failing run. The AI correlates timestamps to find where the flow diverged.")
    c1, c2 = st.columns(2)
    with c1:
        pass_file = st.file_uploader("PASS log", type=FILE_TYPES, key="pass")
    with c2:
        fail_file = st.file_uploader("FAIL log", type=FILE_TYPES, key="fail")

    if pass_file and fail_file and st.button("Compare", type="primary", key="b2"):
        pass_entries, pass_raw = process_file(pass_file)
        fail_entries, fail_raw = process_file(fail_file)
        pass_all = parse_all_lines(pass_raw.decode("utf-8", errors="ignore"), pass_file.name)
        fail_all = parse_all_lines(fail_raw.decode("utf-8", errors="ignore"), fail_file.name)

        pass_msgs = set(m for _, _, _, m in pass_entries)
        fail_only = [(f, ln, ts, m) for f, ln, ts, m in fail_entries if m not in pass_msgs]
        common = [(f, ln, ts, m) for f, ln, ts, m in fail_entries if m in pass_msgs]

        mc = st.columns(4)
        mc[0].metric("PASS entries", len(pass_entries))
        mc[1].metric("FAIL entries", len(fail_entries))
        mc[2].metric("Common", len(common))
        mc[3].metric("FAIL-only", len(fail_only))

        with st.expander("PASS log flow"):
            for fn, ln, ts, m in pass_all[:60]:
                st.text(f"L{ln:>4} @{ts:>15}  {m}" if ts else f"L{ln:>4}                  {m}")

        with st.expander("FAIL log flow (divergences marked \U0001f534)", expanded=True):
            for fn, ln, ts, m in fail_all[:60]:
                is_fail_only = m not in pass_msgs and is_important(m)
                marker = "\U0001f534" if is_fail_only else "  "
                st.text(f"{marker} L{ln:>4} @{ts:>15}  {m}" if ts else f"{marker} L{ln:>4}                  {m}")

        if fail_only:
            with st.expander("FAIL-only errors", expanded=True):
                for fn, ln, ts, m in fail_only[:30]:
                    st.text(f"\U0001f534 [{fn}:L{ln}] @{ts} {m}")

        pass_flow_summary = "PASS LOG FLOW:\n" + "\n".join(
            f"  L{ln} @{ts}: {m}" for _, ln, ts, m in pass_all[:40]
        )
        pass_flow_summary += "\n\nFAIL LOG FLOW:\n" + "\n".join(
            f"  L{ln} @{ts}: {m}" for _, ln, ts, m in fail_all[:40]
        )

        analysis_entries = fail_only if fail_only else fail_entries
        if analysis_entries:
            st.divider()
            question = (
                "Compare the PASS and FAIL log flows. Use TIMESTAMPS to find the exact moment "
                "the FAIL log diverged from PASS. What event triggered the failure? "
                "Reference specific line numbers and timestamps from both logs."
            )
            st.session_state.compare_query = question
            st.session_state.compare_entries = analysis_entries
            st.session_state.compare_extra = pass_flow_summary
            st.session_state.compare_iteration = 1

            with st.spinner("Retrieving relevant logs..."):
                stream = run_rag(analysis_entries, question, api_key, "comparison", extra_context=pass_flow_summary)
            result = st.write_stream(stream)
            st.session_state.compare_result = result
        else:
            st.success("No unique errors in FAIL log \u2014 logs match.")

    if st.session_state.compare_result and st.session_state.compare_iteration > 0:
        st.divider()
        st.markdown(f"**Iteration {st.session_state.compare_iteration}** \u2014 Satisfied?")
        fc1, fc2 = st.columns(2)
        with fc1:
            if st.button("\u2705 Yes", key=f"s2_yes_{st.session_state.compare_iteration}"):
                st.success("Comparison complete.")
                st.session_state.compare_result = None
                st.session_state.compare_iteration = 0
        with fc2:
            feedback = st.text_input("Feedback", key=f"s2_fb_{st.session_state.compare_iteration}")
            if st.button("\U0001f504 Re-analyze", key=f"s2_no_{st.session_state.compare_iteration}"):
                fb = feedback if feedback else "Compare timestamps more carefully across the flows"
                with st.spinner("Retrieving relevant logs..."):
                    stream = run_rag_iterative(
                        st.session_state.compare_entries,
                        st.session_state.compare_query,
                        api_key, "comparison",
                        st.session_state.compare_result, fb,
                        extra_context=st.session_state.compare_extra or "",
                    )
                result = st.write_stream(stream)
                st.session_state.compare_result = result
                st.session_state.compare_iteration += 1

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: Deep Debug — Cross-File Timestamp Correlation
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Deep Debug \u2014 Cross-File Timestamp Correlation")
    st.caption(
        "Upload multiple files from the same test run (egate, uec_1, syslog, rain, runtime). "
        "The AI correlates events by timestamp across files to trace the full failure chain."
    )
    st.markdown(
        "**Debug flow:** egate_console.log (get failure time) \u2192 uec_1.log (confirm UE events) \u2192 "
        "syslog (system events) \u2192 rain (gNB CU events) \u2192 runtime (RLC/MAC stats)"
    )

    deep_files = st.file_uploader("Upload all available log files from the test run", type=FILE_TYPES,
                                   accept_multiple_files=True, key="deep")
    deep_q = st.text_input(
        "What are you debugging?",
        placeholder="e.g., UEs getting released, DL data loss, RRC Reconfiguration failure",
        key="q3",
    )

    if deep_files and st.button("Deep Analyze", type="primary", key="b3"):
        all_entries = []
        per_file = {}
        files_info = []
        for uf in deep_files:
            entries, _ = process_file(uf)
            ftype = detect_file_type(uf.name)
            role = FILE_ROLES.get(ftype, "General log file")
            per_file[uf.name] = entries
            files_info.append(f"  {uf.name} ({ftype}): {len(entries)} entries \u2014 {role}")
            all_entries.extend(entries)

        if not all_entries:
            st.warning("No important entries found.")
        else:
            st.session_state.batch_entries = all_entries
            st.session_state.batch_files_info = "\n".join(files_info)
            q = deep_q if deep_q else "Trace the failure chain across all uploaded files using timestamp correlation"
            st.session_state.batch_query = q
            st.session_state.batch_iteration = 1

            st.markdown("**Files detected:**")
            for info in files_info:
                st.text(info)

            clusters = get_timestamp_window(all_entries)
            if clusters:
                hotspots = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)[:5]
                with st.expander("Timestamp hotspots (events at same time across files)", expanded=True):
                    for ts_bucket, events in hotspots:
                        if len(events) >= 2:
                            files_in_bucket = set(f for f, _, _, _ in events)
                            st.markdown(
                                f"**{ts_bucket}:xx** \u2014 {len(events)} events across "
                                f"{len(files_in_bucket)} file(s): {', '.join(files_in_bucket)}"
                            )
                            for f, ln, ts, m in events[:8]:
                                st.text(f"  [{f}:L{ln}] @{ts} {m}")

            with st.expander("Per-file entry breakdown"):
                for fname, entries_list in per_file.items():
                    st.markdown(f"**{fname}** ({len(entries_list)} entries)")
                    for fn, ln, ts, m in entries_list[:20]:
                        st.text(f"  {SEV_BADGE.get(detect_severity(m), '')} L{ln} @{ts} {m}")

            symptoms = identify_symptoms(all_entries)
            if symptoms:
                with st.expander("Detected debug workflows"):
                    for s in symptoms:
                        wf = DEBUG_WORKFLOWS[s]
                        st.markdown(f"**{wf['name']}**")
                        for step in wf["workflow"]:
                            st.markdown(f"  {step}")

            st.divider()

            extra = f"UPLOADED FILES:\n{st.session_state.batch_files_info}\n\n"
            extra += (
                "INSTRUCTIONS: Correlate events by timestamp across ALL uploaded files. "
                "Trace the failure from first symptom to final outcome. "
                "If files are missing from the debug chain, tell the user EXACTLY which file to upload and where to find it."
            )

            with st.spinner("Retrieving and correlating logs..."):
                stream = run_rag(all_entries, q, api_key, "deep", extra_context=extra)
            result = st.write_stream(stream)
            st.session_state.batch_result = result

    if st.session_state.batch_result and st.session_state.batch_iteration > 0:
        st.divider()
        st.markdown(f"**Iteration {st.session_state.batch_iteration}** \u2014 Need deeper analysis?")
        fc1, fc2 = st.columns(2)
        with fc1:
            if st.button("\u2705 Done", key=f"s3_yes_{st.session_state.batch_iteration}"):
                st.success("Debug complete.")
                st.session_state.batch_result = None
                st.session_state.batch_iteration = 0
        with fc2:
            feedback = st.text_input(
                "What to dig into? (e.g., 'check rain at 13:54:15', 'upload uec_1 and re-analyze')",
                key=f"s3_fb_{st.session_state.batch_iteration}",
            )
            if st.button("\U0001f504 Dig Deeper", key=f"s3_no_{st.session_state.batch_iteration}"):
                fb = feedback if feedback else "Trace the failure chain deeper, check all timestamps, suggest missing files"
                extra = ""
                if st.session_state.batch_files_info:
                    extra = f"UPLOADED FILES:\n{st.session_state.batch_files_info}"
                with st.spinner("Retrieving and correlating logs..."):
                    stream = run_rag_iterative(
                        st.session_state.batch_entries,
                        st.session_state.batch_query,
                        api_key, "deep",
                        st.session_state.batch_result, fb,
                        extra_context=extra,
                    )
                result = st.write_stream(stream)
                st.session_state.batch_result = result
                st.session_state.batch_iteration += 1

st.divider()
st.caption("Iterative RAG Agent \u2014 Telecom Deep Debug \u2014 LangChain + ChromaDB + HuggingFace + Groq")

import streamlit as st
import os, re, io, tarfile, zipfile, hashlib
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ─── Pre-compiled regex patterns (compiled once, reused every call) ───
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')
_TIMESTAMP_RE = re.compile(r'^\d{2}:\d{2}:\d{2}\.\d+ \d{2}:\d{2}:\d{2}\.\d+ ')
_IMPORTANT_RE = re.compile(
    r'warn|error|err\b|fail|timeout|loss|latency|delay|congestion|critical|fatal|'
    r'refused|rejected|denied|abort|crash|exception|unreachable|invalid|mismatch|'
    r'drop|retry|disconnect|panic|oom|not found|degraded|down|offline|'
    r'rrc release|ue context release|crc nok|cell setup|nok|ue release|ngap|rach|'
    r'handover|ho failure|rlf|radio link failure|beam failure|pdu session|registration reject|'
    r's1ap|x2ap|f1ap|e1ap|sctp.*fail|gtp.*error|overload|overflow|underflow|'
    r'segfault|core dump|stack trace|authentication fail|integrity fail|cipher fail|'
    r'drb release|srb fail|rlc retx|harq nack|pucch.*fail|prach.*fail',
    re.IGNORECASE,
)
_SEV_ERROR_RE = re.compile(r'error|data loss|timeout|critical|fatal|segfault|core dump|unreachable|abort|panic', re.IGNORECASE)
_SEV_FAIL_RE = re.compile(r'fail', re.IGNORECASE)
_SEV_WARN_RE = re.compile(r'warn|latency|delay|congestion|degraded|retry|overload', re.IGNORECASE)

# ─── Constants ───
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
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )

# ─── Cached LLM client — reused across calls ───
@st.cache_resource
def get_llm(_api_key):
    return ChatGroq(groq_api_key=_api_key, model_name="llama-3.3-70b-versatile", max_tokens=1200, temperature=0)

# ─── Cached vectorstore — keyed by content hash, not rebuilt on rerun ───
@st.cache_resource
def get_vectorstore(_emb, content_hash, entries_tuple, collection_name):
    docs = [
        Document(
            page_content=m,
            metadata={"source": f, "line": ln, "severity": detect_severity(m)},
        )
        for f, ln, m in entries_tuple
    ]
    return Chroma.from_documents(docs, _emb, collection_name=f"{collection_name}_{content_hash[:8]}")

# ─── Log Processing (all regex pre-compiled for speed) ───
def clean_line(line):
    line = _ANSI_RE.sub('', line)
    line = _TIMESTAMP_RE.sub('', line)
    return line.strip()

def is_important(line):
    return bool(_IMPORTANT_RE.search(line))

def detect_severity(msg):
    if _SEV_ERROR_RE.search(msg): return "ERROR"
    if _SEV_FAIL_RE.search(msg): return "FAIL"
    if _SEV_WARN_RE.search(msg): return "WARNING"
    return "INFO"

def parse_content(content, filename):
    """Returns list of (filename, line_number, cleaned_message)."""
    seen, results = set(), []
    for line_no, line in enumerate(content.splitlines(), start=1):
        c = clean_line(line)
        if c and len(c) >= 10 and is_important(c) and c not in seen:
            seen.add(c)
            results.append((filename, line_no, c))
    return results

def parse_all_lines(content, filename):
    """Returns every line with line numbers (for full-flow context)."""
    results = []
    for line_no, line in enumerate(content.splitlines(), start=1):
        c = clean_line(line)
        if c and len(c) >= 5:
            results.append((filename, line_no, c))
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

def process_file_all_lines(uploaded_bytes, filename):
    """Parse ALL lines (not just important) for flow context."""
    return parse_all_lines(uploaded_bytes.decode("utf-8", errors="ignore"), filename)

def format_docs(docs):
    return "\n".join(
        f"[{d.metadata.get('severity','?')}] [{d.metadata.get('source','?')}:L{d.metadata.get('line','?')}] {d.page_content}"
        for d in docs
    )

def _build_chain(entries, api_key, collection_name, prompt_text):
    """Shared chain builder — reuses cached embeddings, vectorstore, and LLM."""
    emb = get_embeddings()
    content_hash = hashlib.md5(str(entries).encode()).hexdigest()
    vs = get_vectorstore(emb, content_hash, tuple(entries), collection_name)
    ret = vs.as_retriever(search_kwargs={"k": min(8, len(entries))})
    llm = get_llm(api_key)
    prompt = ChatPromptTemplate.from_template(prompt_text)
    return ({"context": ret | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())

def run_rag(entries, question, api_key, collection_name="analysis", extra_context=""):
    prompt_text = (
        "You are a telecom log analyst. Analyze the retrieved logs.\n"
        "Each log entry shows [SEVERITY] [file:Line_number] message.\n"
        "ALWAYS reference the exact file name and line number when citing evidence.\n\n"
        "Format:\n"
        "- Root Cause: (one line)\n"
        "- Severity: CRITICAL / HIGH / MEDIUM / LOW\n"
        "- Error Location: (file name and line numbers where the errors occurred)\n"
        "- Details: (explanation referencing specific line numbers as evidence)\n"
        "- Recommendation: (what to do)\n\n"
    )
    if extra_context:
        prompt_text += "Additional context:\n" + extra_context + "\n\n"
    prompt_text += "Logs:\n{context}\n\nQuestion: {question}"
    chain = _build_chain(entries, api_key, collection_name, prompt_text)
    return chain.stream(question)

def run_rag_iterative(entries, question, api_key, collection_name, prev_answer="", feedback="", extra_context=""):
    """Re-analyze with previous answer + user feedback as additional context."""
    prompt_text = (
        "You are a telecom log analyst. The user was NOT satisfied with the previous analysis.\n"
        "Review the previous answer and the user's feedback, then provide a DEEPER, more thorough analysis.\n"
        "Look at different log entries, consider alternative root causes, and be more specific.\n"
        "Each log entry shows [SEVERITY] [file:Line_number] message.\n"
        "ALWAYS reference the exact file name and line number when citing evidence.\n\n"
        "Previous answer:\n" + prev_answer + "\n\n"
        "User feedback: " + feedback + "\n\n"
    )
    if extra_context:
        prompt_text += "Additional context:\n" + extra_context + "\n\n"
    prompt_text += (
        "Format:\n"
        "- Root Cause: (one line)\n"
        "- Severity: CRITICAL / HIGH / MEDIUM / LOW\n"
        "- Error Location: (file name and line numbers where the errors occurred)\n"
        "- Details: (explanation referencing specific line numbers as evidence)\n"
        "- Recommendation: (what to do)\n\n"
        "Logs:\n{context}\n\nQuestion: {question}"
    )
    chain = _build_chain(entries, api_key, collection_name, prompt_text)
    return chain.stream(question)

# ─── Page Setup ───
st.set_page_config(page_title="Iterative RAG Agent", page_icon="\U0001f4e1", layout="wide")
st.markdown("# \U0001f4e1 Iterative RAG Agent — Telecom Log Analyzer")
st.markdown("Upload logs \u2192 Vector retrieval \u2192 AI root cause analysis \u2192 **Iterative refinement**")
st.divider()

FILE_TYPES = ["txt","log","json","csv","xml","html","htm","cfg","tgz","gz","zip"]

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Groq API Key", type="password")
    st.markdown("[Get a free Groq API key here](https://console.groq.com/keys)")
    st.divider()
    st.markdown("**Supported formats:**")
    st.markdown("`.txt` `.log` `.json` `.csv` `.xml` `.html` `.cfg` `.tgz` `.zip`")

if not api_key:
    st.info("Enter your Groq API key in the sidebar to start. [Get one free here](https://console.groq.com/keys)")
    st.stop()

# Pre-warm embedding model on first load
get_embeddings()

# ─── Session state initialization ───
for key in ["single_result", "single_entries", "single_query", "single_iteration",
            "compare_result", "compare_entries", "compare_query", "compare_iteration", "compare_extra",
            "batch_result", "batch_entries", "batch_query", "batch_iteration"]:
    if key not in st.session_state:
        st.session_state[key] = None if "iteration" not in key else 0

tab1, tab2, tab3 = st.tabs(["\U0001f50d Single Analysis", "\u2194\ufe0f Pass vs Fail", "\U0001f4da Batch Analysis"])

# ═══════════════════════════════════════════
# TAB 1: Single Analysis (iterative)
# ═══════════════════════════════════════════
with tab1:
    st.subheader("Upload a log file for AI analysis")
    uploaded = st.file_uploader("Log file", type=FILE_TYPES, key="single")
    query = st.text_input("Question (optional)", placeholder="e.g., Why did UE connection fail?", key="q1")

    if uploaded and st.button("Analyze", type="primary", key="b1"):
        entries, _ = process_file(uploaded)
        if not entries:
            st.warning("No important entries found in this file.")
        else:
            st.session_state.single_entries = entries
            q = query if query else "Analyze all errors and find root cause"
            st.session_state.single_query = q
            st.session_state.single_iteration = 1

            sev = {"ERROR":0,"FAIL":0,"WARNING":0,"INFO":0}
            for _, _, m in entries: sev[detect_severity(m)] += 1
            cols = st.columns(4)
            for i,(s,c) in enumerate(sev.items()): cols[i].metric(f"{SEV_BADGE.get(s,'')} {s}", c)

            with st.expander(f"Log entries ({len(entries)} total)"):
                for fn, ln, m in entries[:50]:
                    st.text(f"{SEV_BADGE.get(detect_severity(m),'')} [{fn}:L{ln}] {m}")

            st.divider()
            with st.spinner("Retrieving relevant logs..."):
                stream = run_rag(entries, q, api_key, "single")
            result = st.write_stream(stream)
            st.session_state.single_result = result

    # ─── Iterative feedback loop ───
    if st.session_state.single_result and st.session_state.single_iteration > 0:
        if st.session_state.single_iteration > 0:
            st.divider()
            st.markdown(f"**Iteration {st.session_state.single_iteration}** — Are you satisfied with the analysis?")
            fc1, fc2 = st.columns(2)
            with fc1:
                if st.button("\u2705 Yes, I'm satisfied", key=f"s1_yes_{st.session_state.single_iteration}"):
                    st.success("Great! Analysis complete.")
                    st.session_state.single_result = None
                    st.session_state.single_iteration = 0
            with fc2:
                feedback = st.text_input("What's missing or wrong?", key=f"s1_fb_{st.session_state.single_iteration}",
                                         placeholder="e.g., Look deeper at the handover errors")
                if st.button("\U0001f504 Re-analyze", key=f"s1_no_{st.session_state.single_iteration}"):
                    fb = feedback if feedback else "Please analyze more thoroughly, look at different entries"
                    with st.spinner("Retrieving relevant logs..."):
                        stream = run_rag_iterative(
                            st.session_state.single_entries,
                            st.session_state.single_query,
                            api_key, "single",
                            st.session_state.single_result, fb,
                        )
                    result = st.write_stream(stream)
                    st.session_state.single_result = result
                    st.session_state.single_iteration += 1

# ═══════════════════════════════════════════
# TAB 2: Pass vs Fail (flow-aware comparison)
# ═══════════════════════════════════════════
with tab2:
    st.subheader("Compare PASS and FAIL logs — Flow-Aware Analysis")
    st.caption("The AI sees the full PASS log flow to understand what succeeded, then identifies where the FAIL log diverged.")
    c1, c2 = st.columns(2)
    with c1: pass_file = st.file_uploader("PASS log", type=FILE_TYPES, key="pass")
    with c2: fail_file = st.file_uploader("FAIL log", type=FILE_TYPES, key="fail")

    if pass_file and fail_file and st.button("Compare", type="primary", key="b2"):
        pass_entries, pass_raw = process_file(pass_file)
        fail_entries, fail_raw = process_file(fail_file)

        # Full flow of both logs (all lines, not just important)
        pass_all = parse_all_lines(pass_raw.decode("utf-8", errors="ignore"), pass_file.name)
        fail_all = parse_all_lines(fail_raw.decode("utf-8", errors="ignore"), fail_file.name)

        pass_msgs = set(m for _, _, m in pass_entries)
        fail_only = [(f, ln, m) for f, ln, m in fail_entries if m not in pass_msgs]
        common = [(f, ln, m) for f, ln, m in fail_entries if m in pass_msgs]

        mc = st.columns(4)
        mc[0].metric("PASS entries", len(pass_entries))
        mc[1].metric("FAIL entries", len(fail_entries))
        mc[2].metric("Common (shared)", len(common))
        mc[3].metric("FAIL-only (critical)", len(fail_only))

        # Show pass flow
        with st.expander("PASS log flow (full sequence)"):
            for fn, ln, m in pass_all[:60]:
                sev = detect_severity(m) if is_important(m) else "—"
                st.text(f"L{ln:>4}  {m}")

        # Show fail flow with divergence markers
        with st.expander("FAIL log flow (divergences marked \U0001f534)", expanded=True):
            for fn, ln, m in fail_all[:60]:
                is_fail_only = m not in pass_msgs and is_important(m)
                marker = "\U0001f534" if is_fail_only else "  "
                st.text(f"{marker} L{ln:>4}  {m}")

        if fail_only:
            with st.expander("FAIL-only errors (not in PASS log)", expanded=True):
                for fn, ln, m in fail_only[:30]:
                    st.text(f"\U0001f534 [{fn}:L{ln}] {m}")

        # Build pass-flow context for the LLM
        pass_flow_summary = "PASS LOG FLOW (what succeeded):\n"
        pass_flow_summary += "\n".join(f"  L{ln}: {m}" for _, ln, m in pass_all[:40])
        pass_flow_summary += f"\n\nFAIL LOG FLOW (step by step):\n"
        pass_flow_summary += "\n".join(f"  L{ln}: {m}" for _, ln, m in fail_all[:40])

        analysis_entries = fail_only if fail_only else fail_entries
        if analysis_entries:
            st.divider()
            question = (
                "Compare the PASS and FAIL log flows provided in the additional context. "
                "The PASS log shows the normal successful sequence. "
                "Identify at which exact step/line the FAIL log diverged from the PASS log. "
                "What went wrong and why? Reference specific line numbers from both logs."
            )
            st.session_state.compare_query = question
            st.session_state.compare_entries = analysis_entries
            st.session_state.compare_extra = pass_flow_summary
            st.session_state.compare_iteration = 1

            with st.spinner("Retrieving relevant logs..."):
                stream = run_rag(analysis_entries, question, api_key, "comparison", extra_context=pass_flow_summary)
            result = st.write_stream(stream)
            st.session_state.compare_result = result
        else:
            st.success("No unique errors in FAIL log — both logs match.")

    # ─── Iterative feedback loop for Pass vs Fail ───
    if st.session_state.compare_result and st.session_state.compare_iteration > 0:
        st.divider()
        st.markdown(f"**Iteration {st.session_state.compare_iteration}** — Are you satisfied with the comparison?")
        fc1, fc2 = st.columns(2)
        with fc1:
            if st.button("\u2705 Yes, I'm satisfied", key=f"s2_yes_{st.session_state.compare_iteration}"):
                st.success("Great! Comparison complete.")
                st.session_state.compare_result = None
                st.session_state.compare_iteration = 0
        with fc2:
            feedback = st.text_input("What's missing or wrong?", key=f"s2_fb_{st.session_state.compare_iteration}",
                                     placeholder="e.g., Compare the registration steps more carefully")
            if st.button("\U0001f504 Re-analyze", key=f"s2_no_{st.session_state.compare_iteration}"):
                fb = feedback if feedback else "Please analyze more thoroughly, compare the flows step by step"
                with st.spinner("Retrieving relevant logs..."):
                    stream = run_rag_iterative(
                        st.session_state.compare_entries,
                        st.session_state.compare_query,
                        api_key, "comparison",
                        st.session_state.compare_result, fb,
                        extra_context=st.session_state.compare_extra or "",
                    )
                result = st.write_stream(stream)
                st.session_state.compare_result = result
                st.session_state.compare_iteration += 1

# ═══════════════════════════════════════════
# TAB 3: Batch Analysis (iterative)
# ═══════════════════════════════════════════
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
            q = bq if bq else "Analyze all errors and find root cause"
            st.session_state.batch_entries = all_entries
            st.session_state.batch_query = q
            st.session_state.batch_iteration = 1

            with st.spinner("Retrieving relevant logs..."):
                stream = run_rag(all_entries, q, api_key, "batch")
            result = st.write_stream(stream)
            st.session_state.batch_result = result

    # ─── Iterative feedback loop for Batch ───
    if st.session_state.batch_result and st.session_state.batch_iteration > 0:
        st.divider()
        st.markdown(f"**Iteration {st.session_state.batch_iteration}** — Are you satisfied with the analysis?")
        fc1, fc2 = st.columns(2)
        with fc1:
            if st.button("\u2705 Yes, I'm satisfied", key=f"s3_yes_{st.session_state.batch_iteration}"):
                st.success("Great! Analysis complete.")
                st.session_state.batch_result = None
                st.session_state.batch_iteration = 0
        with fc2:
            feedback = st.text_input("What's missing or wrong?", key=f"s3_fb_{st.session_state.batch_iteration}",
                                     placeholder="e.g., Focus on the SCTP errors in log2")
            if st.button("\U0001f504 Re-analyze", key=f"s3_no_{st.session_state.batch_iteration}"):
                fb = feedback if feedback else "Please analyze more thoroughly, look at different entries"
                with st.spinner("Retrieving relevant logs..."):
                    stream = run_rag_iterative(
                        st.session_state.batch_entries,
                        st.session_state.batch_query,
                        api_key, "batch",
                        st.session_state.batch_result, fb,
                    )
                result = st.write_stream(stream)
                st.session_state.batch_result = result
                st.session_state.batch_iteration += 1

st.divider()
st.caption("Iterative RAG Agent \u2014 LangChain + ChromaDB + HuggingFace + Groq")
