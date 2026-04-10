"""
Core analysis engine — shared logic for CLI (rag_system.py) and Web UI (app.py).
Handles log parsing, filtering, auto-detection, embedding, and LLM analysis.
"""

import os
import re
import pickle
import tarfile
import zipfile
import json
import csv
import io
import numpy as np
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Constants
# -----------------------------
VECTOR_DIR = "vector_store"
INDEX_FILE = os.path.join(VECTOR_DIR, "faiss_index.bin")
LOG_FILE = os.path.join(VECTOR_DIR, "logs.pkl")
LOG_FOLDER = "data/logs"
LOG_HASH_FILE = os.path.join(VECTOR_DIR, "log_hash.txt")

SUPPORTED_EXTENSIONS = (".txt", ".log", ".json", ".cfg", ".csv", ".xml", ".html", ".htm")
ARCHIVE_EXTENSIONS = (".tgz", ".tar.gz", ".zip")

INITIAL_CHECK_PATTERNS = ["egate_console.log", "e2e_console_output.log", "log.html"]

ARCHIVE_IMPORTANT_PATTERNS = [
    "syslog", "messages", "dmesg", "journalctl", "kern.log",
    "daemon.log", "boot.log", "auth.log", "cron.log",
    "worker", "egate", "alarm", "error",
    # Telecom-specific files inside sosreport/archives
    "uec_1", "uec_2", "uec_", "btslog", "syslog_files",
    "rain", "runtime", "l3_log", "airphone", "wts",
    "gnb", "enb", "cu_cp", "cu_up", "du_", "l2nrt",
    "ctrl_conn", "firewall", "config"
]

IMPORTANT_KEYWORDS = [
    "warn", "error", "err", "fail", "timeout", "loss",
    "latency", "delay", "congestion", "critical", "fatal",
    "refused", "rejected", "denied", "abort", "crash",
    "exception", "unreachable", "invalid", "mismatch",
    "obsolete", "drop", "retry", "disconnect",
    "panic", "oom", "killed", "segfault", "oops",
    "not found", "degraded", "down", "offline",
    # Telecom-specific error keywords
    "rrc release", "ue context release", "rrc_release",
    "crc nok", "crc:nok", "cell setup", "not received",
    "ctrl_del_ue", "cancel all active fsm", "is idle",
    "sib", "systeminformationblock", "pbchdecoder",
    "firewall", "gtp traffic", "trs firewall",
    "cell sync", "nok", "ue release", "trigger ue release",
    "ngap", "rach", "handover", "ho failure",
    "rlf", "radio link failure", "beam failure",
    "sctp", "x2", "xn", "f1", "e1", "n2", "n3",
    "drb", "srb", "bearer", "pdu session",
    "registration reject", "service reject", "attach reject"
]

# -----------------------------
# Log type detection
# -----------------------------
LOG_TYPE_PATTERNS = {
    "egate_console": {
        "file_patterns": ["egate_console", "egate"],
        "keywords": ["egate", "simulator", "bts", "enb", "gnb", "ue", "cell", "carrier", "uec-", "amf-"],
        "description": "eGate Simulator Console Log"
    },
    "e2e_test": {
        "file_patterns": ["e2e_console", "e2e_output", "test_output"],
        "keywords": ["test case", "testcase", "tc_", "pass", "fail", "verdict", "suite"],
        "description": "End-to-End Test Output"
    },
    "robot_framework": {
        "file_patterns": ["log.html", "output.xml", "report.html"],
        "keywords": ["robot", "keyword", "test suite", "test case", "<robot"],
        "description": "Robot Framework Log"
    },
    "syslog": {
        "file_patterns": ["syslog", "messages", "kern.log", "daemon.log", "syslog_files"],
        "keywords": ["systemd", "kernel", "sshd", "cron", "rsyslog"],
        "description": "Linux System Log"
    },
    "uec_log": {
        "file_patterns": ["uec_1", "uec_2", "uec_"],
        "keywords": ["nrrrcmsghandler", "rrcrelease", "rfma_impl", "rrcfsm", "dl-dcch-nr"],
        "description": "UE Controller Log (from sosreport/wts)"
    },
    "rain_runtime": {
        "file_patterns": ["rain", "runtime", "cp_ue", "cu_cp", "cu_up"],
        "keywords": ["cp_ue", "cuurelease", "concreteuesa", "ue release", "trigger ue release"],
        "description": "RAIN / gNB Runtime Log"
    },
    "btslog": {
        "file_patterns": ["btslog", "l3_log", "airphone", "l2nrt"],
        "keywords": ["cell setup", "cellid", "rfm_event", "inputmessagehandler", "pbchdecoder", "crc nok", "ssburst"],
        "description": "BTS / Airphone Runtime Log"
    },
    "worker_log": {
        "file_patterns": ["worker", "executor", "runner"],
        "keywords": ["worker", "executor", "job", "task", "queue", "pid"],
        "description": "Worker/Executor Log"
    },
    "sosreport": {
        "file_patterns": ["sosreport", "sos_", "wts"],
        "keywords": [],
        "description": "SOSReport Archive"
    },
    "json_log": {
        "file_patterns": [],
        "keywords": [],
        "description": "JSON Structured Log"
    },
    "csv_log": {
        "file_patterns": [],
        "keywords": [],
        "description": "CSV Structured Log"
    },
    "xml_log": {
        "file_patterns": [],
        "keywords": [],
        "description": "XML Log"
    },
    "generic": {
        "file_patterns": [],
        "keywords": [],
        "description": "Generic Text Log"
    }
}


def detect_log_type(filename, content_sample=""):
    """Auto-detect log type based on filename and content."""
    fname_lower = filename.lower()

    # Check by file extension first
    if fname_lower.endswith(".json"):
        return "json_log"
    if fname_lower.endswith(".csv"):
        return "csv_log"
    if fname_lower.endswith((".xml", ".html", ".htm")):
        if any(kw in content_sample.lower() for kw in ["<robot", "robot framework"]):
            return "robot_framework"
        return "xml_log"

    # Check by filename patterns
    for log_type, config in LOG_TYPE_PATTERNS.items():
        for pattern in config["file_patterns"]:
            if pattern in fname_lower:
                return log_type

    # Check by content keywords
    content_lower = content_sample.lower()
    for log_type, config in LOG_TYPE_PATTERNS.items():
        if config["keywords"]:
            matches = sum(1 for kw in config["keywords"] if kw in content_lower)
            if matches >= 2:
                return log_type

    return "generic"


# -----------------------------
# Log cleaning & filtering
# -----------------------------
def clean_line(line):
    """Strip ANSI escape codes and timestamps."""
    line = re.sub(r'\x1b\[[0-9;]*m', '', line)
    line = re.sub(r'^\d{2}:\d{2}:\d{2}\.\d+ \d{2}:\d{2}:\d{2}\.\d+ ', '', line)
    return line.strip()


def is_important(line):
    """Check if a log line contains something worth analyzing."""
    lower = line.lower()
    return any(kw in lower for kw in IMPORTANT_KEYWORDS)


def detect_severity(log_message):
    """Detect severity level from log message."""
    log_lower = log_message.lower()
    if any(kw in log_lower for kw in ["error", "data loss", "timeout"]):
        return "ERROR"
    elif "failed" in log_lower:
        return "FAIL"
    elif any(kw in log_lower for kw in ["warning", "latency", "delay", "congestion"]):
        return "WARNING"
    return "INFO"


# -----------------------------
# Format-specific parsers
# -----------------------------
def parse_json_log(content, filename):
    """Parse JSON log files (single object or JSON-lines)."""
    results = []
    seen = set()

    # Try JSON-lines format first
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            # Flatten JSON to a readable string
            flat = " | ".join(f"{k}={v}" for k, v in obj.items() if v)
            if flat and len(flat) >= 10 and is_important(flat) and flat not in seen:
                seen.add(flat)
                results.append((filename, flat))
        except json.JSONDecodeError:
            # Not JSON-lines, try as plain text
            cleaned = clean_line(line)
            if cleaned and len(cleaned) >= 10 and is_important(cleaned) and cleaned not in seen:
                seen.add(cleaned)
                results.append((filename, cleaned))

    # Try single JSON object if no results
    if not results:
        try:
            obj = json.loads(content)
            if isinstance(obj, list):
                for item in obj:
                    flat = " | ".join(f"{k}={v}" for k, v in item.items() if v)
                    if flat and is_important(flat) and flat not in seen:
                        seen.add(flat)
                        results.append((filename, flat))
        except (json.JSONDecodeError, AttributeError):
            pass

    return results


def parse_csv_log(content, filename):
    """Parse CSV log files."""
    results = []
    seen = set()
    try:
        reader = csv.DictReader(io.StringIO(content))
        for row in reader:
            flat = " | ".join(f"{k}={v}" for k, v in row.items() if v)
            if flat and len(flat) >= 10 and is_important(flat) and flat not in seen:
                seen.add(flat)
                results.append((filename, flat))
    except Exception:
        # Fall back to plain text
        for line in content.splitlines():
            cleaned = clean_line(line)
            if cleaned and len(cleaned) >= 10 and is_important(cleaned) and cleaned not in seen:
                seen.add(cleaned)
                results.append((filename, cleaned))
    return results


def parse_xml_log(content, filename):
    """Parse XML/HTML log files — extract text content."""
    results = []
    seen = set()
    # Simple regex-based extraction (no lxml dependency)
    text_content = re.sub(r'<[^>]+>', ' ', content)
    for line in text_content.splitlines():
        cleaned = clean_line(line)
        if cleaned and len(cleaned) >= 10 and is_important(cleaned) and cleaned not in seen:
            seen.add(cleaned)
            results.append((filename, cleaned))
    return results


# -----------------------------
# Unified log reader (auto-detects format)
# -----------------------------
def read_log_content(content, filename):
    """Read and filter log content, auto-detecting format."""
    content_sample = content[:2000]
    log_type = detect_log_type(filename, content_sample)

    if log_type == "json_log":
        return parse_json_log(content, filename), log_type
    elif log_type == "csv_log":
        return parse_csv_log(content, filename), log_type
    elif log_type in ("xml_log", "robot_framework"):
        return parse_xml_log(content, filename), log_type
    else:
        # Standard text log parsing
        seen = set()
        results = []
        for line in content.splitlines():
            cleaned = clean_line(line)
            if not cleaned or len(cleaned) < 10:
                continue
            if is_important(cleaned) and cleaned not in seen:
                seen.add(cleaned)
                results.append((filename, cleaned))
        return results, log_type


def read_filtered_logs(filepath, filename):
    """Read a log file from disk, filter to important unique lines."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    entries, log_type = read_log_content(content, filename)
    return entries


def read_all_lines(content, filename):
    """Read ALL lines from content (no filtering). Used for pass/fail comparison."""
    results = []
    for line in content.splitlines():
        cleaned = clean_line(line)
        if cleaned and len(cleaned) >= 5:
            results.append((filename, cleaned))
    return results


# -----------------------------
# Archive handling
# -----------------------------
def extract_and_read_archive(archive_path, archive_name):
    """Extract important log files from .tgz/.zip archives and filter them."""
    results = []

    try:
        if archive_path.endswith((".tgz", ".tar.gz")):
            with tarfile.open(archive_path, "r:gz") as tar:
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    member_lower = member.name.lower()
                    if not any(p in member_lower for p in ARCHIVE_IMPORTANT_PATTERNS):
                        continue
                    if member.size > 50 * 1024 * 1024:
                        continue
                    try:
                        f = tar.extractfile(member)
                        if f is None:
                            continue
                        content = f.read().decode("utf-8", errors="ignore")
                        short_name = f"{archive_name}/{os.path.basename(member.name)}"
                        entries, _ = read_log_content(content, short_name)
                        results.extend(entries)
                    except Exception:
                        continue

        elif archive_path.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zf:
                for name in zf.namelist():
                    name_lower = name.lower()
                    if not any(p in name_lower for p in ARCHIVE_IMPORTANT_PATTERNS):
                        continue
                    info = zf.getinfo(name)
                    if info.file_size > 50 * 1024 * 1024:
                        continue
                    try:
                        content = zf.read(name).decode("utf-8", errors="ignore")
                        short_name = f"{archive_name}/{os.path.basename(name)}"
                        entries, _ = read_log_content(content, short_name)
                        results.extend(entries)
                    except Exception:
                        continue

    except Exception:
        pass

    return results


def extract_archive_bytes(archive_bytes, archive_name):
    """Extract from in-memory archive bytes (for web uploads)."""
    results = []
    try:
        if archive_name.endswith((".tgz", ".tar.gz")):
            with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
                for member in tar.getmembers():
                    if not member.isfile() or member.size > 50 * 1024 * 1024:
                        continue
                    member_lower = member.name.lower()
                    if not any(p in member_lower for p in ARCHIVE_IMPORTANT_PATTERNS):
                        continue
                    try:
                        f = tar.extractfile(member)
                        if f is None:
                            continue
                        content = f.read().decode("utf-8", errors="ignore")
                        short_name = f"{archive_name}/{os.path.basename(member.name)}"
                        entries, _ = read_log_content(content, short_name)
                        results.extend(entries)
                    except Exception:
                        continue
        elif archive_name.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as zf:
                for name in zf.namelist():
                    name_lower = name.lower()
                    if not any(p in name_lower for p in ARCHIVE_IMPORTANT_PATTERNS):
                        continue
                    info = zf.getinfo(name)
                    if info.file_size > 50 * 1024 * 1024:
                        continue
                    try:
                        content = zf.read(name).decode("utf-8", errors="ignore")
                        short_name = f"{archive_name}/{os.path.basename(name)}"
                        entries, _ = read_log_content(content, short_name)
                        results.extend(entries)
                    except Exception:
                        continue
    except Exception:
        pass
    return results


def scan_log_folder(folder):
    """Scan folder recursively for log files and archives."""
    all_logs = []
    for root, dirs, files in os.walk(folder):
        for file in sorted(files):
            filepath = os.path.join(root, file)
            rel_path = os.path.relpath(filepath, folder)
            if file.endswith(SUPPORTED_EXTENSIONS):
                entries = read_filtered_logs(filepath, rel_path)
                all_logs.extend(entries)
            elif file.endswith(ARCHIVE_EXTENSIONS):
                entries = extract_and_read_archive(filepath, rel_path)
                all_logs.extend(entries)
    return all_logs


# Max size per file when scanning large directories (e.g., sosreport)
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB

# Extensions to scan inside sosreport/large directories
SCAN_EXTENSIONS = (
    ".txt", ".log", ".json", ".cfg", ".csv", ".xml", ".html", ".htm",
    ".conf", ".out", ".err", ""
)


def scan_local_path(path, targeted_patterns=None):
    """Scan a local directory path (e.g. sosreport folder) for log files.

    Args:
        path: absolute path to a directory or file
        targeted_patterns: optional list of filename patterns to look for
                          (e.g. ["uec_1", "syslog", "rain"]). If provided,
                          only files matching these patterns are read.

    Returns:
        tuple: (entries, file_index)
            entries: list of (filename, log_line) tuples
            file_index: dict mapping short_name -> absolute_path for all found files
    """
    path = os.path.abspath(path)
    entries = []
    file_index = {}  # short_name -> full_path (for later targeted lookups)

    if not os.path.exists(path):
        return entries, file_index

    # Single file
    if os.path.isfile(path):
        fname = os.path.basename(path)
        file_index[fname] = path
        if targeted_patterns is None or any(p.lower() in fname.lower() for p in targeted_patterns):
            try:
                size = os.path.getsize(path)
                if size <= MAX_FILE_SIZE_BYTES:
                    file_entries = read_filtered_logs(path, fname)
                    entries.extend(file_entries)
            except Exception:
                pass
        return entries, file_index

    # Directory — walk and index all files, but only read matching ones
    for root, dirs, files in os.walk(path):
        # Skip hidden dirs, __pycache__, etc
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        for file in sorted(files):
            filepath = os.path.join(root, file)
            rel_path = os.path.relpath(filepath, path)
            short_name = rel_path.replace("\\", "/")
            file_index[short_name] = filepath

            # Decide if we should read this file
            should_read = False
            file_lower = file.lower()

            if targeted_patterns:
                # Only read files matching the targeted patterns
                should_read = any(p.lower() in file_lower or p.lower() in short_name.lower()
                                  for p in targeted_patterns)
            else:
                # Default: read files matching important patterns or extensions
                has_ext = any(file_lower.endswith(ext) for ext in SCAN_EXTENSIONS if ext)
                no_ext = '.' not in file  # many log files have no extension
                is_important = any(p in file_lower or p in short_name.lower()
                                   for p in ARCHIVE_IMPORTANT_PATTERNS)
                should_read = (has_ext or no_ext) and is_important

            if should_read:
                try:
                    size = os.path.getsize(filepath)
                    if size > MAX_FILE_SIZE_BYTES:
                        continue
                    file_entries = read_filtered_logs(filepath, short_name)
                    entries.extend(file_entries)
                except Exception:
                    continue

    return entries, file_index


def scan_path_for_files(path, requested_files):
    """Search a local path for specific files the AI requested.

    Args:
        path: base directory to search in
        requested_files: list of filename patterns (e.g. ["uec_1.log", "syslog"])

    Returns:
        list of (filename, log_line) tuples from matched files
    """
    entries, _ = scan_local_path(path, targeted_patterns=requested_files)
    return entries


def list_local_path_files(path, max_files=200):
    """List files in a local path (for showing the user what's available).

    Returns:
        list of dicts: [{name, rel_path, size_mb, is_important}]
    """
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        return []

    file_list = []
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        for file in sorted(files):
            filepath = os.path.join(root, file)
            rel_path = os.path.relpath(filepath, path).replace("\\", "/")
            try:
                size = os.path.getsize(filepath)
            except Exception:
                size = 0
            size_mb = round(size / (1024 * 1024), 2)
            file_lower = file.lower()
            important = any(p in file_lower or p in rel_path.lower()
                            for p in ARCHIVE_IMPORTANT_PATTERNS)
            file_list.append({
                "name": file,
                "rel_path": rel_path,
                "size_mb": size_mb,
                "is_important": important
            })
            if len(file_list) >= max_files:
                return file_list
    return file_list


# -----------------------------
# Groq LLM client
# -----------------------------
def get_groq_client():
    """Get Groq client with API key."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


# =============================================================
# TELECOM-SPECIFIC PROMPT ENGINEERING
# =============================================================

SYSTEM_PROMPT_ANALYZE = """You are an expert telecom automated testing log analyst for 5G gNB (and 4G eNB) testlines.

You understand the full telecom test environment:
- AP (Airphone) and gNB software builds are installed on testline hardware
- After installation, Robot Framework test cases are executed
- Results are in log.html, and detailed traces are in egate_console, syslogs, sosreport

Two categories of failures:
1. INSTALLATION FAILURES: No log.html / egate / syslogs available. Only e2e_console or worker logs exist.
   → Focus on build installation errors, dependency issues, connectivity problems.
2. TEST CASE FAILURES: log.html, egate, syslogs, sosreport are all available.
   → Follow the standard debug flow below.

Standard telecom debug flow for test case failures:
1. Check log.html → identify WHICH test case failed and the failure message
2. Check egate_console → find what happened to that test case, note the EXACT TIMING of failure
3. Using the timing from egate, check syslogs/sosreport → what happened at that time on the system
4. Drill into specific files:
   - UE release issues → check uec_1.log (from sosreport → wts → log → uec_1.log)
   - Cell setup issues → check btslog → syslog_files, L3 runtime logs, airphone logs
   - gNB-side events → check rain runtime logs (cp_ue, cu_cp, cu_up)
   - Network/firewall issues → check TRS Firewall logs, GTP traffic logs

Key telecom error patterns you MUST recognize:
- UE RRC Release from gNB: "Starting RRC Release on Receiving RRC Release Message from GNB" → check uec_1.log to confirm, then rain runtime for gNB-side trigger
- UE Context Release: "UE CONTEXT RELEASE COMPLETE" with AMF_UE_NGAP_ID / RAN_UE_NGAP_ID
- CRC NOK: "CRC:NOK" or "Received CRC NOK for ssBlockId" → cell sync / cell setup issue
- Cell Setup Failure: "Cell Setup completed" for some cells but not others → check which cell IDs are missing
- SIB not received: "SystemInformationBlockType1 not received" → cell not broadcasting SIB
- Firewall issues: excessive "TRS Firewall for GTP traffic" prints (compare count with PASS logs)
- Radio Link Failure (RLF): beam failure, RACH failure, handover failure

ALWAYS:
- Note EXACT timestamps from error logs
- Correlate timestamps across different log files
- Identify if error is on UE side (egate/uec) vs gNB side (rain/btslog) vs infra (syslog)
- Compare error counts between PASS and FAIL if both are available

Format your response as:
**Root Cause:** [one-line summary]
**Severity:** [CRITICAL / HIGH / MEDIUM / LOW]
**Category:** [Infrastructure / Product / Configuration / Environment / Timing / Radio / Protocol]
**Affected Component:** [UE / gNB / Cell / AMF / Testline / Build]
**Timestamps:** [key timestamps from the logs]
**Details:** [detailed explanation with specific log evidence]
**Recommendation:** [actionable fix — what to check, what to change]"""

SYSTEM_PROMPT_COMPARE = """You are an expert telecom 5G/4G testline log analyst specializing in PASS vs FAIL log comparison.

Context: AP and gNB builds are installed on testline hardware, then Robot Framework test cases run.
The user will provide PASS and FAIL logs from different test runs.

Your job is to:
1. Find errors ONLY in the FAIL log → these are the real cause
2. Identify errors common in both → IGNORABLE (normal infra noise)
3. Count recurring patterns: e.g., "TRS Firewall for GTP traffic" may appear 10x in PASS but 512x in FAIL
4. Compare timing: errors happening at different times may indicate a different root cause
5. Check for missing events: something that EXISTS in PASS but is ABSENT in FAIL (e.g., cell setup for a cell ID)

Key comparison patterns:
- CRC NOK count difference between PASS and FAIL
- TRS Firewall print count (high count = possible firewall issue on testline)
- Cell setup completion: all cells in PASS vs missing cells in FAIL
- UE release patterns: normal in PASS vs unexpected in FAIL
- SIB reception: successful in PASS vs "not received" in FAIL

Format your response as:
**Ignorable Errors (present in both PASS and FAIL):**
- [list with counts if relevant]

**Critical Errors (ONLY in FAIL log):**
- [list with timestamps]

**Suspicious Count Differences:**
- [patterns that appear much more in FAIL than PASS]

**Missing Events (present in PASS, absent in FAIL):**
- [events that should have happened but didn't]

**Root Cause:** [one-line summary]
**Severity:** [CRITICAL / HIGH / MEDIUM / LOW]
**Affected Component:** [UE / gNB / Cell / AMF / Testline / Build]
**Details:** [detailed explanation referencing specific log evidence]
**Recommendation:** [actionable fix]"""

SYSTEM_PROMPT_INVESTIGATE = """You are a telecom 5G/4G testline log debugging agent. You follow a structured investigation flow.

You understand the telecom testline environment:
- AP (Airphone) and gNB builds are installed on testline hardware by developers
- After installation, Robot Framework test cases are run
- If INSTALLATION fails: only e2e_console or worker logs are available (no log.html, no egate)
- If TEST CASE fails: log.html, egate_console, syslogs, sosreport etc. are available

=== YOUR INVESTIGATION FLOW ===

Phase 1: Determine failure type
- If log.html / egate_console are available → TEST CASE failure → go to Phase 2
- If only e2e_console / worker logs → INSTALLATION failure → analyze those directly

Phase 2: Check log.html / e2e_console
- Find WHICH test case failed and the failure message
- Note the test case name, error description
- REQUEST: egate_console.log (to see what happened during that test case)

Phase 3: Check egate_console
- Find what happened during the failed test case
- Note EXACT TIMESTAMPS of errors (e.g., "13:54:15.592")
- Look for: RRC Release, UE Context Release, registration failures, bearer setup failures
- Based on what you find, request the right deep-dive file:
  * UE release / RRC issues → REQUEST: uec_1.log (from sosreport → wts → log → uec_1.log)
  * Cell setup / CRC issues → REQUEST: btslog syslog_files, L3 runtime logs
  * gNB-side events → REQUEST: rain runtime log (cp_ue logs)
  * System crashes → REQUEST: syslog, dmesg

Phase 4: Deep-dive into specific logs
- Correlate TIMESTAMPS from egate with timestamps in the new file
- Confirm the root cause with evidence from multiple files
- If UE release: confirm in uec_1.log (look for "rrcRelease", "rfma_impl", "RrcFsm")
- Then check gNB side: rain runtime (look for "trigger ue release", "CuUeReleaseSaEventHandler")
- If cell setup issue: check airphone logs for CRC NOK, SsBurstHandler errors
- If firewall issue: count "TRS Firewall for GTP traffic" prints (normal is few, problematic is 500+)

Phase 5: Final diagnosis
- Provide root cause with evidence from ALL files checked
- Include specific timestamps and log lines as proof
- Identify affected component (UE, gNB, cell, AMF, testline infra)

=== IMPORTANT RULES ===
- Always note TIMESTAMPS from error messages
- Always tell the user WHY you need a specific file
- Suggest the exact file path when possible (e.g., "sosreport → wts → log → uec_1.log")
- If a requested file is not available, explain what you can still conclude and what remains uncertain
- When you have enough evidence, provide final analysis — don't keep requesting files unnecessarily

Reply with this exact format at the end:
CHECK_NEXT: filename1, filename2
Or if investigation is complete:
CHECK_NEXT: DONE"""


def ask_llm(groq_client, messages, max_tokens=800):
    """Send messages to Groq LLM and get response."""
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI Error: {e}"


# =============================================================
# PASS vs FAIL COMPARISON
# =============================================================

def compare_pass_fail(pass_content, fail_content, pass_name, fail_name, groq_client):
    """Compare PASS and FAIL logs, find unique errors in FAIL."""

    # Parse all lines (not just important) for comparison
    pass_lines = set()
    for line in pass_content.splitlines():
        cleaned = clean_line(line)
        if cleaned and len(cleaned) >= 10:
            pass_lines.add(cleaned)

    fail_lines = []
    fail_set = set()
    for line in fail_content.splitlines():
        cleaned = clean_line(line)
        if cleaned and len(cleaned) >= 10 and cleaned not in fail_set:
            fail_set.add(cleaned)
            fail_lines.append(cleaned)

    # Classify lines
    common_errors = []
    fail_only_errors = []
    fail_only_all = []

    for line in fail_lines:
        is_in_pass = line in pass_lines
        is_err = is_important(line)

        if is_err and is_in_pass:
            common_errors.append(line)
        elif is_err and not is_in_pass:
            fail_only_errors.append(line)

        if not is_in_pass:
            fail_only_all.append(line)

    # Build summary for LLM
    common_summary = "\n".join(f"  - {l}" for l in common_errors[:20]) or "  (none)"
    fail_only_summary = "\n".join(f"  - {l}" for l in fail_only_errors[:30]) or "  (none)"

    stats = {
        "total_pass_lines": len(pass_lines),
        "total_fail_lines": len(fail_set),
        "common_errors": len(common_errors),
        "fail_only_errors": len(fail_only_errors),
        "fail_only_all": len(fail_only_all),
    }

    # Ask LLM for analysis
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_COMPARE},
        {"role": "user", "content": (
            f"PASS log: {pass_name} ({stats['total_pass_lines']} lines)\n"
            f"FAIL log: {fail_name} ({stats['total_fail_lines']} lines)\n\n"
            f"Errors in BOTH pass and fail (ignorable):\n{common_summary}\n\n"
            f"Errors ONLY in fail log (critical):\n{fail_only_summary}"
        )}
    ]

    analysis = ask_llm(groq_client, messages, max_tokens=1000)

    return {
        "stats": stats,
        "common_errors": common_errors,
        "fail_only_errors": fail_only_errors,
        "fail_only_all": fail_only_all,
        "analysis": analysis
    }


# =============================================================
# SINGLE LOG ANALYSIS
# =============================================================

def _parse_check_next(response):
    """Extract CHECK_NEXT file list from LLM response."""
    for line in response.splitlines():
        if line.strip().upper().startswith("CHECK_NEXT:"):
            value = line.split(":", 1)[1].strip()
            if value.upper() == "DONE":
                return None
            return [f.strip() for f in value.split(",") if f.strip()]
    return None


def _build_initial_context(logs, query):
    """Build initial error summary and file list for investigation."""
    initial_errors = {}
    for filename, msg in logs:
        fname_lower = filename.lower()
        for pattern in INITIAL_CHECK_PATTERNS:
            if pattern in fname_lower or pattern.replace(".", "") in fname_lower.replace(".", ""):
                if filename not in initial_errors:
                    initial_errors[filename] = []
                initial_errors[filename].append(msg)
                break

    error_summary = ""
    for fname, entries in initial_errors.items():
        error_summary += f"\n[{fname}] ({len(entries)} issues):\n"
        for entry in entries[:15]:
            error_summary += f"  - {entry}\n"

    if not error_summary:
        error_summary = "\n[All files]:\n"
        for fname, msg in logs[:30]:
            error_summary += f"  - [{fname}] {msg}\n"

    available_files = sorted(set(f for f, _ in logs))
    file_list = "\n".join(f"  - {f}" for f in available_files)

    return error_summary, file_list


def analyze_logs(logs, groq_client, query="Analyze all errors and find root cause"):
    """Run full LLM-driven investigation on logs (non-interactive, for batch/backward compat)."""

    error_summary, file_list = _build_initial_context(logs, query)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_INVESTIGATE},
        {"role": "user", "content": (
            f"User question: {query}\n\n"
            f"Initial error scan:{error_summary}\n\n"
            f"Available files:\n{file_list}"
        )}
    ]

    steps = []

    response = ask_llm(groq_client, messages)
    steps.append({"title": "Initial Diagnosis", "content": response})
    messages.append({"role": "assistant", "content": response})

    for iteration in range(3):
        requested_files = _parse_check_next(response)
        if not requested_files:
            break

        deep_logs = ""
        for pattern in requested_files:
            pattern_lower = pattern.lower()
            matched = [(f, m) for f, m in logs if pattern_lower in f.lower()]
            if matched:
                deep_logs += f"\n[{pattern}] ({len(matched)} entries):\n"
                for fname, msg in matched[:20]:
                    deep_logs += f"  - {msg}\n"
            else:
                deep_logs += f"\n[{pattern}]: No matching file found.\n"

        if not deep_logs.strip():
            break

        messages.append({
            "role": "user",
            "content": (
                f"Here are the logs from the files you requested:\n{deep_logs}\n\n"
                "Analyze these and give deeper root cause analysis. "
                "If you need more files, reply with CHECK_NEXT: filename1, filename2\n"
                "If investigation is complete, reply with CHECK_NEXT: DONE"
            )
        })

        response = ask_llm(groq_client, messages)
        steps.append({"title": f"Deep Analysis (Iteration {iteration + 1})", "content": response})
        messages.append({"role": "assistant", "content": response})

    return steps


# =============================================================
# INTERACTIVE INVESTIGATION (for Web UI)
# =============================================================

def interactive_analyze_start(logs, groq_client, query="Analyze all errors and find root cause"):
    """Start an interactive investigation. Returns steps so far, messages history,
    and a list of files the AI wants the user to upload (if any).

    Returns:
        dict with keys:
            steps: list of analysis steps completed so far
            messages: LLM conversation history (for continuation)
            requested_files: list of filenames the AI wants next (empty if done)
    """
    error_summary, file_list = _build_initial_context(logs, query)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_INVESTIGATE},
        {"role": "user", "content": (
            f"User question: {query}\n\n"
            f"Initial error scan:{error_summary}\n\n"
            f"Available files:\n{file_list}"
        )}
    ]

    steps = []
    available_file_names = set(f.lower() for f, _ in logs)

    response = ask_llm(groq_client, messages)
    steps.append({"title": "Initial Diagnosis", "content": response})
    messages.append({"role": "assistant", "content": response})

    # Do up to 2 internal iterations with already-available files
    for iteration in range(2):
        requested = _parse_check_next(response)
        if not requested:
            # AI said DONE or didn't request anything
            return {"steps": steps, "messages": messages, "requested_files": []}

        # Split into available vs missing
        available_requests = []
        missing_requests = []
        for pattern in requested:
            pattern_lower = pattern.lower()
            matched = [(f, m) for f, m in logs if pattern_lower in f.lower()]
            if matched:
                available_requests.append((pattern, matched))
            else:
                missing_requests.append(pattern)

        # If some files are missing, stop and ask the user
        if missing_requests and not available_requests:
            # All requested files are missing — ask user for them
            return {"steps": steps, "messages": messages, "requested_files": missing_requests}

        # Process available files first
        if available_requests:
            deep_logs = ""
            for pattern, matched in available_requests:
                deep_logs += f"\n[{pattern}] ({len(matched)} entries):\n"
                for fname, msg in matched[:20]:
                    deep_logs += f"  - {msg}\n"

            if missing_requests:
                deep_logs += "\n\nNote: The following files were NOT available: "
                deep_logs += ", ".join(missing_requests)

            messages.append({
                "role": "user",
                "content": (
                    f"Here are the logs from the files you requested:\n{deep_logs}\n\n"
                    "Analyze these and give deeper root cause analysis. "
                    "If you need more files, reply with CHECK_NEXT: filename1, filename2\n"
                    "If investigation is complete, reply with CHECK_NEXT: DONE"
                )
            })

            response = ask_llm(groq_client, messages)
            steps.append({"title": f"Deep Analysis (Iteration {iteration + 1})", "content": response})
            messages.append({"role": "assistant", "content": response})

        # If we had missing files, ask user after processing available ones
        if missing_requests:
            return {"steps": steps, "messages": messages, "requested_files": missing_requests}

    # After 2 internal iterations, check if AI still wants more files
    requested = _parse_check_next(response)
    if requested:
        missing = [p for p in requested
                   if not any(p.lower() in f.lower() for f, _ in logs)]
        if missing:
            return {"steps": steps, "messages": messages, "requested_files": missing}

    return {"steps": steps, "messages": messages, "requested_files": []}


def interactive_analyze_continue(logs_new, groq_client, prev_messages, prev_steps):
    """Continue an interactive investigation with newly uploaded files.

    Args:
        logs_new: new log entries from the files the user just uploaded
        groq_client: Groq client
        prev_messages: conversation history from the previous round
        prev_steps: steps from the previous round

    Returns:
        dict with keys: steps, messages, requested_files (same as start)
    """
    messages = list(prev_messages)
    steps = list(prev_steps)

    # Build log content from new files
    new_file_names = sorted(set(f for f, _ in logs_new))
    deep_logs = ""
    for fname in new_file_names:
        matched = [(f, m) for f, m in logs_new if f == fname]
        deep_logs += f"\n[{fname}] ({len(matched)} entries):\n"
        for f, msg in matched[:25]:
            deep_logs += f"  - {msg}\n"

    messages.append({
        "role": "user",
        "content": (
            f"The user has provided the additional log files you requested.\n"
            f"Here are the new logs:\n{deep_logs}\n\n"
            "Continue your root cause investigation with this new data. "
            "If you need more files, reply with CHECK_NEXT: filename1, filename2\n"
            "If investigation is complete, reply with CHECK_NEXT: DONE"
        )
    })

    response = ask_llm(groq_client, messages, max_tokens=1000)
    step_num = len(steps) + 1
    steps.append({"title": f"Deep Analysis (after new files)", "content": response})
    messages.append({"role": "assistant", "content": response})

    # One more iteration if AI requests more files from existing new data
    requested = _parse_check_next(response)
    if requested:
        missing = [p for p in requested
                   if not any(p.lower() in f.lower() for f, _ in logs_new)]
        if missing:
            return {"steps": steps, "messages": messages, "requested_files": missing}

        # Requested files are in the new data — process them
        deep_logs2 = ""
        for pattern in requested:
            pattern_lower = pattern.lower()
            matched = [(f, m) for f, m in logs_new if pattern_lower in f.lower()]
            if matched:
                deep_logs2 += f"\n[{pattern}] ({len(matched)} entries):\n"
                for fname, msg in matched[:20]:
                    deep_logs2 += f"  - {msg}\n"

        if deep_logs2.strip():
            messages.append({
                "role": "user",
                "content": (
                    f"Here are the additional logs:\n{deep_logs2}\n\n"
                    "Continue analysis. Reply with CHECK_NEXT: DONE if complete, "
                    "or CHECK_NEXT: filename1, filename2 if you need more."
                )
            })
            response = ask_llm(groq_client, messages, max_tokens=1000)
            steps.append({"title": "Final Analysis", "content": response})
            messages.append({"role": "assistant", "content": response})

            requested2 = _parse_check_next(response)
            if requested2:
                return {"steps": steps, "messages": messages, "requested_files": requested2}

    return {"steps": steps, "messages": messages, "requested_files": []}
