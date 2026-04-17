"""
parser.py — Structured Telecom Log Parser

Parses raw telecom log files (eGate, UEC, RAIN, Robot, syslog, etc.)
into structured records with extracted metadata fields:
  timestamp, log_level, module/component, error_code, message

Supports plain-text logs and archive extraction (.tgz, .tar.gz, .zip).
"""

import os
import re
import io
import tarfile
import zipfile
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple


# ─── Structured log record ─────────────────────────────────────────────────────

@dataclass
class LogRecord:
    """A single parsed log entry with structured metadata."""
    source: str            # filename / relative path
    line_number: int       # 1-based line number in file
    timestamp: str         # extracted timestamp string
    log_level: str         # ERROR, FAIL, WARNING, INFO
    module: str            # component / module name
    error_code: str        # extracted error code (if any)
    message: str           # cleaned full message text
    raw_text: str          # original unmodified line

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def display(self) -> str:
        ts = f"@{self.timestamp}" if self.timestamp else ""
        return f"[{self.log_level:7s}] [{self.source}:L{self.line_number}] {ts} {self.message}"


# ─── Pre-compiled regex patterns ───────────────────────────────────────────────

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')

# Timestamp extraction — ordered by specificity
_TS_PATTERNS = [
    ("egate",   re.compile(r'^(\d{2}:\d{2}:\d{2}\.\d+)\s+\d{2}:\d{2}:\d{2}\.\d+\s+')),
    ("uec",     re.compile(r"^[de]\s+(\d{2}:\d{2}:\d{2}'\d{3}\"\d{3})")),
    ("uec_err", re.compile(r"^ERROR!!\s+(\d{2}:\d{2}:\d{2}'\d{3}\"\d{3})")),
    ("robot",   re.compile(r'^(\d{8}\s+\d{2}:\d{2}:\d{2}\.\d+)')),
    ("generic", re.compile(r'^(\d{2}:\d{2}:\d{2}[\.\:]\d+)')),
    ("rain",    re.compile(r'<(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)Z?>')),
    ("iso",     re.compile(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})')),
]

# Module / component extraction
_MODULE_PATTERNS = [
    re.compile(r'(\w+\.cpp)\[(\d+)\]'),           # C++ source:line
    re.compile(r'INF/(\S+)/(\w+\.cpp)'),           # rain-style: INF/component/file.cpp
    re.compile(r'DBG/(\S+)/(\w+\.cpp)'),           # rain-style: DBG/component/file.cpp
    re.compile(r'(UEC-\d+):\s+(UE\d+)'),           # UEC controller + UE ID
    re.compile(r'(AMF-\d+):\s+(GNB-\d+)'),         # AMF + gNB
    re.compile(r'(GNB-\d+)'),                       # gNB ID
    re.compile(r'\[ueIdCu:(\d+)\]'),               # CU UE ID
]

# Error code extraction
_ERROR_CODE_RE = re.compile(
    r'(?:code\s*[=:]?\s*(\d+))|'
    r'(?:AMF_UE_NGAP_ID\s*[=:]?\s*(\d+))|'
    r'(?:RAN_UE_NGAP_ID\s*[=:]?\s*(\d+))|'
    r'(?:cause\s*[=:]?\s*(\w+))',
    re.IGNORECASE,
)

# Importance filter — telecom-domain keywords
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

# Severity classification
_SEV_ERROR_RE = re.compile(
    r'error|data loss|timeout|critical|fatal|segfault|core dump|unreachable|abort|panic|'
    r'forward jump|packets lost|rrc reconfiguration.*failure|ctrl_del_ue',
    re.IGNORECASE,
)
_SEV_FAIL_RE = re.compile(r'fail|nok|reject', re.IGNORECASE)
_SEV_WARN_RE = re.compile(
    r'warn|latency|delay|congestion|degraded|retry|overload|retx|retransmission',
    re.IGNORECASE,
)

# Supported file types
SUPPORTED_EXTENSIONS = (".txt", ".log", ".json", ".cfg", ".csv", ".xml", ".html", ".htm")
ARCHIVE_EXTENSIONS = (".tgz", ".tar.gz", ".zip")
ARCHIVE_PATTERNS = [
    "syslog", "messages", "dmesg", "kern.log", "daemon.log",
    "worker", "egate", "alarm", "error", "uec_1", "uec_2",
    "btslog", "rain", "runtime", "gnb", "enb", "cu_cp", "cu_up", "firewall",
    "log.html", "e2e_console", "cpu_utilization", "PacketReceiver",
]


# ─── TelecomLogParser ──────────────────────────────────────────────────────────

class TelecomLogParser:
    """
    Structured parser for telecom log files.

    Extracts timestamp, log_level, module, error_code, and message
    from multiple telecom log formats.  Returns a list of LogRecord
    objects suitable for downstream embedding and retrieval.
    """

    def __init__(self, filter_important: bool = True, min_line_length: int = 10):
        self.filter_important = filter_important
        self.min_line_length = min_line_length

    # ── Public API ─────────────────────────────────────────────────────────

    def parse_file(self, filepath: str) -> List[LogRecord]:
        """Parse a single log file from disk."""
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
            content = fh.read()
        return self._parse_content(content, filename)

    def parse_text(self, content: str, filename: str) -> List[LogRecord]:
        """Parse log content provided as a string."""
        return self._parse_content(content, filename)

    def parse_bytes(self, data: bytes, filename: str) -> List[LogRecord]:
        """Parse raw bytes (auto-detects archives)."""
        if filename.lower().endswith(ARCHIVE_EXTENSIONS):
            return self._parse_archive_bytes(data, filename)
        text = data.decode("utf-8", errors="ignore")
        return self._parse_content(text, filename)

    def parse_folder(self, folder: str) -> List[LogRecord]:
        """Recursively parse all supported files in a folder."""
        records: List[LogRecord] = []
        for root, _, files in os.walk(folder):
            for f in sorted(files):
                fp = os.path.join(root, f)
                rel = os.path.relpath(fp, folder)
                if f.lower().endswith(SUPPORTED_EXTENSIONS):
                    with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                        records.extend(self._parse_content(fh.read(), rel))
                elif f.lower().endswith(ARCHIVE_EXTENSIONS):
                    with open(fp, "rb") as fh:
                        records.extend(self._parse_archive_bytes(fh.read(), rel))
        return records

    def parse_all_lines(self, content: str, filename: str) -> List[LogRecord]:
        """Parse every line (no importance filter) — useful for comparison."""
        records: List[LogRecord] = []
        for line_no, line in enumerate(content.splitlines(), start=1):
            cleaned = self._clean(line)
            if cleaned and len(cleaned) >= 5:
                records.append(self._build_record(filename, line_no, cleaned, line))
        return records

    # ── Internal helpers ───────────────────────────────────────────────────

    def _clean(self, line: str) -> str:
        return _ANSI_RE.sub('', line).strip()

    def _extract_timestamp(self, line: str) -> str:
        for _, pattern in _TS_PATTERNS:
            m = pattern.search(line)
            if m:
                return m.group(1)
        return ""

    def _extract_module(self, line: str) -> str:
        for pattern in _MODULE_PATTERNS:
            m = pattern.search(line)
            if m:
                return m.group(0)
        return ""

    def _extract_error_code(self, line: str) -> str:
        m = _ERROR_CODE_RE.search(line)
        if m:
            # Return first non-None group
            for g in m.groups():
                if g:
                    return g
        return ""

    def _detect_severity(self, msg: str) -> str:
        if _SEV_ERROR_RE.search(msg):
            return "ERROR"
        if _SEV_FAIL_RE.search(msg):
            return "FAIL"
        if _SEV_WARN_RE.search(msg):
            return "WARNING"
        return "INFO"

    def _is_important(self, line: str) -> bool:
        return bool(_IMPORTANT_RE.search(line))

    def _build_record(self, filename: str, line_no: int,
                      cleaned: str, raw: str) -> LogRecord:
        return LogRecord(
            source=filename,
            line_number=line_no,
            timestamp=self._extract_timestamp(cleaned),
            log_level=self._detect_severity(cleaned),
            module=self._extract_module(cleaned),
            error_code=self._extract_error_code(cleaned),
            message=cleaned,
            raw_text=raw,
        )

    def _parse_content(self, content: str, filename: str) -> List[LogRecord]:
        seen: set = set()
        records: List[LogRecord] = []
        for line_no, line in enumerate(content.splitlines(), start=1):
            cleaned = self._clean(line)
            if not cleaned or len(cleaned) < self.min_line_length:
                continue
            if self.filter_important and not self._is_important(cleaned):
                continue
            if cleaned in seen:
                continue
            seen.add(cleaned)
            records.append(self._build_record(filename, line_no, cleaned, line))
        return records

    def _parse_archive_bytes(self, data: bytes, name: str) -> List[LogRecord]:
        records: List[LogRecord] = []
        try:
            if name.lower().endswith((".tgz", ".tar.gz")):
                with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
                    for m in tar.getmembers():
                        if not m.isfile() or m.size > 50 * 1024 * 1024:
                            continue
                        if not any(p in m.name.lower() for p in ARCHIVE_PATTERNS):
                            continue
                        f = tar.extractfile(m)
                        if f is None:
                            continue
                        text = f.read().decode("utf-8", errors="ignore")
                        inner_name = f"{name}/{os.path.basename(m.name)}"
                        records.extend(self._parse_content(text, inner_name))
            elif name.lower().endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
                    for entry in zf.namelist():
                        if not any(p in entry.lower() for p in ARCHIVE_PATTERNS):
                            continue
                        if zf.getinfo(entry).file_size > 50 * 1024 * 1024:
                            continue
                        text = zf.read(entry).decode("utf-8", errors="ignore")
                        inner_name = f"{name}/{os.path.basename(entry)}"
                        records.extend(self._parse_content(text, inner_name))
        except Exception as e:
            print(f"  Archive error {name}: {e}")
        return records

    # ── Utility ────────────────────────────────────────────────────────────

    @staticmethod
    def normalize_timestamp(ts: str) -> str:
        """Normalize timestamp to HH:MM:SS for cross-file correlation."""
        if not ts:
            return ""
        m = re.match(r'(\d{2}:\d{2}:\d{2})', ts)
        if m:
            return m.group(1)
        m = re.search(r'T(\d{2}:\d{2}:\d{2})', ts)
        if m:
            return m.group(1)
        return ts[:8] if len(ts) >= 8 else ts

    @staticmethod
    def severity_counts(records: List[LogRecord]) -> dict:
        """Count records by severity level."""
        counts = {"ERROR": 0, "FAIL": 0, "WARNING": 0, "INFO": 0}
        for r in records:
            counts[r.log_level] = counts.get(r.log_level, 0) + 1
        return counts
