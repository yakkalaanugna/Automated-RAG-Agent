"""
Streamlit Web UI for Automated Root Cause Analysis using RAG + Agent.
Features: Log upload, auto-analysis, pass vs fail comparison, log type detection.
"""

import streamlit as st
import os
import tempfile
from engine import (
    get_groq_client, read_log_content, read_all_lines, extract_archive_bytes,
    detect_log_type, detect_severity, is_important, clean_line,
    compare_pass_fail, analyze_logs, ask_llm,
    interactive_analyze_start, interactive_analyze_continue,
    scan_local_path, scan_path_for_files, list_local_path_files,
    SUPPORTED_EXTENSIONS, ARCHIVE_EXTENSIONS,
    SYSTEM_PROMPT_ANALYZE, LOG_TYPE_PATTERNS
)

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="AutoRCA — Telecom Log Analyzer",
    page_icon="🔍",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: bold; color: #1E88E5; margin-bottom: 0; }
    .sub-header { font-size: 1rem; color: #666; margin-top: 0; }
    .severity-error { color: #D32F2F; font-weight: bold; }
    .severity-warning { color: #F57C00; font-weight: bold; }
    .severity-info { color: #1976D2; }
    .stat-box { background: #f0f2f6; border-radius: 10px; padding: 20px; text-align: center; }
    .stat-number { font-size: 2rem; font-weight: bold; color: #1E88E5; }
    .stat-label { font-size: 0.9rem; color: #666; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown('<p class="main-header">AutoRCA — Automated Root Cause Analysis</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Telecom Automated Testing Log Analyzer | RAG + LLM Agent</p>', unsafe_allow_html=True)
st.divider()

# -----------------------------
# Sidebar — API key
# -----------------------------
with st.sidebar:
    st.header("Settings")
    api_key_input = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Get your free key from https://console.groq.com/keys"
    )
    if api_key_input:
        os.environ["GROQ_API_KEY"] = api_key_input

    st.divider()
    st.markdown("**Supported file types:**")
    st.markdown("`.txt` `.log` `.json` `.csv` `.xml` `.html` `.cfg` `.tgz` `.tar.gz` `.zip`")

    st.divider()
    st.markdown("**How it works:**")
    st.markdown("""
    1. Upload log file(s) or paste a local path
    2. System auto-detects log format
    3. AI agent investigates errors
    4. AI asks for more files if needed
    5. Provide files via upload or path
    6. Get root cause analysis
    """)
    st.divider()
    st.markdown("**💡 Tip for large files (sosreport):**")
    st.markdown(
        "Use the **Local path** option — paste the path to your sosreport folder. "
        "The system reads it directly from disk, no upload needed."
    )

# Check API key
groq_client = get_groq_client()
if not groq_client:
    st.warning("Please enter your Groq API key in the sidebar to enable AI analysis.")


# =============================================================
# Helper functions
# =============================================================

def process_uploaded_file(uploaded_file):
    """Process an uploaded file and return parsed log entries."""
    filename = uploaded_file.name
    file_bytes = uploaded_file.read()

    # Handle archives
    if filename.lower().endswith(ARCHIVE_EXTENSIONS):
        entries = extract_archive_bytes(file_bytes, filename)
        log_type = "archive"
        return entries, log_type, file_bytes

    # Handle text-based files
    try:
        content = file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        content = file_bytes.decode("latin-1", errors="ignore")

    entries, log_type = read_log_content(content, filename)
    return entries, log_type, content


def show_severity_badge(severity):
    """Return colored severity badge."""
    colors = {
        "ERROR": "🔴", "FAIL": "🔴",
        "WARNING": "🟡", "INFO": "🔵"
    }
    return colors.get(severity, "⚪")


# =============================================================
# TABS
# =============================================================
tab1, tab2, tab3 = st.tabs([
    "📊 Single Log Analysis",
    "🔄 Pass vs Fail Comparison",
    "📁 Batch Analysis"
])


# =============================================================
# TAB 1: Single Log Analysis (Interactive Investigation)
# =============================================================
with tab1:
    st.subheader("Upload a log file for AI-powered root cause analysis")

    # --- Input method: Upload OR Local Path ---
    input_method = st.radio(
        "How do you want to provide log files?",
        ["📤 Upload file (small logs)", "📂 Local/network path (sosreport, large dirs)"],
        key="input_method",
        horizontal=True
    )

    uploaded_file = None
    local_path = None

    if input_method == "📤 Upload file (small logs)":
        uploaded_file = st.file_uploader(
            "Drop your log file here",
            type=["txt", "log", "json", "csv", "xml", "html", "htm", "cfg", "tgz", "gz", "zip"],
            key="single_upload",
            help="Supports text logs, JSON, CSV, XML, HTML, archives (.tgz, .zip)"
        )
    else:
        local_path = st.text_input(
            "Enter the full path to the log file or directory",
            placeholder=r"e.g., C:\logs\sosreport-xyz  or  /home/user/testrun/logs",
            key="local_path_input",
            help="Paste the path to a sosreport folder, log directory, or a single log file. "
                 "The system will scan it and find important log files automatically."
        )
        if local_path and os.path.isdir(local_path):
            with st.expander("Preview files in this directory"):
                file_list = list_local_path_files(local_path, max_files=100)
                if not file_list:
                    st.warning("No files found in this directory.")
                else:
                    important_files = [f for f in file_list if f["is_important"]]
                    other_files = [f for f in file_list if not f["is_important"]]
                    if important_files:
                        st.markdown(f"**Important files found ({len(important_files)}):**")
                        for f in important_files[:50]:
                            st.text(f"  ⭐ {f['rel_path']}  ({f['size_mb']} MB)")
                    if other_files:
                        st.markdown(f"**Other files ({len(other_files)}):**")
                        for f in other_files[:30]:
                            st.text(f"     {f['rel_path']}  ({f['size_mb']} MB)")
                    total_size = sum(f["size_mb"] for f in file_list)
                    st.caption(f"Total: {len(file_list)} files, {total_size:.1f} MB scanned")

    custom_query = st.text_input(
        "Custom question (optional — leave empty for auto-analysis)",
        placeholder="e.g., Why did the UE connection fail?",
        key="single_query"
    )

    # Initialize session state for interactive investigation
    if "investigation" not in st.session_state:
        st.session_state.investigation = None
    if "base_path" not in st.session_state:
        st.session_state.base_path = None

    has_input = uploaded_file or (local_path and os.path.exists(local_path))

    if has_input and st.button("Analyze", key="btn_analyze", type="primary"):
        if not groq_client:
            st.error("Please enter your Groq API key in the sidebar.")
        else:
            entries = []
            log_type = "generic"
            source_name = ""

            if uploaded_file:
                with st.spinner("Processing uploaded file..."):
                    entries_raw, log_type, raw_content = process_uploaded_file(uploaded_file)
                    entries = entries_raw
                    source_name = uploaded_file.name
                st.session_state.base_path = None
            elif local_path:
                with st.spinner(f"Scanning {local_path} for important log files..."):
                    entries, file_index = scan_local_path(local_path)
                    source_name = os.path.basename(local_path) or local_path
                    if os.path.isfile(local_path):
                        log_type = detect_log_type(
                            os.path.basename(local_path),
                            open(local_path, "r", errors="ignore").read(2000)
                            if os.path.getsize(local_path) < 100*1024*1024 else ""
                        )
                    else:
                        log_type = "directory"
                # Store the path so follow-up can scan it for requested files
                st.session_state.base_path = local_path

            # Show file info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Source", source_name)
            with col2:
                if log_type == "directory":
                    st.metric("Type", "Directory Scan")
                else:
                    type_desc = LOG_TYPE_PATTERNS.get(log_type, {}).get("description", log_type.replace("_", " ").title())
                    st.metric("Detected Type", type_desc)
            with col3:
                st.metric("Filtered Entries", len(entries))

            if not entries:
                st.warning("No important entries found. The file/directory may not contain error/warning lines.")
            else:
                # Show severity breakdown
                severity_counts = {"ERROR": 0, "FAIL": 0, "WARNING": 0, "INFO": 0}
                for _, msg in entries:
                    sev = detect_severity(msg)
                    severity_counts[sev] = severity_counts.get(sev, 0) + 1

                st.divider()
                cols = st.columns(4)
                for i, (sev, count) in enumerate(severity_counts.items()):
                    with cols[i]:
                        st.metric(f"{show_severity_badge(sev)} {sev}", count)

                # Show sample entries
                with st.expander("Preview filtered log entries"):
                    for fname, msg in entries[:50]:
                        sev = detect_severity(msg)
                        st.text(f"{show_severity_badge(sev)} [{fname}] {msg}")

                # Run interactive LLM analysis
                st.divider()
                st.subheader("🔍 AI Investigation")

                query = custom_query if custom_query else "Analyze all errors and find root cause"

                with st.spinner("AI agent is investigating..."):
                    result = interactive_analyze_start(entries, groq_client, query)

                # Store in session state for continuation
                st.session_state.investigation = result

                # Display completed steps
                for i, step in enumerate(result["steps"]):
                    with st.expander(f"Step {i + 1}: {step['title']}", expanded=(i == len(result["steps"]) - 1)):
                        st.markdown(step["content"])

                # If AI requests more files, show the request
                if result["requested_files"]:
                    st.divider()
                    st.warning(
                        "🔎 **The AI agent needs additional log files to continue the investigation.**\n\n"
                        "Provide the files below using upload or a local path:"
                    )
                    for fname in result["requested_files"]:
                        st.markdown(f"- **{fname}**")

    # --- Continuation: Provide requested files ---
    if (st.session_state.investigation
            and st.session_state.investigation.get("requested_files")):
        inv = st.session_state.investigation

        st.divider()
        st.subheader("📂 Provide Requested Files to Continue Investigation")
        st.info(
            "The AI agent requested: **"
            + ", ".join(inv["requested_files"])
            + "**"
        )

        followup_method = st.radio(
            "How to provide the requested files?",
            ["📤 Upload files", "📂 Enter local path (sosreport / log directory)"],
            key="followup_method",
            horizontal=True
        )

        followup_files = None
        followup_path = None

        if followup_method == "📤 Upload files":
            followup_files = st.file_uploader(
                "Upload the requested log file(s)",
                type=["txt", "log", "json", "csv", "xml", "html", "htm", "cfg", "tgz", "gz", "zip"],
                accept_multiple_files=True,
                key="followup_upload"
            )
        else:
            default_path = st.session_state.base_path or ""
            followup_path = st.text_input(
                "Enter path to directory containing the requested files",
                value=default_path,
                placeholder=r"e.g., C:\logs\sosreport-xyz",
                key="followup_path_input",
                help="The system will search this directory for the files the AI requested "
                     f"({', '.join(inv['requested_files'])})"
            )

        has_followup = followup_files or (followup_path and os.path.exists(followup_path))

        if has_followup and st.button("Continue Investigation", key="btn_continue", type="primary"):
            if not groq_client:
                st.error("Please enter your Groq API key in the sidebar.")
            else:
                new_entries = []

                if followup_files:
                    for f in followup_files:
                        file_entries, log_type, _ = process_uploaded_file(f)
                        new_entries.extend(file_entries)
                elif followup_path:
                    with st.spinner(f"Searching {followup_path} for requested files..."):
                        new_entries = scan_path_for_files(
                            followup_path, inv["requested_files"]
                        )
                    if new_entries:
                        found_files = sorted(set(f for f, _ in new_entries))
                        st.success(f"Found {len(found_files)} matching file(s): {', '.join(found_files)}")

                if not new_entries:
                    st.warning(
                        "No matching entries found. Make sure the path contains the requested files "
                        f"({', '.join(inv['requested_files'])})."
                    )
                else:
                    with st.spinner("AI agent is continuing the investigation with new files..."):
                        result = interactive_analyze_continue(
                            new_entries, groq_client,
                            inv["messages"], inv["steps"]
                        )

                    st.session_state.investigation = result

                    # Display ALL steps (including previous)
                    for i, step in enumerate(result["steps"]):
                        with st.expander(
                            f"Step {i + 1}: {step['title']}",
                            expanded=(i == len(result["steps"]) - 1)
                        ):
                            st.markdown(step["content"])

                    # Check if AI needs even more files
                    if result["requested_files"]:
                        st.divider()
                        st.warning(
                            "🔎 **The AI agent needs more log files to continue.**\n\n"
                            "Provide these files:"
                        )
                        for fname in result["requested_files"]:
                            st.markdown(f"- **{fname}**")
                    else:
                        st.success("✅ Investigation complete!")
                        st.session_state.investigation = None


# =============================================================
# TAB 2: Pass vs Fail Comparison
# =============================================================
with tab2:
    st.subheader("Compare PASS and FAIL logs to find the real failure")
    st.markdown("""
    Upload a log from a **passing** test and a **failing** test.
    The AI will compare them and identify errors **unique to the fail log** — ignoring common noise.
    """)

    col_pass, col_fail = st.columns(2)

    with col_pass:
        st.markdown("**PASS Log** (from a successful test run)")
        pass_file = st.file_uploader(
            "Upload PASS log",
            type=["txt", "log", "json", "csv", "xml", "html", "htm", "cfg", "tgz", "gz", "zip"],
            key="pass_upload"
        )

    with col_fail:
        st.markdown("**FAIL Log** (from a failed test run)")
        fail_file = st.file_uploader(
            "Upload FAIL log",
            type=["txt", "log", "json", "csv", "xml", "html", "htm", "cfg", "tgz", "gz", "zip"],
            key="fail_upload"
        )

    if pass_file and fail_file and st.button("Compare", key="btn_compare", type="primary"):
        if not groq_client:
            st.error("Please enter your Groq API key in the sidebar.")
        else:
            with st.spinner("Processing and comparing logs..."):
                # Read raw content
                pass_bytes = pass_file.read()
                fail_bytes = fail_file.read()

                try:
                    pass_content = pass_bytes.decode("utf-8", errors="ignore")
                    fail_content = fail_bytes.decode("utf-8", errors="ignore")
                except Exception:
                    pass_content = pass_bytes.decode("latin-1", errors="ignore")
                    fail_content = fail_bytes.decode("latin-1", errors="ignore")

                # Detect types
                pass_type = detect_log_type(pass_file.name, pass_content[:2000])
                fail_type = detect_log_type(fail_file.name, fail_content[:2000])

                # Run comparison
                result = compare_pass_fail(
                    pass_content, fail_content,
                    pass_file.name, fail_file.name,
                    groq_client
                )

            # Show stats
            st.divider()
            stats = result["stats"]

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("PASS Lines", stats["total_pass_lines"])
            with col2:
                st.metric("FAIL Lines", stats["total_fail_lines"])
            with col3:
                st.metric("Common Errors", stats["common_errors"], help="Errors in BOTH logs — safe to ignore")
            with col4:
                st.metric("FAIL-Only Errors", stats["fail_only_errors"], help="Errors ONLY in the fail log — these caused the failure")
            with col5:
                st.metric("Unique FAIL Lines", stats["fail_only_all"], help="All lines only in the fail log")

            # Show common (ignorable) errors
            if result["common_errors"]:
                with st.expander(f"Ignorable Errors ({len(result['common_errors'])} — present in BOTH logs)"):
                    for line in result["common_errors"][:30]:
                        st.text(f"⚪ {line}")

            # Show fail-only errors
            if result["fail_only_errors"]:
                with st.expander(f"Critical Errors ({len(result['fail_only_errors'])} — ONLY in FAIL log)", expanded=True):
                    for line in result["fail_only_errors"][:30]:
                        st.text(f"🔴 {line}")

            # Show AI analysis
            st.divider()
            st.subheader("AI Comparison Analysis")
            st.markdown(result["analysis"])


# =============================================================
# TAB 3: Batch Analysis (multiple files)
# =============================================================
with tab3:
    st.subheader("Upload multiple log files for batch analysis")

    batch_files = st.file_uploader(
        "Drop multiple log files here",
        type=["txt", "log", "json", "csv", "xml", "html", "htm", "cfg", "tgz", "gz", "zip"],
        accept_multiple_files=True,
        key="batch_upload"
    )

    batch_query = st.text_input(
        "Custom question (optional)",
        placeholder="e.g., What caused the test failure?",
        key="batch_query"
    )

    if batch_files and st.button("Analyze All", key="btn_batch", type="primary"):
        if not groq_client:
            st.error("Please enter your Groq API key in the sidebar.")
        else:
            all_entries = []
            file_info = []

            progress = st.progress(0, text="Processing files...")

            for i, f in enumerate(batch_files):
                progress.progress((i + 1) / len(batch_files), text=f"Processing {f.name}...")
                entries, log_type, _ = process_uploaded_file(f)
                type_desc = LOG_TYPE_PATTERNS.get(log_type, {}).get("description", log_type)
                file_info.append({
                    "file": f.name,
                    "type": type_desc,
                    "entries": len(entries)
                })
                all_entries.extend(entries)

            progress.empty()

            # Show file summary table
            st.divider()
            st.markdown("**Files processed:**")

            for info in file_info:
                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    st.text(info["file"])
                with col2:
                    st.text(info["type"])
                with col3:
                    st.text(f"{info['entries']} entries")

            st.metric("Total Entries", len(all_entries))

            if not all_entries:
                st.warning("No important entries found in the uploaded files.")
            else:
                # Severity breakdown
                severity_counts = {"ERROR": 0, "FAIL": 0, "WARNING": 0, "INFO": 0}
                for _, msg in all_entries:
                    sev = detect_severity(msg)
                    severity_counts[sev] = severity_counts.get(sev, 0) + 1

                cols = st.columns(4)
                for i, (sev, count) in enumerate(severity_counts.items()):
                    with cols[i]:
                        st.metric(f"{show_severity_badge(sev)} {sev}", count)

                # Run analysis
                st.divider()
                st.subheader("AI Investigation")
                query = batch_query if batch_query else "Analyze all errors and find root cause"

                with st.spinner("AI agent is investigating all files..."):
                    steps = analyze_logs(all_entries, groq_client, query)

                for i, step in enumerate(steps):
                    with st.expander(f"Step {i + 1}: {step['title']}", expanded=(i == len(steps) - 1)):
                        st.markdown(step["content"])


# -----------------------------
# Footer
# -----------------------------
st.divider()
st.markdown(
    '<p style="text-align:center;color:#999;font-size:0.8rem;">'
    'AutoRCA — Automated Root Cause Analysis using RAG + Agent | '
    'Powered by Groq LLM + FAISS + Sentence Transformers'
    '</p>',
    unsafe_allow_html=True
)
