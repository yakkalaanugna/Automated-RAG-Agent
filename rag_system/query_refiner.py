"""
query_refiner.py — LLM-Based Query Rewriting

Refines user queries before each retrieval iteration by injecting
domain-specific context: detected error codes, module names, error
messages, and findings from previous iterations.

This improves retrieval precision across iterations by making queries
progressively more specific and targeted.
"""

import re
from typing import Dict, List, Optional

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ─── Query Refiner ─────────────────────────────────────────────────────────────

class QueryRefiner:
    """
    LLM-based query rewriting for iterative retrieval improvement.

    Each iteration refines the search query using:
      - Detected error messages and codes from retrieved logs
      - Module / component names mentioned in results
      - Findings and gaps identified in the previous LLM analysis
      - The original user intent

    Example transformation:
      Initial:  "network failure"
      Refined:  "RRC Reconfiguration failure code 4 in rfma_impl.cpp UE4
                 triggering UE Context Release at 18:34:08"
    """

    REFINEMENT_PROMPT = ChatPromptTemplate.from_template(
        """You are a telecom log analysis expert. Your task is to refine a search query
to retrieve more relevant log entries from a vector database of telecom logs.

Original user query: {original_query}

Current iteration: {iteration} of {max_iterations}

Previous query used: {previous_query}

Key findings so far:
{findings_summary}

Detected error patterns:
{error_patterns}

Detected modules/components:
{modules}

Instructions:
1. Make the query MORE SPECIFIC by incorporating discovered error codes, module names, and timestamps.
2. Focus on aspects NOT yet covered in the findings.
3. If the root cause is partially identified, query for CORROBORATING evidence.
4. Include telecom-specific terminology (RRC, NGAP, UE, gNB, CU-CP, etc.).
5. Keep the query concise (under 50 words) but information-dense.

Return ONLY the refined query text, nothing else."""
    )

    def __init__(self, llm: ChatGroq):
        self.llm = llm
        self.chain = self.REFINEMENT_PROMPT | llm | StrOutputParser()

    def refine(
        self,
        original_query: str,
        previous_query: str,
        findings_summary: str,
        retrieved_docs_text: str,
        iteration: int,
        max_iterations: int,
    ) -> str:
        """
        Generate a refined query based on current context.

        Args:
            original_query:     The user's original question
            previous_query:     Query used in the last iteration
            findings_summary:   Summary of LLM analysis so far
            retrieved_docs_text: Raw text of retrieved documents
            iteration:          Current iteration number (1-based)
            max_iterations:     Maximum iterations allowed

        Returns:
            Refined query string for the next retrieval pass.
        """
        error_patterns = self._extract_error_patterns(retrieved_docs_text)
        modules = self._extract_modules(retrieved_docs_text)

        refined = self.chain.invoke({
            "original_query": original_query,
            "previous_query": previous_query,
            "findings_summary": findings_summary if findings_summary else "No findings yet (first iteration).",
            "error_patterns": error_patterns if error_patterns else "None detected yet.",
            "modules": modules if modules else "None detected yet.",
            "iteration": iteration,
            "max_iterations": max_iterations,
        })

        return refined.strip()

    # ── Pattern extraction helpers ─────────────────────────────────────────

    @staticmethod
    def _extract_error_patterns(text: str) -> str:
        """Extract error codes, error messages, and failure indicators."""
        patterns = set()

        # Error codes
        for m in re.finditer(r'(?:code|cause)\s*[=:]?\s*(\d+)', text, re.IGNORECASE):
            patterns.add(f"error code {m.group(1)}")

        # NGAP IDs
        for m in re.finditer(r'(AMF_UE_NGAP_ID|RAN_UE_NGAP_ID)\s*[=:]?\s*(\d+)', text):
            patterns.add(f"{m.group(1)}={m.group(2)}")

        # UE identifiers
        for m in re.finditer(r'(UE\d+|ueIdCu:\d+)', text):
            patterns.add(m.group(0))

        # Error keywords with context
        for m in re.finditer(
            r'((?:RRC\s*)?(?:Reconfiguration|Release|Reject)\s*(?:failure|error)?)',
            text, re.IGNORECASE
        ):
            patterns.add(m.group(0).strip())

        # Forward jumps (data loss indicator)
        for m in re.finditer(r'forward jump.*?delta=(\d+)', text, re.IGNORECASE):
            patterns.add(f"forward jump delta={m.group(1)}")

        return "; ".join(sorted(patterns)) if patterns else ""

    @staticmethod
    def _extract_modules(text: str) -> str:
        """Extract module and component names from log text."""
        modules = set()

        for m in re.finditer(r'(\w+\.cpp)\[(\d+)\]', text):
            modules.add(m.group(1))

        for m in re.finditer(r'(?:INF|DBG|ERR)/(\S+)/(\w+\.cpp)', text):
            modules.add(f"{m.group(1)}/{m.group(2)}")

        for m in re.finditer(r'(UEC-\d+|AMF-\d+|GNB-\d+)', text):
            modules.add(m.group(0))

        for m in re.finditer(r'(CU-CP|CU-UP|DU|RU|cu_cp|cu_up)', text, re.IGNORECASE):
            modules.add(m.group(0))

        return ", ".join(sorted(modules)) if modules else ""

    @staticmethod
    def extract_key_findings(analysis_text: str) -> str:
        """
        Extract a condensed summary of key findings from an LLM analysis.
        Used to feed into the next iteration's query refinement.
        """
        lines = analysis_text.split("\n")
        findings = []

        # Extract root cause line
        for i, line in enumerate(lines):
            if "root cause" in line.lower() and i + 1 < len(lines):
                findings.append(lines[i + 1].strip("# -"))
                break

        # Extract severity
        for line in lines:
            if any(sev in line.upper() for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]):
                findings.append(line.strip("# -"))
                break

        # Extract any timestamps mentioned
        timestamps = re.findall(r'\d{2}:\d{2}:\d{2}[\.\:]\d+', analysis_text)
        if timestamps:
            findings.append(f"Key timestamps: {', '.join(set(timestamps[:5]))}")

        # Extract mentioned UE/gNB identifiers
        ues = set(re.findall(r'UE\d+', analysis_text))
        if ues:
            findings.append(f"Affected UEs: {', '.join(sorted(ues))}")

        return " | ".join(findings) if findings else "No specific findings extracted."
