"""Agent orchestration utilities."""

from __future__ import annotations

from typing import Iterable
import asyncio
import inspect

from dotenv import load_dotenv

from .ingest import DEFAULT_STORE_DIR, ensure_ingested
from .schema import Antenna
from .tools import list_bundle_files, read_file, retrieve_passages, summarize_figure

load_dotenv(override=True)

try:
    from agents import Agent, Runner, trace
except ImportError:  # pragma: no cover - provide friendly error when runtime missing
    Agent = Runner = trace = None  # type: ignore


def _build_instructions(pdf_path: str, store_dir: str) -> str:
    return f"""
ROLE
You are the Antenna Info Extractor. From a single PDF, build a complete Antenna object using the available tools. Be precise, minimize tool calls, and include provenance wherever possible. Do not guess; use null for unknowns.

TOOLS
- ensure_ingested(pdf_path={pdf_path!r}, store_dir={store_dir!r}): build or reuse the ingest store for this PDF (always call this first).
- list_bundle_files(bundle_dir, subdir="pages"): inspect available page or figure files with previews.
- retrieve_passages(bundle_dir, query, top_k): semantic search over cached text chunks. Use this before manual reading.
- read_file(path, start_line?, end_line?, max_chars?): read targeted portions of cached text.
- summarize_figure(fig_path, instruction?): ask the vision model to summarize relevant figure details in concise text.

STORE LAYOUT
- bundle/pages/page_0001.txt : textual pages (chunk with read_file).
- bundle/figures/fig_001.png : cropped figures.
- bundle/vector_store : persisted embeddings for retrieval.

OUTPUT
Return exactly one valid Antenna JSON (no extra prose). Follow the schema strictly:
- antenna_type (string)
- polarization (type and optional sense)
- operating_points (non-empty list of OperatingPoint)
- stackup (list of Layer from bottom order=0 upward)
- feed (optional)
- notes (optional)
Provide provenance for numeric/structural facts whenever possible.

UNITS AND CONVENTIONS
- Frequencies in GHz (convert from MHz if needed).
- Dimensions in mm.
- Gain in dBi.
- Axial ratio in dB.
- If circular polarization, set sense to RHCP or LHCP when stated; otherwise null.
- Leave fields null rather than writing "none" when unknown.

WORKFLOW
1. Call ensure_ingested to get the manifest (and vector store path).
2. Review manifest.warnings, figure counts, and bundle paths.
3. Use retrieve_passages for focused queries (antenna type, geometry, performance, feed, materials).
4. Use list_bundle_files to inspect available page files and figure references if needed.
5. Call read_file with explicit line spans or max_chars when drilling into text.
6. When a figure caption suggests geometry/performance, call summarize_figure to obtain a concise textual summary. Avoid repeated calls.
7. Populate the Antenna object, ensure operating_points is not empty, and attach provenance referencing page/figure files.
8. Return only the final Antenna JSON object (no explanations).

EFFICIENCY
- Prefer retrieval + targeted reads over full-page dumps.
- Limit figure summaries to the most relevant ones.
- Keep responses concise to avoid context overflows.
"""


def create_information_extraction_agent(pdf_path: str, store_dir: str = DEFAULT_STORE_DIR):
    if Agent is None:
        raise RuntimeError("openai-agents runtime is not installed")

    tools = [ensure_ingested, list_bundle_files, retrieve_passages, read_file, summarize_figure]
    instructions = _build_instructions(pdf_path, store_dir)
    return Agent(
        name="Antenna Info Extractor",
        instructions=instructions,
        tools=tools,
        model="gpt-4o-mini",
        output_type=Antenna,
    )


async def run_information_extraction_async(pdf_path: str, store_dir: str = DEFAULT_STORE_DIR, max_turns: int = 30):
    if Runner is None or trace is None:
        raise RuntimeError("openai-agents runtime is not installed")

    agent = create_information_extraction_agent(pdf_path, store_dir)
    with trace("Extract antenna info"):
        run_result = Runner.run(agent, f"Extract antenna info from {pdf_path}", max_turns=max_turns)
    if inspect.isawaitable(run_result):
        return await run_result
    return run_result


def run_information_extraction(pdf_path: str, store_dir: str = DEFAULT_STORE_DIR, max_turns: int = 30):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            run_information_extraction_async(pdf_path, store_dir, max_turns=max_turns)
        )
    else:
        return loop.create_task(
            run_information_extraction_async(pdf_path, store_dir, max_turns=max_turns)
        )


__all__: Iterable[str] = (
    "create_information_extraction_agent",
    "run_information_extraction",
    "run_information_extraction_async",
)
