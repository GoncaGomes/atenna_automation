"""Agent orchestration utilities."""

from __future__ import annotations

from typing import Iterable, Any, Optional, Tuple
import asyncio
import inspect
import json
from pathlib import Path

from dotenv import load_dotenv

from .ingest import DEFAULT_STORE_DIR, ensure_ingested
from .schema import Antenna
from .tools import function_tool, list_bundle_files, read_file, retrieve_passages, summarize_figure

DEFAULT_OUTPUT_FILENAME = "antenna_output.json"


load_dotenv(override=True)

ensure_ingested_tool = function_tool(ensure_ingested, name_override="ensure_ingested")


try:
    from agents import Agent, Runner, trace
except ImportError:  # pragma: no cover - provide friendly error when runtime missing
    Agent = Runner = trace = None  # type: ignore


def _build_instructions(pdf_path: str, store_dir: str) -> str:
    return f"""
ROLE
You are the Antenna Info Extractor. From a single PDF, build a complete Antenna object using the available tools. Be precise, minimize tool calls, ground every fact in the article, and include provenance. Do not guess; use null for unknowns.

TOOLS
The available tools are: ensure_ingested(pdf_path={pdf_path!r}, store_dir={store_dir!r}), which builds or reuses the ingest bundle and must be called first; list_bundle_files(bundle_dir, subdir="pages"), which lets you inspect page or figure files with short previews; retrieve_passages(bundle_dir, query, top_k), which performs semantic search over cached text chunks and should be used to locate candidates before any manual reading; read_file(path, start_line?, end_line?, max_chars?), which reads targeted portions of cached text to confirm details and capture precise quotes; summarize_figure(fig_path, instruction?), which uses a vision model to summarize figure details and should be invoked only when text evidence is insufficient or when you need to confirm geometry inferred from text.

STORE LAYOUT
The bundle contains textual pages under bundle/pages/page_0001.txt, cropped figures under bundle/figures/fig_001.png, and a persisted vector store under bundle/vector_store for retrieval.

OUTPUT
Return exactly one valid Antenna JSON and no extra prose. The object must contain antenna_type, polarization (with optional sense), a non-empty operating_points list, a stackup described from bottom to top (order=0 is the bottom layer), and optional feed and notes fields. Provide provenance for numeric and structural facts wherever possible.

UNITS AND CONVENTIONS
Report frequencies in GHz, dimensions in mm, gain in dBi, and axial ratio in dB. If polarization is circular, set the sense to RHCP or LHCP when stated; otherwise leave it null. Leave fields null rather than writing "none" when information is missing.

WORKFLOW
Begin by calling ensure_ingested to obtain the manifest and bundle paths. Review warnings, figure counts, and directories in the manifest. Use retrieve_passages to surface likely text locations for antenna type, geometry, materials, performance, and feed. Read the exact evidence with read_file by specifying explicit line spans or a character cap, and capture short quotes for provenance. Only when the relevant geometry cannot be confirmed from text or when you must validate a textual inference should you call summarize_figure, and you should keep such calls to a strict minimum. Populate the Antenna object, ensure operating_points is not empty, attach provenance that references the specific page or figure file, and return only the final JSON.

EFFICIENCY
Favor retrieval followed by targeted reads instead of full-page dumps. Avoid redundant tool calls, cache what you learn, and keep responses concise to stay within context.

Return exactly one valid Antenna JSON object conforming to the Antenna Pydantic model. Do not return extra text.
"""


def create_information_extraction_tool(
    pdf_path: str,
    store_dir: str = DEFAULT_STORE_DIR,
    *,
    max_turns: int = 30,
    tool_name: str = "antenna_info_extractor",
    tool_description: str = "Extract a structured Antenna object from a PDF article.",
):
    """Return the Antenna extraction agent wrapped as a callable tool."""

    agent = create_information_extraction_agent(pdf_path, store_dir)
    return agent.as_tool(tool_name=tool_name, tool_description=tool_description, max_turns=max_turns)

def create_information_extraction_agent(pdf_path: str, store_dir: str = DEFAULT_STORE_DIR):
    if Agent is None:
        raise RuntimeError("openai-agents runtime is not installed")

    tools = [ensure_ingested_tool, list_bundle_files, retrieve_passages, read_file, summarize_figure]
    instructions = _build_instructions(pdf_path, store_dir)
    return Agent(
        name="Antenna Info Extractor",
        instructions=instructions,
        tools=tools,
        model="gpt-4o-mini",
        output_type=Antenna,
    )


def _prepare_output_text(result: Any) -> Tuple[str, bool]:
    """Return a textual representation along with a flag indicating valid JSON."""

    if hasattr(result, "model_dump_json"):
        return result.model_dump_json(indent=2), True  # type: ignore[no-any-return]
    if hasattr(result, "model_dump"):
        try:
            return json.dumps(result.model_dump(), indent=2), True  # type: ignore[call-arg]
        except TypeError:
            pass
    if isinstance(result, (dict, list)):
        return json.dumps(result, indent=2), True
    if isinstance(result, str):
        stripped = result.strip()
        if stripped:
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return result, False
            else:
                return json.dumps(parsed, indent=2), True
        return result, False

    try:
        return json.dumps(result, indent=2, default=str), True
    except TypeError:
        return str(result), False


def _resolve_output_path(
    pdf_path: str, store_dir: str, explicit_output: Optional[Path]
) -> Optional[Path]:
    if explicit_output is not None:
        return explicit_output

    manifest = ensure_ingested(pdf_path, store_dir)
    bundle_dir = manifest.get("paths", {}).get("bundle_dir") if isinstance(manifest, dict) else None
    if not bundle_dir:
        return None
    return Path(bundle_dir) / DEFAULT_OUTPUT_FILENAME


def _persist_result_text(
    text: str, output_path: Path, *, encoding: str = "utf-8"
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding=encoding)


async def run_information_extraction_async(
    pdf_path: str,
    store_dir: str = DEFAULT_STORE_DIR,
    max_turns: int = 30,
    *,
    save_json: bool = True,
    output_path: Optional[str | Path] = None,
):
    if Runner is None or trace is None:
        raise RuntimeError("openai-agents runtime is not installed")

    agent = create_information_extraction_agent(pdf_path, store_dir)
    with trace("Extract antenna info"):
        run_result = Runner.run(agent, f"Extract antenna info from {pdf_path}", max_turns=max_turns)
    if inspect.isawaitable(run_result):
        run_result = await run_result

    if save_json:
        explicit_path = Path(output_path).expanduser() if output_path is not None else None
        target_path = _resolve_output_path(pdf_path, store_dir, explicit_path)
        if target_path is not None:
            text, _ = _prepare_output_text(run_result)
            _persist_result_text(text, target_path)

    return run_result


def run_information_extraction(
    pdf_path: str,
    store_dir: str = DEFAULT_STORE_DIR,
    max_turns: int = 30,
    *,
    save_json: bool = True,
    output_path: Optional[str | Path] = None,
):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            run_information_extraction_async(
                pdf_path,
                store_dir,
                max_turns=max_turns,
                save_json=save_json,
                output_path=output_path,
            )
        )
    else:
        return loop.create_task(
            run_information_extraction_async(
                pdf_path,
                store_dir,
                max_turns=max_turns,
                save_json=save_json,
                output_path=output_path,
            )
        )


__all__: Iterable[str] = (
    "create_information_extraction_agent",
    "create_information_extraction_tool",
    "run_information_extraction",
    "run_information_extraction_async",
)
