"""Tool definitions exposed to the OpenAI agent runtime."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from .ingest import EMBEDDING_MODEL

load_dotenv(override=True)

try:
    from agents import function_tool
except ImportError:  # pragma: no cover - fallback when the agent runtime is unavailable
    def function_tool(func=None, **_kwargs):
        if func is None:
            return lambda f: f
        return func

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - library optional in tests
    OpenAI = None  # type: ignore

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:  # pragma: no cover - optional dependency when tests skip RAG
    chromadb = None
    embedding_functions = None


def _read_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines()


@function_tool
def read_file(
    file_path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
    max_chars: Optional[int] = None,
) -> str:
    """ Read a UTF‑8 file and return a targeted slice.
    Args:
    file_path: Absolute or repo‑relative path to a .txt or .json file.
    start_line: 1‑based inclusive start line. Omit to read from the top.
    end_line: 1‑based inclusive end line. Omit to read to the end.
    max_chars: Optional hard cap on returned text length.

    Returns:
    The requested text slice (possibly truncated by max_chars).

    Raises:
    FileNotFoundError: When the file does not exist.
    ValueError: When the extension is not .txt or .json.
    """

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if path.suffix.lower() not in {".txt", ".json"}:
        raise ValueError("Only text and json files can be read through this tool")

    if start_line is None and end_line is None and max_chars is None:
        contents = path.read_text(encoding="utf-8")
    else:
        lines = _read_lines(path)
        start = max(1, start_line or 1)
        end = end_line or (len(lines) + 1)
        if end <= start:
            end = start + 1
        snippet = "\n".join(lines[start - 1 : end - 1])
        contents = snippet

    if max_chars is not None and len(contents) > max_chars:
        contents = contents[:max_chars]
    return contents


@function_tool
def list_bundle_files(
    bundle_dir: str,
    subdir: str = "pages",
    limit: int = 5,
    preview_chars: int = 160,
) -> List[Dict[str, Any]]:
    """
    List files in a bundle subdirectory with small previews.

    Args:
    bundle_dir: Ingest bundle root (contains pages/, figures/, vector_store/).
    subdir: Subfolder to list; usually 'pages' or 'figures'.
    limit: Maximum entries to return.
    preview_chars: Preview size for text/json files.

    Returns:
    A list of {name, preview, size} objects.

    Raises:
    FileNotFoundError: When the subdir does not exist.
    """

    base = Path(bundle_dir) / subdir
    if not base.exists():
        raise FileNotFoundError(f"Directory not found: {base}")

    entries: List[Dict[str, Any]] = []
    for path in sorted(base.glob("*"))[:limit]:
        preview = ""
        if path.is_file() and path.suffix.lower() in {".txt", ".json"}:
            preview = path.read_text(encoding="utf-8")[:preview_chars]
        entries.append({"name": path.name, "preview": preview, "size": path.stat().st_size})
    return entries


def _encode_image(path: Path) -> str:
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/{path.suffix.lstrip('.').lower()};base64,{b64}"


@function_tool
def summarize_figure(
    fig_path: str,
    instruction: str = "Summarize the antenna-relevant information in this figure.",
    max_chars: int = 400,
) -> Dict[str, Any]:
    """
    Use a vision model to summarize antenna‑relevant details from a figure.

    Args:
    fig_path: Path to a PNG/JPG produced by the ingest step.
    instruction: Short directive for what to extract.
    max_chars: Hard cap on the returned summary.

    Returns:
    {'summary': str, 'fig_path': str}

    Notes: 
    Sends 'input_text' + 'input_image' via the Responses API.
    Use only when text evidence is insufficient or to confirm geometry.
    """

    if OpenAI is None:
        raise RuntimeError("openai package is not available")

    path = Path(fig_path)
    if not path.exists():
        raise FileNotFoundError(f"Figure not found: {fig_path}")

    client = OpenAI()
    data_url = _encode_image(path)
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": instruction},
                    {"type": "input_image", "image_url": {"url": data_url}},
                ],
            }
        ],
    )

    text = response.output_text.strip()
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return {"summary": text, "fig_path": fig_path}


@function_tool
def retrieve_passages(bundle_dir: str, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
    """
    Query the persisted Chroma vector store for passages relevant to a query.
    
    Prerequisite:
    Run ensure_ingested first to build the vector_store and meta.json.
    
    Args:
    bundle_dir: Ingest bundle root (the folder containing vector_store/).
    query: Natural‑language query string.
    top_k: Number of passages to return.
    
    Returns:
    A list of matches: {page_file, page_number, distance, snippet}.
    
    Raises:
    FileNotFoundError: When the vector store metadata is missing.
    RuntimeError: When chromadb/embedding utilities are unavailable or OPENAI_API_KEY is unset.
    """

    if chromadb is None or embedding_functions is None:
        raise RuntimeError("chromadb (with embedding utilities) is not installed; cannot perform retrieval")

    store_path = Path(bundle_dir) / "vector_store"
    meta_path = store_path / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError("Vector store metadata not found. Run ensure_ingested first.")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    client = chromadb.PersistentClient(path=str(store_path))

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set to perform retrieval")

    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=EMBEDDING_MODEL,
    )
    collection = client.get_collection(
        meta["collection_name"],
        embedding_function=embedding_fn,
    )

    results = collection.query(query_texts=[query], n_results=top_k)
    matches: List[Dict[str, Any]] = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    for doc, metadata, distance in zip(documents, metadatas, distances):
        matches.append(
            {
                "page_file": metadata.get("page_file"),
                "page_number": metadata.get("page_number"),
                "distance": distance,
                "snippet": doc[:400],
            }
        )
    return matches


__all__ = [
    "read_file",
    "list_bundle_files",
    "summarize_figure",
    "retrieve_passages",
]



