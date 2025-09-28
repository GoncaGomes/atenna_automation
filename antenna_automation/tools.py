"""Tool definitions exposed to the OpenAI agent runtime."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

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
except ImportError:  # pragma: no cover - optional dependency when tests skip RAG
    chromadb = None


def _read_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines()


@function_tool
def read_file(
    file_path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
    max_chars: Optional[int] = None,
) -> str:
    """Read part of a UTF-8 text file."""

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
    """List files inside a bundle folder with a small preview to help navigation."""

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
    """Summarize a figure using an OpenAI vision-capable model."""

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
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": data_url}},
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
    """Run similarity search over the cached vector store and return top passages."""

    if chromadb is None:
        raise RuntimeError("chromadb is not installed; cannot perform retrieval")

    store_path = Path(bundle_dir) / "vector_store"
    meta_path = store_path / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError("Vector store metadata not found. Run ensure_ingested first.")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    client = chromadb.PersistentClient(path=str(store_path))
    collection = client.get_collection(meta["collection_name"])

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
