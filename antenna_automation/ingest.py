"""PDF ingestion pipeline that extracts pages, figures, and vector indices."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import fitz  # PyMuPDF

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:  # pragma: no cover - optional at runtime
    chromadb = None
    embedding_functions = None


PARSER_VERSION = "ingest_v1.1"
DEFAULT_STORE_DIR = "ingest_store"
VECTOR_SUBDIR = "vector_store"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "text-embedding-3-large"

_CAP_RE = re.compile(r"^\s*(fig(?:ure)?\.?\s*\d+[:\.\s])", re.IGNORECASE)
_FREQ_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(kHz|MHz|GHz)", re.IGNORECASE)
_UNIT_FACTORS = {"khz": 1e3, "mhz": 1e6, "ghz": 1e9}


@dataclass
class FigureEntry:
    figure_id: str
    page: int
    bbox: Tuple[float, float, float, float]
    caption: Optional[str]
    caption_preview: Optional[str]
    image_path: str
    warnings: List[str]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def sha256_of_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def ts_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat() + "Z"


def rect_overlap_horiz(a: fitz.Rect, b: fitz.Rect) -> float:
    inter = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    if a.width == 0 or b.width == 0:
        return 0.0
    return inter / max(1e-9, min(a.width, b.width))


def normalize_frequencies(text: str) -> List[float]:
    freqs: List[float] = []
    for match in _FREQ_RE.finditer(text):
        val = float(match.group(1))
        unit = match.group(2).lower()
        freqs.append(val * _UNIT_FACTORS[unit])
    seen, out = set(), []
    for hz in freqs:
        if hz not in seen:
            seen.add(hz)
            out.append(hz)
    return out


def detect_multicol(blocks: List[Tuple[float, float, float, float, str]]) -> bool:
    if not blocks:
        return False
    page_min = min(b[0] for b in blocks)
    page_max = max(b[2] for b in blocks)
    mid = (page_min + page_max) / 2.0
    centers = [(b[0] + b[2]) / 2.0 for b in blocks]
    left = [x for x in centers if x < mid]
    right = [x for x in centers if x >= mid]
    return len(left) >= 3 and len(right) >= 3


def _extract_page_text(page: fitz.Page) -> tuple[str, list[tuple[float, float, float, float, str]]]:
    blocks = page.get_text("blocks")
    blocks_sorted = sorted(blocks, key=lambda b: (b[1], b[0]))
    text = "\n".join(b[4].strip() for b in blocks_sorted if b[4].strip())
    text_blocks = [(b[0], b[1], b[2], b[3], b[4]) for b in blocks_sorted if b[4].strip()]
    return text, text_blocks


def _find_caption_near_image(
    page: fitz.Page,
    text_blocks: list[tuple[float, float, float, float, str]],
    img_rect: fitz.Rect,
) -> tuple[Optional[str], Optional[str], list[str]]:
    warnings: List[str] = []
    lines_info = page.get_text("dict")
    line_spans: List[tuple[fitz.Rect, str]] = []
    for block in lines_info.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            xs0 = [span["bbox"][0] for span in line.get("spans", []) if "bbox" in span]
            ys0 = [span["bbox"][1] for span in line.get("spans", []) if "bbox" in span]
            xs1 = [span["bbox"][2] for span in line.get("spans", []) if "bbox" in span]
            ys1 = [span["bbox"][3] for span in line.get("spans", []) if "bbox" in span]
            if not xs0 or not ys0 or not xs1 or not ys1:
                continue
            rect = fitz.Rect(min(xs0), min(ys0), max(xs1), max(ys1))
            text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
            if text:
                line_spans.append((rect, text))

    below, above = [], []
    for rect, line in line_spans:
        if rect.y0 >= img_rect.y1 and rect.y0 <= img_rect.y1 + 80:
            if rect_overlap_horiz(rect, img_rect) >= 0.5:
                below.append((abs(rect.y0 - img_rect.y1), line))
        elif rect.y1 <= img_rect.y0 and rect.y1 >= img_rect.y0 - 60:
            if rect_overlap_horiz(rect, img_rect) >= 0.5:
                above.append((abs(img_rect.y0 - rect.y1), line))

    def pick(neigh: list[tuple[float, str]]) -> list[str]:
        neigh_sorted = sorted(neigh, key=lambda t: t[0])
        return [t[1] for t in neigh_sorted[:3]]

    candidates = pick(below) or pick(above)
    if not candidates:
        warnings.append("no_nearby_caption")
        return None, None, warnings

    caps = [c for c in candidates if _CAP_RE.match(c)]
    chosen = caps or candidates
    full = " ".join(chosen).strip()
    preview = full[:140] + ("…" if len(full) > 140 else "")
    if not caps:
        warnings.append("caption_not_prefixed_by_figure")
    return full, preview, warnings


def _extract_figures(
    page: fitz.Page,
    page_num: int,
    figures_dir: str,
    text_blocks: list[tuple[float, float, float, float, str]],
    fig_start_index: int,
) -> tuple[list[FigureEntry], int]:
    entries: List[FigureEntry] = []
    page_info = page.get_text("dict")
    image_rects: List[fitz.Rect] = [
        fitz.Rect(*block["bbox"])
        for block in page_info.get("blocks", [])
        if block.get("type", 0) == 1 and "bbox" in block
    ]
    for rect in image_rects:
        fig_id = f"fig_{fig_start_index:03d}"
        fig_start_index += 1

        zoom = 300.0 / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
        img_path = os.path.join(figures_dir, f"{fig_id}.png")
        pix.save(img_path)

        caption, caption_preview, warns = _find_caption_near_image(page, text_blocks, rect)
        entries.append(
            FigureEntry(
                figure_id=fig_id,
                page=page_num,
                bbox=(float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)),
                caption=caption,
                caption_preview=caption_preview,
                image_path=img_path,
                warnings=warns,
            )
        )
    return entries, fig_start_index


def _chunk_text(text: str, chunk_size: int, overlap: int) -> Iterable[str]:
    if not text.strip():
        return []
    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == length:
            break
        start = max(end - overlap, start + 1)
    return chunks


def _build_vector_index(bundle_dir: str, manifest: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if chromadb is None or embedding_functions is None:
        return None

    pages_dir = Path(manifest["paths"]["pages_dir"])
    if not pages_dir.exists():
        return None

    doc_ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for page_file in sorted(pages_dir.glob("page_*.txt")):
        text = page_file.read_text(encoding="utf-8")
        for idx, chunk in enumerate(_chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)):
            doc_ids.append(f"{page_file.stem}_{idx:03d}")
            documents.append(chunk)
            metadatas.append(
                {
                    "page_file": page_file.name,
                    "page_number": int(page_file.stem.split("_")[-1]),
                }
            )

    document_count = len(documents)

    vector_dir = Path(bundle_dir) / VECTOR_SUBDIR
    ensure_dir(vector_dir)

    client = chromadb.PersistentClient(path=str(vector_dir))
    collection_name = "page_chunks"
    meta_path = vector_dir / "meta.json"

    existing_meta: Optional[Dict[str, Any]] = None
    if meta_path.exists():
        try:
            existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing_meta = None

    if existing_meta:
        existing_name = existing_meta.get("collection_name", collection_name)
        try:
            collection = client.get_collection(existing_name)
            collection_count = collection.count()
        except Exception:  # pragma: no cover - corrupted cache path
            collection = None
            collection_count = -1

        if (
            collection is not None
            and existing_meta.get("chunk_size") == CHUNK_SIZE
            and existing_meta.get("chunk_overlap") == CHUNK_OVERLAP
            and existing_meta.get("embedding_model") == EMBEDDING_MODEL
            and existing_meta.get("document_count") == document_count
            and collection_count >= document_count
        ):
            return existing_meta

        if collection is not None:
            try:
                client.delete_collection(existing_name)
            except Exception:
                pass

    if not documents:
        meta = {
            "persist_dir": str(vector_dir),
            "collection_name": collection_name,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "embedding_model": EMBEDDING_MODEL,
            "document_count": 0,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    embedding_function = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name=EMBEDDING_MODEL,
    )
    collection = client.get_or_create_collection(
        name=collection_name, embedding_function=embedding_function
    )

    collection.add(ids=doc_ids, metadatas=metadatas, documents=documents)

    meta = {
        "persist_dir": str(vector_dir),
        "collection_name": collection_name,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embedding_model": EMBEDDING_MODEL,
        "document_count": document_count,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _empty_manifest(file_hash: str, pdf_abs: str, bundle_dir: str) -> Dict[str, Any]:
    return {
        "bundle_id": file_hash,
        "parser_version": PARSER_VERSION,
        "status": "failed",
        "cache": "miss",
        "created_at": ts_now(),
        "pdf_path": pdf_abs,
        "page_count": 0,
        "figure_count": 0,
        "pages_multi_column": [],
        "used_ocr": False,
        "frequency_hz_candidates": [],
        "vector_index": None,
        "paths": {
            "bundle_dir": os.path.abspath(bundle_dir),
            "pages_dir": os.path.join(os.path.abspath(bundle_dir), "pages"),
            "figures_dir": os.path.join(os.path.abspath(bundle_dir), "figures"),
            "manifest_path": os.path.join(os.path.abspath(bundle_dir), "manifest.json"),
            "figures_index": os.path.join(
                os.path.abspath(bundle_dir), "figures", "index.json"
            ),
        },
        "warnings": [],
    }


def _ensure_ingested_impl(pdf_path: str, store_dir: str = DEFAULT_STORE_DIR) -> Dict[str, Any]:
    if not isinstance(pdf_path, str) or not pdf_path.lower().endswith(".pdf"):
        return {"status": "failed", "cache": "miss", "error": "pdf_path must be a .pdf file"}
    if not os.path.isfile(pdf_path):
        return {"status": "failed", "cache": "miss", "error": f"file not found: {pdf_path}"}

    pdf_abs = os.path.abspath(pdf_path)
    file_hash = sha256_of_file(pdf_abs)
    ensure_dir(store_dir)
    bundle_dir = os.path.join(store_dir, file_hash)
    manifest_path = os.path.join(bundle_dir, "manifest.json")

    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        manifest["cache"] = "hit"
        manifest.setdefault("status", "ready")
        manifest.setdefault("vector_index", None)
        return manifest

    ensure_dir(bundle_dir)
    pages_dir = os.path.join(bundle_dir, "pages")
    figures_dir = os.path.join(bundle_dir, "figures")
    ensure_dir(pages_dir)
    ensure_dir(figures_dir)

    manifest = _empty_manifest(file_hash, pdf_abs, bundle_dir)
    manifest["status"] = "ready"
    manifest["cache"] = "miss"

    pages_multi_col: List[int] = []
    freq_candidates: List[float] = []
    figure_entries: List[FigureEntry] = []

    try:
        doc = fitz.open(pdf_abs)
        manifest["page_count"] = doc.page_count
        fig_index = 1

        for i in range(doc.page_count):
            page_no = i + 1
            page = doc.load_page(i)
            text, text_blocks = _extract_page_text(page)

            with open(os.path.join(pages_dir, f"page_{page_no:04d}.txt"), "w", encoding="utf-8") as fh:
                fh.write(text)

            if detect_multicol(text_blocks):
                pages_multi_col.append(page_no)
            if text:
                freq_candidates.extend(normalize_frequencies(text))

            figs, fig_index = _extract_figures(page, page_no, figures_dir, text_blocks, fig_index)
            figure_entries.extend(figs)

        for entry in figure_entries:
            if entry.caption:
                freq_candidates.extend(normalize_frequencies(entry.caption))

        seen, unique_freqs = set(), []
        for hz in freq_candidates:
            if hz not in seen:
                seen.add(hz)
                unique_freqs.append(hz)

        figures_index_path = os.path.join(figures_dir, "index.json")
        with open(figures_index_path, "w", encoding="utf-8") as index_file:
            json.dump(
                [
                    {
                        "figure_id": fe.figure_id,
                        "page": fe.page,
                        "bbox": fe.bbox,
                        "caption": fe.caption,
                        "caption_preview": fe.caption_preview,
                        "image_path": os.path.relpath(fe.image_path, bundle_dir),
                        "warnings": fe.warnings,
                    }
                    for fe in figure_entries
                ],
                index_file,
                ensure_ascii=False,
                indent=2,
            )

        manifest.update(
            {
                "figure_count": len(figure_entries),
                "pages_multi_column": pages_multi_col,
                "frequency_hz_candidates": unique_freqs,
                "paths": {
                    "bundle_dir": os.path.abspath(bundle_dir),
                    "pages_dir": os.path.abspath(pages_dir),
                    "figures_dir": os.path.abspath(figures_dir),
                    "manifest_path": os.path.abspath(manifest_path),
                    "figures_index": os.path.abspath(figures_index_path),
                },
            }
        )

        vector_meta = _build_vector_index(bundle_dir, manifest)
        manifest["vector_index"] = vector_meta

        with open(manifest_path, "w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)

        return manifest

    except Exception as exc:  # pragma: no cover - defensive path
        manifest["status"] = "failed"
        manifest["warnings"].append(f"ingest_error:{exc!r}")
        try:
            with open(manifest_path, "w", encoding="utf-8") as manifest_file:
                json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return manifest


def ensure_ingested(pdf_path: str, store_dir: str = DEFAULT_STORE_DIR) -> Dict[str, Any]:
    """Parse a PDF into a cached bundle with page text, figure crops, and embeddings."""

    return _ensure_ingested_impl(pdf_path=pdf_path, store_dir=store_dir)


__all__ = ["ensure_ingested", "PARSER_VERSION", "DEFAULT_STORE_DIR"]
