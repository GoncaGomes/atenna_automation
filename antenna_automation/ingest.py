"""PDF ingestion pipeline that extracts pages, figures, and vector indices."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
from collections import Counter
from statistics import median
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

try:
    import pdfplumber
except ImportError:  # pragma: no cover - optional dependency
    pdfplumber = None


PARSER_VERSION = "ingest_v1.1"
DEFAULT_STORE_DIR = "ingest_store"
VECTOR_SUBDIR = "vector_store"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "text-embedding-3-small"
HEADER_SAMPLE_LINES = 3
FOOTER_SAMPLE_LINES = 3
HEADER_FOOTER_THRESHOLD_RATIO = 0.6

_CAP_RE = re.compile(r"^\s*(fig(?:ure)?\.?\s*\d+[:\.\s])", re.IGNORECASE)
_FREQ_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(kHz|MHz|GHz)", re.IGNORECASE)
_TABLE_LABEL_RE = re.compile(
    r"^\s*table\s+(\d+)\s*[:\.\-]?\s*(.*)$",
    re.IGNORECASE,
)
_REFERENCES_RE = re.compile(r"^\s*references\b", re.IGNORECASE)
_CANONICAL_SECTION_KEYWORDS = [
    "abstract",
    "introduction",
    "background",
    "method",
    "methodology",
    "methods",
    "design",
    "analysis",
    "implementation",
    "simulation",
    "results",
    "discussion",
    "evaluation",
    "conclusion",
    "future work",
    "acknowledgments",
    "acknowledgements",
]
_UNIT_FACTORS = {"khz": 1e3, "mhz": 1e6, "ghz": 1e9}


def _starts_with_canonical(text: str) -> bool:
    stripped = text.lstrip().lower()
    for keyword in _CANONICAL_SECTION_KEYWORDS:
        if stripped.startswith(keyword):
            next_pos = len(keyword)
            if len(stripped) == next_pos:
                return True
            if stripped[next_pos] in (" ", ":", ".", "-", "\u2013", "\u2014"):
                return True
    return False


@dataclass
class FigureEntry:
    figure_id: str
    page: int
    bbox: Tuple[float, float, float, float]
    caption: Optional[str]
    caption_preview: Optional[str]
    image_path: str
    warnings: List[str]


@dataclass
class TableEntry:
    table_id: str
    page: int
    bbox: Tuple[float, float, float, float]
    data: List[List[str]]
    file_path: str
    source: str = "pdfplumber"
    title: Optional[str] = None
    legend: Optional[str] = None


def _parse_table_label(text: str) -> tuple[Optional[str], Optional[str]]:
    stripped = text.strip()
    match = _TABLE_LABEL_RE.match(stripped)
    if not match:
        return None, None
    legend = match.group(2).strip(" .:-") if match.group(2) else None
    return match.group(1), legend or None


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


def _table_bbox_close(
    bbox_a: Tuple[float, float, float, float],
    bbox_b: Tuple[float, float, float, float],
    tolerance: float = 4.0,
) -> bool:
    return all(abs(a - b) <= tolerance for a, b in zip(bbox_a, bbox_b))


def _is_duplicate_table(
    existing: List[TableEntry],
    page: int,
    bbox: Tuple[float, float, float, float],
) -> bool:
    for entry in existing:
        if entry.page != page:
            continue
        if _table_bbox_close(entry.bbox, bbox):
            return True
    return False


def _extract_tables_from_text_block(
    page: "pdfplumber.page.Page",
    page_no: int,
    tables_dir: str,
    table_start_index: int,
    existing: List[TableEntry],
) -> tuple[list[TableEntry], int]:
    entries: List[TableEntry] = []
    try:
        text = page.extract_text() or ""
    except Exception:
        return entries, table_start_index

    if not text.strip():
        return entries, table_start_index

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    count = len(lines)

    def has_digits(value: str) -> bool:
        return any(ch.isdigit() for ch in value)

    idx = 0
    while idx < count:
        line = lines[idx]
        lower = line.lower()
        table_number, legend_text = _parse_table_label(line)

        if table_number == "1":
            header_idx = idx + 1
            if header_idx >= count:
                idx += 1
                continue
            header_line = lines[header_idx]
            header_parts = header_line.split()
            if len(header_parts) < 2:
                idx += 1
                continue
            header_left = header_parts[0]
            header_right = " ".join(header_parts[1:])
            rows = [[header_left, header_right]]
            data_idx = header_idx + 1
            while data_idx < count:
                tokens = lines[data_idx].split()
                if len(tokens) < 2:
                    break
                if not has_digits(tokens[-1]):
                    break
                name = " ".join(tokens[:-1])
                value = tokens[-1]
                rows.append([name, value])
                data_idx += 1
            if len(rows) > 1:
                table_id = f"table_{table_start_index:03d}"
                table_start_index += 1
                output_path = os.path.join(tables_dir, f"{table_id}.json")
                entry = TableEntry(
                    table_id=table_id,
                    page=page_no,
                    bbox=(0.0, 0.0, 0.0, 0.0),
                    data=rows,
                    file_path=output_path,
                    source="pdfplumber",
                    title=f"Table {table_number}",
                    legend=legend_text,
                )
                with open(output_path, "w", encoding="utf-8") as table_file:
                    json.dump(
                        {
                            "table_id": entry.table_id,
                            "page": entry.page,
                            "bbox": entry.bbox,
                            "rows": entry.data,
                            "source": entry.source,
                            "title": entry.title,
                            "legend": entry.legend,
                        },
                        table_file,
                        ensure_ascii=False,
                        indent=2,
                    )
                entries.append(entry)
            idx = header_idx + 1
            continue

        if table_number == "2" and len(lower) > len("table 2"):
            data_idx = idx + 1
            while data_idx < count and not lines[data_idx]:
                data_idx += 1
            if data_idx >= count:
                idx += 1
                continue

            header_tokens = lines[data_idx].split()
            if not header_tokens or not any(tok.lower().endswith("axis") for tok in header_tokens):
                idx += 1
                continue
            column_names = header_tokens
            data_idx += 1

            row_entries: List[List[str]] = []
            while data_idx < count:
                tokens = lines[data_idx].split()
                if len(tokens) <= len(column_names):
                    break
                values: List[str] = []
                labels: List[str] = []
                for tok in reversed(tokens):
                    if has_digits(tok) and len(values) < len(column_names):
                        values.append(tok)
                    else:
                        labels.insert(0, tok)
                values.reverse()
                if len(values) != len(column_names) or not labels:
                    break
                row_entries.append([" ".join(labels)] + values)
                data_idx += 1

            if row_entries:
                header_row = ["Parameter"] + column_names
                rows = [header_row] + row_entries

                table_id = f"table_{table_start_index:03d}"
                table_start_index += 1
                output_path = os.path.join(tables_dir, f"{table_id}.json")
                entry = TableEntry(
                    table_id=table_id,
                    page=page_no,
                    bbox=(0.0, 0.0, 0.0, 0.0),
                    data=rows,
                    file_path=output_path,
                    source="pdfplumber",
                    title=f"Table {table_number}",
                    legend=legend_text,
                )
                with open(output_path, "w", encoding="utf-8") as table_file:
                    json.dump(
                        {
                            "table_id": entry.table_id,
                            "page": entry.page,
                            "bbox": entry.bbox,
                            "rows": entry.data,
                            "source": entry.source,
                            "title": entry.title,
                            "legend": entry.legend,
                        },
                        table_file,
                        ensure_ascii=False,
                        indent=2,
                    )
                entries.append(entry)
                idx = data_idx
                continue

            idx += 1
            continue

        idx += 1

    return entries, table_start_index

    if len(words) < 2:
        return entries, table_start_index

    for idx, word in enumerate(words[:-1]):
        if word.get("text", "").strip().lower() != "table":
            continue
        next_token = words[idx + 1].get("text", "").strip()
        match = re.match(r"(\d+)", next_token)
        if not match:
            continue
        table_number = match.group(1)

        label_bottom = max(word.get("bottom", 0.0), words[idx + 1].get("bottom", 0.0))
        candidate_words = [
            w
            for w in words
            if label_bottom <= w.get("top", 0.0) <= label_bottom + max_scan_height
        ]
        if not candidate_words:
            continue

        candidate_words = [w for w in candidate_words if w.get("top", 0.0) >= label_bottom + 2.0]
        numeric_positions = [
            w.get("x0", 0.0)
            for w in candidate_words
            if any(ch.isdigit() for ch in w.get("text", ""))
        ]
        if not numeric_positions:
            continue
        numeric_positions.sort()
        value_x = numeric_positions[len(numeric_positions) // 2]
        value_min = value_x - 30.0
        value_max = value_x + 30.0

        param_positions = [
            w.get("x0", 0.0)
            for w in candidate_words
            if w.get("x0", 0.0) < value_min - 10.0
        ]
        if not param_positions:
            param_positions = [
                w.get("x0", 0.0)
                for w in candidate_words
                if not any(ch.isdigit() for ch in w.get("text", ""))
            ]
        if not param_positions:
            continue
        param_positions.sort()
        param_x = param_positions[len(param_positions) // 2]
        param_min = param_x - 40.0
        param_max = param_x + 60.0

        filtered_words = [
            w
            for w in candidate_words
            if param_min <= w.get("x0", 0.0) <= param_max or value_min <= w.get("x0", 0.0) <= value_max
        ]
        if len(filtered_words) < 4:
            continue

        numeric_words = [
            w for w in filtered_words if any(ch.isdigit() for ch in w.get("text", ""))
        ]
        if numeric_words:
            numeric_max_bottom = max(w.get("bottom", 0.0) for w in numeric_words)
            filtered_words = [
                w for w in filtered_words if w.get("top", 0.0) <= numeric_max_bottom + 2.0
            ]
        if len(filtered_words) < 4:
            continue

        row_map: Dict[float, List[dict[str, Any]]] = {}
        for w in filtered_words:
            row_key = round(w.get("top", 0.0), 1)
            row_map.setdefault(row_key, []).append(w)

        ordered_row_keys = sorted(row_map.keys())
        rows_data: List[List[str]] = []
        row_word_objs: List[List[dict[str, Any]]] = []

        for row_key in ordered_row_keys:
            row_words = sorted(row_map[row_key], key=lambda w: w.get("x0", 0.0))
            cells: List[str] = []
            cell_words: List[List[dict[str, Any]]] = []
            current_cell: List[str] = []
            current_words: List[dict[str, Any]] = []
            current_x: Optional[float] = None

            for w in row_words:
                x0 = w.get("x0", 0.0)
                if current_x is None or x0 - current_x <= column_gap_threshold:
                    current_cell.append(w.get("text", ""))
                    current_words.append(w)
                    current_x = x0 if current_x is None else (current_x + x0) / 2.0
                else:
                    if current_cell:
                        cells.append(" ".join(current_cell).strip())
                        cell_words.append(current_words)
                    current_cell = [w.get("text", "")]
                    current_words = [w]
                    current_x = x0

            if current_cell:
                cells.append(" ".join(current_cell).strip())
                cell_words.append(current_words)

            if len(cells) > 2:
                left_words = [item for group in cell_words[:1] for item in group]
                right_words = [item for group in cell_words[1:] for item in group]
                right_text = " ".join(cells[1:]).strip()
                cells = [cells[0], right_text]
                cell_words = [left_words, right_words]

            word_count = sum(len(cell.split()) for cell in cells)
            if len(cells) <= 1 and word_count > 6:
                break
            if len(cells) < 2:
                continue

            rows_data.append(cells)
            row_word_objs.append([item for group in cell_words for item in group])

        if len(rows_data) < 2:
            continue

        max_cols = max(len(row) for row in rows_data)
        if max_cols <= 1:
            continue

        for row in rows_data:
            if len(row) < max_cols:
                row.extend([""] * (max_cols - len(row)))

        data_rows = rows_data[1:]
        if data_rows:
            numeric_rows = sum(
                1 for row in data_rows if any(ch.isdigit() for cell in row for ch in cell)
            )
            if numeric_rows < max(1, len(data_rows) - 1):
                continue

        header_text = rows_data[0][0]
        if len(header_text.split()) > 6:
            continue

        used_words = [w for group in row_word_objs for w in group]
        if not used_words:
            continue

        bbox = (
            float(min(w.get("x0", 0.0) for w in used_words)),
            float(min(w.get("top", 0.0) for w in used_words)),
            float(max(w.get("x1", 0.0) for w in used_words)),
            float(max(w.get("bottom", 0.0) for w in used_words)),
        )
        if _is_duplicate_table(existing + entries, page_no, bbox):
            continue

        table_id = f"table_{table_start_index:03d}"
        table_start_index += 1
        output_path = os.path.join(tables_dir, f"{table_id}.json")
        entry = TableEntry(
            table_id=table_id,
            page=page_no,
            bbox=bbox,
            data=rows_data,
            file_path=output_path,
            source="pdfplumber",
            title=f"Table {table_number}",
        )
        with open(output_path, "w", encoding="utf-8") as table_file:
            json.dump(
                {
                    "table_id": entry.table_id,
                    "page": entry.page,
                    "bbox": entry.bbox,
                    "rows": entry.data,
                    "source": entry.source,
                    "title": entry.title,
                },
                table_file,
                ensure_ascii=False,
                indent=2,
            )
        entries.append(entry)

    return entries, table_start_index

    if not words:
        return entries, table_start_index

    total_words = len(words)
    idx = 0
    while idx < total_words:
        word = words[idx]
        text_lower = word.get("text", "").lower()
        if text_lower != "table":
            idx += 1
            continue

        if idx + 1 >= total_words:
            break
        next_text = words[idx + 1]["text"]
        if not any(ch.isdigit() for ch in next_text):
            idx += 1
            continue

        label_bottom = word.get("bottom", 0.0)
        candidate_words = [
            w
            for w in words
            if label_bottom <= w.get("top", 0.0) <= label_bottom + max_scan_height
        ]
        if not candidate_words:
            idx += 1
            continue

        candidate_words = [w for w in candidate_words if w.get("top", 0.0) >= label_bottom + 2.0]
        numeric_positions = [
            w.get("x0", 0.0)
            for w in candidate_words
            if any(ch.isdigit() for ch in w.get("text", ""))
        ]
        if not numeric_positions:
            idx += 1
            continue
        numeric_positions.sort()
        value_x = numeric_positions[len(numeric_positions) // 2]
        value_min = value_x - 30.0
        value_max = value_x + 30.0

        param_positions = [
            w.get("x0", 0.0)
            for w in candidate_words
            if w.get("x0", 0.0) < value_min - 10.0
        ]
        if not param_positions:
            param_positions = [
                w.get("x0", 0.0)
                for w in candidate_words
                if not any(ch.isdigit() for ch in w.get("text", ""))
            ]
        if not param_positions:
            idx += 1
            continue
        param_positions.sort()
        param_x = param_positions[len(param_positions) // 2]
        param_min = param_x - 40.0
        param_max = param_x + 60.0

        filtered_words = []
        for w in candidate_words:
            x0 = w.get("x0", 0.0)
            if param_min <= x0 <= param_max or value_min <= x0 <= value_max:
                filtered_words.append(w)
        if not filtered_words:
            idx += 1
            continue

        candidate_words = filtered_words

        numeric_words = [
            w for w in candidate_words if any(ch.isdigit() for ch in w.get("text", ""))
        ]
        if numeric_words:
            numeric_max_bottom = max(w.get("bottom", 0.0) for w in numeric_words)
            candidate_words = [
                w
                for w in candidate_words
                if w.get("top", 0.0) <= numeric_max_bottom + 2.0
            ]
            if not candidate_words:
                idx += 1
                continue

        row_map: Dict[float, List[dict[str, Any]]] = {}
        for w in candidate_words:
            row_key = round(w.get("top", 0.0), 1)
            row_map.setdefault(row_key, []).append(w)

        ordered_row_keys = sorted(row_map.keys())
        rows_data: List[List[str]] = []
        row_word_objs: List[List[dict[str, Any]]] = []

        for row_key in ordered_row_keys:
            row_words = sorted(row_map[row_key], key=lambda w: w.get("x0", 0.0))
            cells: List[str] = []
            cell_words: List[List[dict[str, Any]]] = []
            current_cell: List[str] = []
            current_words: List[dict[str, Any]] = []
            current_x: Optional[float] = None

            for w in row_words:
                x0 = w.get("x0", 0.0)
                if current_x is None or x0 - current_x <= column_gap_threshold:
                    current_cell.append(w.get("text", ""))
                    current_words.append(w)
                    current_x = x0 if current_x is None else (current_x + x0) / 2.0
                else:
                    if current_cell:
                        cells.append(" ".join(current_cell).strip())
                        cell_words.append(current_words)
                    current_cell = [w.get("text", "")]
                    current_words = [w]
                    current_x = x0

            if current_cell:
                cells.append(" ".join(current_cell).strip())
                cell_words.append(current_words)

            if len(cells) > 2:
                left_words = [item for group in cell_words[:1] for item in group]
                right_words = [item for group in cell_words[1:] for item in group]
                right_text = " ".join(cells[1:]).strip()
                cells = [cells[0], right_text]
                cell_words = [left_words, right_words]

            word_count = sum(len(cell.split()) for cell in cells)
            if len(cells) <= 1 and word_count > 6:
                break
            if len(cells) < 2:
                continue

            rows_data.append(cells)
            row_word_objs.append([item for group in cell_words for item in group])

        if len(rows_data) < 2:
            idx += 1
            continue

        max_cols = max(len(row) for row in rows_data)
        if max_cols <= 1:
            idx += 1
            continue

        for row in rows_data:
            if len(row) < max_cols:
                row.extend([""] * (max_cols - len(row)))

        filtered_rows: List[List[str]] = []
        filtered_word_objs: List[List[dict[str, Any]]] = []
        for row, words in zip(rows_data, row_word_objs):
            if row and row[0].lower().startswith("table"):
                continue
            filtered_rows.append(row)
            filtered_word_objs.append(words)

        rows_data = filtered_rows
        row_word_objs = filtered_word_objs

        if not rows_data:
            idx += 1
            continue

        used_words = [w for group in row_word_objs for w in group]
        if not used_words:
            idx += 1
            continue

        bbox = (
            float(min(w.get("x0", 0.0) for w in used_words)),
            float(min(w.get("top", 0.0) for w in used_words)),
            float(max(w.get("x1", 0.0) for w in used_words)),
            float(max(w.get("bottom", 0.0) for w in used_words)),
        )
        if _is_duplicate_table(existing + entries, page_no, bbox):
            idx += 1
            continue

        table_id = f"table_{table_start_index:03d}"
        table_start_index += 1
        output_path = os.path.join(tables_dir, f"{table_id}.json")
        entry = TableEntry(
            table_id=table_id,
            page=page_no,
            bbox=bbox,
            data=rows_data,
            file_path=output_path,
            source="pdfplumber_text",
            title=rows_data[0][0] if rows_data and rows_data[0] else None,
        )
        try:
            with open(output_path, "w", encoding="utf-8") as table_file:
                json.dump(
                    {
                        "table_id": entry.table_id,
                        "page": entry.page,
                        "bbox": entry.bbox,
                        "rows": entry.data,
                        "source": entry.source,
                        "title": entry.title,
                    },
                    table_file,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception:
            idx += 1
            continue

        entries.append(entry)
        idx += 1

    return entries, table_start_index



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


def _extract_tables_pdfplumber(
    pdf_path: str,
    tables_dir: str,
    table_start_index: int,
    existing: List[TableEntry],
) -> tuple[list[TableEntry], int]:
    entries: List[TableEntry] = []
    if pdfplumber is None:
        return entries, table_start_index

    try:
        with pdfplumber.open(pdf_path) as plumber_doc:
            for page in plumber_doc.pages:
                text_entries, table_start_index = _extract_tables_from_text_block(
                    page,
                    page.page_number,
                    tables_dir,
                    table_start_index,
                    existing + entries,
                )
                entries.extend(text_entries)
    except Exception:
        return entries, table_start_index

    return entries, table_start_index

    all_existing = list(existing)

    try:
        with pdfplumber.open(pdf_path) as plumber_doc:
            for page in plumber_doc.pages:
                page_no = page.page_number
                try:
                    tables = page.find_tables(
                        table_settings={
                            "vertical_strategy": "lines",
                            "horizontal_strategy": "lines",
                        }
                    )
                except Exception:
                    tables = []

                if not tables:
                    try:
                        tables = page.find_tables(
                            table_settings={
                                "vertical_strategy": "text",
                                "horizontal_strategy": "text",
                                "intersection_tolerance": 5,
                            }
                        )
                    except Exception:
                        tables = []

                for table in tables or []:
                    try:
                        bbox_raw = tuple(float(v) for v in table.bbox)
                    except Exception:
                        continue
                    page_width = float(getattr(page, "width", 0) or 0.0)
                    page_height = float(getattr(page, "height", 0) or 0.0)
                    if page_width and page_height:
                        width = bbox_raw[2] - bbox_raw[0]
                        height = bbox_raw[3] - bbox_raw[1]
                        area_ratio = (width * height) / (page_width * page_height)
                        if (width >= page_width * 0.85 and height >= page_height * 0.35) or area_ratio >= 0.25:
                            continue
                        if height >= page_height * 0.2:
                            continue
                    if _is_duplicate_table(all_existing, page_no, bbox_raw):
                        continue

                extracted = table.extract() or []
                cleaned_rows: List[List[str]] = []
                for idx_row, row in enumerate(extracted):
                    if not isinstance(row, list):
                        continue
                    cleaned = [(cell or "").strip() for cell in row[:2]]
                    if not any(cleaned):
                        continue
                    has_digit = any(any(ch.isdigit() for ch in cell) for cell in cleaned)
                    if idx_row > 0 and not has_digit:
                        break
                    cleaned_rows.append(cleaned)
                if len(cleaned_rows) < 2:
                    continue
                total_cells = sum(len(row) for row in cleaned_rows)
                if total_cells == 0:
                    continue
                filled_cells = sum(1 for row in cleaned_rows for cell in row if cell)
                fill_ratio = filled_cells / total_cells
                if fill_ratio < 0.3:
                    continue
                max_cols = max(len(row) for row in cleaned_rows)
                if max_cols <= 1:
                    continue
                if filled_cells < 2:
                    continue
                if len(cleaned_rows) > 20:
                    continue
                contains_digit = any(
                    any(ch.isdigit() for ch in cell)
                    for row in cleaned_rows
                    for cell in row
                )
                if not contains_digit:
                    continue
                data_rows = cleaned_rows[1:]
                if data_rows:
                    numeric_rows = sum(
                        1 for row in data_rows if any(ch.isdigit() for cell in row for ch in cell)
                    )
                    if numeric_rows < max(1, len(data_rows) - 1):
                        continue
                header_text = cleaned_rows[0][0]
                if len(header_text.split()) > 6:
                    continue

                table_id = f"table_{table_start_index:03d}"
                table_start_index += 1
                output_path = os.path.join(tables_dir, f"{table_id}.json")
                entry = TableEntry(
                    table_id=table_id,
                    page=page_no,
                    bbox=bbox_raw,
                    data=cleaned_rows,
                    file_path=output_path,
                    source="pdfplumber",
                    title=cleaned_rows[0][0] if cleaned_rows and cleaned_rows[0] else None,
                )
                try:
                    _serialize_entry(entry)
                except Exception:
                    continue
                entries.append(entry)
                all_existing.append(entry)

                if not any(existing_entry.page == page_no for existing_entry in all_existing):
                    text_entries, table_start_index = _extract_tables_from_text_block(
                        page,
                        page_no,
                        tables_dir,
                        table_start_index,
                        all_existing,
                    )
                    for entry in text_entries:
                        entries.append(entry)
                        all_existing.append(entry)
    except Exception:
        return entries, table_start_index

    return entries, table_start_index


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


def _derive_common_lines(counter: Counter, total_pages: int) -> set[str]:
    if total_pages <= 0:
        return set()
    threshold = max(2, int(total_pages * HEADER_FOOTER_THRESHOLD_RATIO))
    return {line for line, count in counter.items() if count >= threshold and line}


def _clean_page_lines(
    raw_lines: List[str],
    header_lines: set[str],
    footer_lines: set[str],
    references_started: bool,
) -> tuple[List[str], bool, bool]:
    cleaned: List[str] = []
    in_references = references_started
    references_triggered = False
    total = len(raw_lines)

    for idx, original in enumerate(raw_lines):
        stripped = original.strip()
        if in_references:
            references_triggered = True
            break
        if stripped:
            if idx < HEADER_SAMPLE_LINES and stripped in header_lines:
                continue
            if idx >= max(0, total - FOOTER_SAMPLE_LINES) and stripped in footer_lines:
                continue
            if _REFERENCES_RE.match(stripped):
                in_references = True
                references_triggered = True
                break
        cleaned.append(original)

    return cleaned, in_references, references_triggered


def _extract_title_and_sections(
    page_infos: List[Dict[str, Any]],
    toc: Optional[List[Tuple[int, str, int]]] = None,
    metadata_title: Optional[str] = None,
) -> tuple[Optional[str], List[Dict[str, Any]]]:
    if not page_infos:
        clean_title = metadata_title.strip() if metadata_title else None
        return clean_title or None, []

    title: Optional[str] = metadata_title.strip() if metadata_title and metadata_title.strip() else None
    sections: List[Dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    def add_section(name: str, page: int) -> None:
        clean = name.strip(" :-")
        if not clean or len(clean) < 3:
            return
        key = (clean.lower(), page)
        if key in seen:
            return
        sections.append({"title": clean, "page": page})
        seen.add(key)

    # Prefer outline / table of contents when available
    if toc:
        for level, name, page in toc:
            if not isinstance(level, int) or not isinstance(page, int):
                continue
            cleaned = name.strip()
            if not cleaned:
                continue
            if title is None and level == 1 and len(cleaned) >= 5:
                title = cleaned
            if cleaned and page > 0:
                if title and cleaned.lower() == title.lower():
                    continue
                add_section(cleaned.rstrip(":"), page)

    # Fallback to layout heuristics when TOC is missing or incomplete
    line_counts: Counter[str] = Counter()
    for info in page_infos:
        for line in info.get("lines_with_sizes", []):
            text = line["text"].strip().lower()
            if text:
                line_counts[text] += 1

    for info in page_infos:
        page_no = info["page_no"]
        median_size = info.get("median_font_size", 0.0)
        for line in info.get("lines_with_sizes", []):
            text = line["text"].strip()
            if not text or len(text) > 120:
                continue
            if _REFERENCES_RE.match(text):
                break
            lower = text.lower()
            if lower.startswith(("journal of", "doi", "paper", "this content was downloaded")):
                continue
            is_numbered = bool(re.match(r"^\d+(?:\.\d+)*[)\.]?\s+", text))
            is_upper = text.isupper() and len(text.split()) <= 12
            size = line["size"]
            is_large = median_size == 0 or size >= median_size * 1.18 or size - median_size >= 1.5
            keyword_prefix = _starts_with_canonical(text)
            short_enough = len(text.split()) <= 18
            unique_large = is_large and short_enough and line_counts[lower] == 1

            if page_no == 1 and not (is_numbered or keyword_prefix or is_upper):
                continue

            if line_counts[lower] > 1 and not (keyword_prefix or is_numbered):
                continue

            if not (is_numbered or is_upper or keyword_prefix or unique_large):
                continue

            section_text = text
            if is_numbered:
                section_text = re.sub(r"^\d+(?:\.\d+)*[)\.]?\s+", "", section_text).strip()
            if keyword_prefix:
                section_text = re.split(r"[:\-\u2013\u2014\.]", section_text, maxsplit=1)[0].strip()
            section_text = section_text.rstrip(":")

            word_count = len(section_text.split())
            if word_count == 0:
                continue
            if word_count == 1 and not (keyword_prefix or is_upper or is_numbered):
                continue
            if word_count > 8 and not (is_numbered or keyword_prefix):
                continue
            if len(section_text) > 60 and not (is_numbered or keyword_prefix):
                continue
            alpha_chars = sum(1 for ch in section_text if ch.isalpha())
            if alpha_chars < 3:
                continue

            used_unique_large = unique_large and not (is_numbered or is_upper or keyword_prefix)
            if used_unique_large:
                words = section_text.split()
                if not words:
                    continue
                title_like = sum(1 for w in words if w[:1].isupper())
                if title_like / len(words) < 0.6:
                    continue
                digit_count = sum(1 for ch in section_text if ch.isdigit())
                if digit_count > 2:
                    continue

            add_section(section_text, page_no)

    # Fall back to first-page font sizing for title if outline + metadata failed
    if title is None and page_infos:
        first_page = page_infos[0]
        candidates: List[tuple[float, float, str]] = []
        for line in first_page.get("lines_with_sizes", []):
            text = line["text"].strip()
            if not text:
                continue
            lower = text.lower()
            if lower.startswith(("journal of", "doi", "paper", "this content was downloaded")):
                continue
            candidates.append((line["size"], line["y0"], text))

        if candidates:
            candidates.sort(key=lambda t: (-t[0], t[1]))
            for _, _, text in candidates:
                normalized = text.strip(" -")
                if len(normalized) >= 5:
                    title = normalized
                    break

    filtered_sections: List[Dict[str, Any]] = []
    seen_titles: set[tuple[str, int]] = set()
    for entry in sections:
        name = entry["title"].strip()
        page = entry["page"]
        key = (name.lower(), page)
        if key in seen_titles:
            continue
        words = name.split()
        keyword_start = _starts_with_canonical(name)
        if keyword_start:
            filtered_sections.append(entry)
            seen_titles.add(key)
            continue

        if len(words) > 8:
            continue
        if sum(1 for ch in name if ch.isalpha()) < 3:
            continue
        if name[:1].isalpha() and not name[:1].isupper():
            continue
        if any(sym in name for sym in "=±*/"):
            continue
        if "," in name:
            continue
        alpha_words = sum(1 for w in words if any(ch.isalpha() for ch in w))
        if alpha_words <= 1:
            continue
        if sum(1 for ch in name if ch.isalnum() or ch.isspace()) / max(1, len(name)) < 0.6:
            continue
        filtered_sections.append(entry)
        seen_titles.add(key)

    return title, filtered_sections


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

    tables_index_path = Path(manifest["paths"].get("tables_index", ""))
    bundle_path = Path(manifest["paths"].get("bundle_dir", bundle_dir))
    if tables_index_path.is_file():
        try:
            tables_index = json.loads(tables_index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            tables_index = []
        for idx, entry in enumerate(tables_index):
            rel_path = entry.get("path")
            if not rel_path:
                continue
            table_file = bundle_path / rel_path
            if not table_file.is_file():
                continue
            try:
                table_payload = json.loads(table_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            rows = table_payload.get("rows", [])
            if not isinstance(rows, list):
                continue
            flattened_rows: List[str] = []
            for row in rows:
                if not isinstance(row, list):
                    continue
                flattened_rows.append(" | ".join(str(cell).strip() for cell in row))
            flattened_text = "\n".join(r for r in flattened_rows if r.strip())
            if not flattened_text.strip():
                continue
            doc_ids.append(f"{entry.get('table_id', f'table_{idx:03d}')}_table")
            documents.append(flattened_text)
            metadatas.append(
                {
                    "page_file": entry.get("table_id"),
                    "page_number": entry.get("page"),
                    "source": "table",
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

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("CHROMA_OPENAI_API_KEY")
    if not api_key:
        return None

    embedding_function = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
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
        "table_count": 0,
        "pages_multi_column": [],
        "used_ocr": False,
        "frequency_hz_candidates": [],
        "vector_index": None,
        "tables": [],
        "document_metadata": {
            "title": None,
            "sections": [],
            "references_start_page": None,
        },
        "paths": {
            "bundle_dir": os.path.abspath(bundle_dir),
            "pages_dir": os.path.join(os.path.abspath(bundle_dir), "pages"),
            "figures_dir": os.path.join(os.path.abspath(bundle_dir), "figures"),
            "tables_dir": os.path.join(os.path.abspath(bundle_dir), "tables"),
            "manifest_path": os.path.join(os.path.abspath(bundle_dir), "manifest.json"),
            "figures_index": os.path.join(
                os.path.abspath(bundle_dir), "figures", "index.json"
            ),
            "tables_index": os.path.join(
                os.path.abspath(bundle_dir), "tables", "index.json"
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
    tables_dir = os.path.join(bundle_dir, "tables")
    ensure_dir(pages_dir)
    ensure_dir(figures_dir)
    ensure_dir(tables_dir)

    manifest = _empty_manifest(file_hash, pdf_abs, bundle_dir)
    manifest["status"] = "ready"
    manifest["cache"] = "miss"

    pages_multi_col: List[int] = []
    freq_candidates: List[float] = []
    figure_entries: List[FigureEntry] = []
    table_entries: List[TableEntry] = []
    page_infos: List[Dict[str, Any]] = []
    header_counter: Counter[str] = Counter()
    footer_counter: Counter[str] = Counter()

    try:
        doc = fitz.open(pdf_abs)
        manifest["page_count"] = doc.page_count
        toc_entries = doc.get_toc() or []
        metadata_title = (doc.metadata or {}).get("title") if hasattr(doc, "metadata") else None
        fig_index = 1
        table_index = 1

        for i in range(doc.page_count):
            page_no = i + 1
            page = doc.load_page(i)
            text, text_blocks = _extract_page_text(page)
            raw_lines = text.splitlines()

            if raw_lines:
                for line in raw_lines[:HEADER_SAMPLE_LINES]:
                    if line:
                        header_counter[line] += 1
                for line in raw_lines[-FOOTER_SAMPLE_LINES:]:
                    if line:
                        footer_counter[line] += 1

            if detect_multicol(text_blocks):
                pages_multi_col.append(page_no)

            page_dict = page.get_text("dict")
            line_entries: List[Dict[str, Any]] = []
            span_sizes: List[float] = []
            for block in page_dict.get("blocks", []):
                if block.get("type", 0) != 0:
                    continue
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                    text_line = "".join(span.get("text", "") for span in spans).strip()
                    if not text_line:
                        continue
                    sizes = [float(span.get("size", 0.0)) for span in spans if "size" in span]
                    if sizes:
                        span_sizes.extend(sizes)
                    bbox_values = [
                        span.get("bbox")
                        for span in spans
                        if isinstance(span.get("bbox"), (list, tuple)) and len(span["bbox"]) >= 2
                    ]
                    y0 = min(b[1] for b in bbox_values) if bbox_values else 0.0
                    line_entries.append(
                        {
                            "text": text_line,
                            "size": max(sizes) if sizes else 0.0,
                            "y0": y0,
                        }
                    )

            median_size = median(span_sizes) if span_sizes else 0.0
            page_infos.append(
                {
                    "page_no": page_no,
                    "raw_lines": raw_lines,
                    "lines_with_sizes": line_entries,
                    "median_font_size": median_size,
                    "blocks": text_blocks,
                }
            )

            figs, fig_index = _extract_figures(page, page_no, figures_dir, text_blocks, fig_index)
            figure_entries.extend(figs)

        pdfplumber_tables, table_index = _extract_tables_pdfplumber(
            pdf_abs, tables_dir, table_index, table_entries
        )
        table_entries.extend(pdfplumber_tables)

        header_lines = _derive_common_lines(header_counter, doc.page_count)
        footer_lines = _derive_common_lines(footer_counter, doc.page_count)
        references_started = False
        references_page: Optional[int] = None

        for info in page_infos:
            cleaned_lines, references_started, references_triggered = _clean_page_lines(
                info["raw_lines"], header_lines, footer_lines, references_started
            )
            if references_triggered and references_page is None:
                references_page = info["page_no"]
            info["cleaned_lines"] = cleaned_lines
            cleaned_text = "\n".join(cleaned_lines).strip()

            page_path = os.path.join(pages_dir, f"page_{info['page_no']:04d}.txt")
            with open(page_path, "w", encoding="utf-8") as fh:
                fh.write(cleaned_text)

            if cleaned_text:
                freq_candidates.extend(normalize_frequencies(cleaned_text))

        for entry in figure_entries:
            if entry.caption:
                freq_candidates.extend(normalize_frequencies(entry.caption))

        title, sections = _extract_title_and_sections(page_infos, toc_entries, metadata_title)

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

        tables_dir_path = Path(tables_dir)
        table_files = sorted(tables_dir_path.glob("table_*.json"))
        tables_index_path = os.path.join(tables_dir, "index.json")
        tables_summary: List[Dict[str, Any]] = []
        index_entries: List[Dict[str, Any]] = []
        for table_file in table_files:
            try:
                payload = json.loads(table_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            rows = payload.get("rows", [])
            source = payload.get("source")
            if isinstance(rows, list) and source == "pdfplumber_text":
                trimmed: List[List[str]] = []
                for idx_row, row in enumerate(rows):
                    if not isinstance(row, list) or not row:
                        continue
                    cells = row[:2]
                    has_digit = any(ch.isdigit() for ch in cells[-1]) if cells else False
                    if idx_row > 0 and not has_digit:
                        break
                    trimmed.append(cells)
                    if len(trimmed) >= 12:
                        break
                if trimmed:
                    rows = trimmed
                    payload["rows"] = rows
                    table_file.write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
            preview_rows = rows[:3] if isinstance(rows, list) else []
            n_rows = len(rows) if isinstance(rows, list) else 0
            n_cols = max((len(row) for row in rows if isinstance(row, list)), default=0)
            entry = {
                "table_id": payload.get("table_id"),
                "page": payload.get("page"),
                "bbox": payload.get("bbox"),
                "path": os.path.relpath(table_file, bundle_dir),
                "n_rows": n_rows,
                "n_cols": n_cols,
                "preview_rows": preview_rows,
                "source": payload.get("source"),
                "title": payload.get("title"),
                "legend": payload.get("legend"),
            }
            index_entries.append(entry)
            tables_summary.append(entry)
        with open(tables_index_path, "w", encoding="utf-8") as tables_file:
            json.dump(index_entries, tables_file, ensure_ascii=False, indent=2)

        manifest.update(
            {
                "figure_count": len(figure_entries),
                "table_count": len(table_files),
                "pages_multi_column": pages_multi_col,
                "frequency_hz_candidates": unique_freqs,
                "tables": tables_summary,
                "document_metadata": {
                    "title": title,
                    "sections": sections,
                    "references_start_page": references_page,
                },
                "paths": {
                    "bundle_dir": os.path.abspath(bundle_dir),
                    "pages_dir": os.path.abspath(pages_dir),
                    "figures_dir": os.path.abspath(figures_dir),
                    "tables_dir": os.path.abspath(tables_dir),
                    "manifest_path": os.path.abspath(manifest_path),
                    "figures_index": os.path.abspath(figures_index_path),
                    "tables_index": os.path.abspath(tables_index_path),
                },
            }
        )

        vector_meta: Optional[Dict[str, Any]] = None
        try:
            vector_meta = _build_vector_index(bundle_dir, manifest)
        except Exception as exc:
            manifest.setdefault("warnings", []).append(f"vector_index_error:{exc!r}")
            vector_meta = None
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
