"""CLI smoke test that exercises the ingestion pipeline and validates table exports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from antenna_automation.ingest import DEFAULT_STORE_DIR, ensure_ingested  # noqa: E402


def _parse_expected(header: str) -> List[str]:
    return [part.strip() for part in header.split("|") if part.strip()]


def _find_table(manifest: dict, table_id: str) -> dict | None:
    for entry in manifest.get("tables", []):
        if entry.get("table_id") == table_id:
            return entry
    return None


def _load_rows(bundle_dir: Path, table_entry: dict) -> tuple[List[List[str]], dict]:
    rel_path = table_entry.get("path")
    if not rel_path:
        return [], {}
    table_path = bundle_dir / rel_path
    if not table_path.exists():
        raise FileNotFoundError(f"Table JSON not found at {table_path}")
    payload = json.loads(table_path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    cleaned: List[List[str]] = []
    for row in rows:
        if isinstance(row, list):
            cleaned.append([str(cell) for cell in row])
    return cleaned, payload


def _assert_header(rows: List[List[str]], expected_header: Iterable[str]) -> None:
    if not rows:
        raise AssertionError("Table contains no rows")
    header = rows[0]
    expected = list(expected_header)
    if header != expected:
        raise AssertionError(f"Header mismatch.\nExpected: {expected}\nGot:      {header}")


def _assert_row_count(rows: List[List[str]], min_rows: int) -> None:
    if len(rows) < min_rows:
        raise AssertionError(f"Expected at least {min_rows} rows, found {len(rows)}.")


def _assert_column_consistency(rows: List[List[str]], expected_cols: int) -> None:
    for idx, row in enumerate(rows, start=1):
        if len(row) != expected_cols:
            raise AssertionError(
                f"Row {idx} has {len(row)} columns, expected {expected_cols}: {row}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="Path to the PDF to ingest")
    parser.add_argument(
        "--store-dir",
        type=Path,
        default=Path(DEFAULT_STORE_DIR),
        help="Directory to cache ingestion bundles",
    )
    parser.add_argument(
        "--table-id",
        default="table_002",
        help="Table identifier to validate (default: table_002)",
    )
    parser.add_argument(
        "--expected-header",
        default="Parameter|X-axis|Y-axis|Z-axis",
        help="Pipe-separated header expected for the table",
    )
    parser.add_argument(
        "--expected-legend",
        default="",
        help="Expected legend/caption text for the table (leave blank to skip check)",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=3,
        help="Minimum number of rows (including header) the table should contain",
    )
    args = parser.parse_args()

    manifest = ensure_ingested(str(args.pdf), str(args.store_dir))
    status = manifest.get("status")
    if status != "ready":
        print(f"Ingestion failed (status={status}).", file=sys.stderr)
        return 1

    table_entry = _find_table(manifest, args.table_id)
    if not table_entry:
        print(f"Table {args.table_id!r} not found in manifest.", file=sys.stderr)
        return 1

    bundle_dir = Path(manifest.get("paths", {}).get("bundle_dir", args.store_dir))
    rows, table_payload = _load_rows(bundle_dir, table_entry)

    try:
        expected_header = _parse_expected(args.expected_header)
        _assert_row_count(rows, args.min_rows)
        _assert_header(rows, expected_header)
        _assert_column_consistency(rows, len(expected_header))
        if args.expected_legend:
            manifest_legend = table_entry.get("legend")
            if manifest_legend != args.expected_legend:
                raise AssertionError(
                    f"Legend mismatch in manifest.\nExpected: {args.expected_legend!r}\n"
                    f"Got:      {manifest_legend!r}"
                )
            table_legend = table_payload.get("legend")
            if table_legend != args.expected_legend:
                raise AssertionError(
                    f"Legend mismatch in table payload.\nExpected: {args.expected_legend!r}\n"
                    f"Got:      {table_legend!r}"
                )
    except AssertionError as exc:  # pragma: no cover - CLI validation path
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 1

    print(f"Table {args.table_id} validated successfully.")
    print(f"- Bundle directory: {bundle_dir}")
    print(f"- Header: {' | '.join(rows[0])}")
    print(f"- Row count: {len(rows)}")
    print(f"- Legend: {table_entry.get('legend')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
