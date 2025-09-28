"""Command-line entry point for running the antenna information extractor."""

from __future__ import annotations

import argparse
from pathlib import Path

from antenna_automation.agent import run_information_extraction
from antenna_automation.ingest import DEFAULT_STORE_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract antenna information from a PDF article")
    parser.add_argument("pdf", type=Path, help="Path to the local PDF file")
    parser.add_argument(
        "--store-dir",
        type=Path,
        default=Path(DEFAULT_STORE_DIR),
        help="Directory where cached ingestion bundles are stored",
    )
    parser.add_argument(
        "--max-turns", type=int, default=30, help="Maximum agent turns for the extraction run"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_information_extraction(str(args.pdf), str(args.store_dir), max_turns=args.max_turns)
    print(result)


if __name__ == "__main__":
    main()
