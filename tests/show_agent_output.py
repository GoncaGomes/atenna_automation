from __future__ import annotations

"""Utility script to run the antenna extraction agent and print structured output."""

import argparse
import json
from pathlib import Path

from antenna_automation.agent import run_information_extraction
from antenna_automation.ingest import DEFAULT_STORE_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="Path to the PDF article")
    parser.add_argument(
        "--store-dir",
        type=Path,
        default=Path(DEFAULT_STORE_DIR),
        help="Cache directory where ingestion bundles are stored",
    )
    parser.add_argument(
        "--max-turns", type=int, default=30, help="Maximum number of agent turns"
    )
    args = parser.parse_args()

    result = run_information_extraction(
        str(args.pdf), str(args.store_dir), max_turns=args.max_turns
    )

    if hasattr(result, "model_dump_json"):
        print(result.model_dump_json(indent=2))
    elif hasattr(result, "model_dump"):
        print(json.dumps(result.model_dump(), indent=2))
    else:
        print(result)


if __name__ == "__main__":
    main()
