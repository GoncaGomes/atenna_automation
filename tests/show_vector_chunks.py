"""Utility script to inspect random chunks from the cached Chroma vector store."""

import argparse
import json
import random
from pathlib import Path
from typing import Sequence

from antenna_automation.ingest import DEFAULT_STORE_DIR, ensure_ingested

try:
    import chromadb
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit("chromadb is required to inspect the vector store") from exc


def _sample_ids(all_ids: Sequence[str], sample_size: int, seed: int) -> list[str]:
    if not all_ids:
        return []
    rng = random.Random(seed)
    return rng.sample(list(all_ids), min(sample_size, len(all_ids)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="Path to the PDF that has been ingested")
    parser.add_argument(
        "--store-dir",
        type=Path,
        default=Path(DEFAULT_STORE_DIR),
        help="Cache directory where ingestion bundles are stored",
    )
    parser.add_argument(
        "--samples", type=int, default=3, help="Number of random chunks to print"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Deterministic seed for sampling"
    )
    args = parser.parse_args()

    manifest = ensure_ingested(str(args.pdf), str(args.store_dir))
    vector_meta = manifest.get("vector_index")
    if not vector_meta:
        raise SystemExit("No vector index metadata found. Run ingestion first.")

    store_path = Path(vector_meta["persist_dir"])
    meta_path = store_path / "meta.json"
    if not meta_path.exists():
        raise SystemExit(
            "Vector store meta.json not found. Delete the cache folder and ingest again."
        )

    print("Using vector store metadata:\n" + json.dumps(vector_meta, indent=2))

    client = chromadb.PersistentClient(path=str(store_path))
    collection = client.get_collection(vector_meta["collection_name"])

    id_result = collection.get(include=["ids"])
    all_ids = id_result.get("ids", [])
    sample_ids = _sample_ids(all_ids, args.samples, args.seed)
    if not sample_ids:
        raise SystemExit("The vector store is empty; no chunks available to sample.")

    records = collection.get(ids=sample_ids, include=["documents", "metadatas"])

    for idx, (doc, metadata) in enumerate(
        zip(records.get("documents", []), records.get("metadatas", [])), start=1
    ):
        page = metadata.get("page_number")
        page_file = metadata.get("page_file")
        print("\n=== Sample {} | page {} ({}) ===".format(idx, page, page_file))
        print(doc.strip())


if __name__ == "__main__":
    main()
