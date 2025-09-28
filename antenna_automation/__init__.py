"""Core package for antenna automation workflow."""

from .schema import Antenna
from .agent import create_information_extraction_agent
from .ingest import ensure_ingested

__all__ = [
    "Antenna",
    "create_information_extraction_agent",
    "ensure_ingested",
]
