"""Identify the exact code, model config and fixtures used for each run."""
import hashlib
from pathlib import Path


def provenance(model):
    paths = sorted(Path("dwarf_violetto").glob("*.py"))
    paths += [Path("data/math_checks.json"), Path(model) / "config.json"]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
