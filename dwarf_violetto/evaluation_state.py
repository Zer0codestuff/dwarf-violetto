"""Atomic evaluation records and strict resume identity checks."""
import json


def save(destination, result):
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(destination)


def validate_resume(previous, current):
    for key in ("settings", "sha256", "packages"):
        if key not in previous or previous[key] != current[key]:
            raise ValueError("Cannot resume with different settings, code, packages, weights, tokenizer or fixtures")
    ids = [row["id"] for row in previous["cases"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Saved evaluation contains duplicate case IDs")
