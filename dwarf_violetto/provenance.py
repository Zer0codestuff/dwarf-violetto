"""Hash code, fixtures, tokenizer and weights without exposing local paths."""
import hashlib
from importlib.metadata import version
from pathlib import Path
from .checkpoint import checkpoint_files

ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def public_path(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return f"checkpoint/{path.parent.name}/{path.name}"


def script_hashes(paths):
    return {public_path(p): file_sha256(p) for p in paths}


def provenance(model):
    paths = sorted((ROOT / "dwarf_violetto").glob("*.py"))
    paths += [ROOT / "data/math_checks.json", ROOT / "requirements.lock"]
    paths += checkpoint_files(model)
    return script_hashes(paths)


def package_versions():
    return {name: version(name) for name in ("mlx", "mlx-lm", "transformers", "tokenizers", "numpy")}
