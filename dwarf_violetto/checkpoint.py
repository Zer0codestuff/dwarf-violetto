"""Validate local checkpoints and publish conversions only after completion."""
import json
import struct
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


def checkpoint_files(folder):
    folder = Path(folder)
    config = json.loads((folder / "config.json").read_text())
    if config.get("model_type") != "limite" or config.get("model_file"):
        raise ValueError("Use a Limite checkpoint without external model code")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        json.loads((folder / name).read_text())
    index = folder / "model.safetensors.index.json"
    if index.exists():
        names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        names = sorted(p.name for p in folder.glob("*.safetensors"))
    if not names:
        raise ValueError("Checkpoint has no Safetensors weights")
    weights = []
    for name in names:
        if Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError("Invalid checkpoint shard name")
        path = folder / name
        size = path.stat().st_size
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise ValueError(f"Incomplete Safetensors file: {name}")
            header_size = struct.unpack("<Q", prefix)[0]
            if not 2 <= header_size <= min(16_000_000, size - 8):
                raise ValueError(f"Invalid Safetensors header: {name}")
            header = json.loads(handle.read(header_size))
        payload_size = size - 8 - header_size
        offsets = [v["data_offsets"] for k, v in header.items() if k != "__metadata__"]
        if (not offsets or max(end for _, end in offsets) != payload_size or
                any(not 0 <= start <= end <= payload_size for start, end in offsets)):
            raise ValueError(f"Incomplete Safetensors payload: {name}")
        weights.append(path)
    metadata = [p for p in sorted(folder.iterdir())
                if p.is_file() and p.suffix in {".json", ".jinja"}]
    return metadata + weights


@contextmanager
def staged_checkpoint(destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "checkpoint"
        staging.mkdir()
        yield staging
        checkpoint_files(staging)
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite {destination}")
        staging.rename(destination)
