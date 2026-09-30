"""Download the pinned original checkpoint without executing remote code."""
import json
import hashlib
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
REVISION = "9402e422e87b220507963cd42c997459c1e87b43"
WEIGHT_SHA256 = "9dc7e89e30acf473629f98ac3ade535fb05cc044fe7e399b78d0bc957283a5f5"

if __name__ == "__main__":
    dest = ROOT / "models" / "original"
    snapshot_download(
        "paradigma-inc/limite-1b-violetto", revision=REVISION,
        local_dir=dest,
        allow_patterns=["*.json", "*.safetensors", "*.jinja", "README.md", "LICENSE", "NOTICE"],
    )
    with (dest / "model.safetensors").open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual != WEIGHT_SHA256:
        raise SystemExit("Checkpoint hash does not match the pinned official weights")
    (dest / "source.json").write_text(json.dumps({
        "repository": "paradigma-inc/limite-1b-violetto", "revision": REVISION,
        "weights_sha256": actual,
    }, indent=2) + "\n")
    print(dest)
