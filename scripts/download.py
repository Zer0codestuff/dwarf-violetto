"""Download the pinned original checkpoint without executing remote code."""
import json
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
REVISION = "9402e422e87b220507963cd42c997459c1e87b43"

if __name__ == "__main__":
    dest = ROOT / "models" / "original"
    snapshot_download(
        "paradigma-inc/limite-1b-violetto", revision=REVISION,
        local_dir=dest,
        allow_patterns=["*.json", "*.safetensors", "*.jinja", "README.md"],
    )
    (dest / "source.json").write_text(json.dumps({
        "repository": "paradigma-inc/limite-1b-violetto", "revision": REVISION,
    }, indent=2) + "\n")
    print(dest)
