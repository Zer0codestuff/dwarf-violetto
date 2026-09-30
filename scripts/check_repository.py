"""Check distributable files, saved answer grading and result consistency."""
import json
import math
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dwarf_violetto.grading import is_correct, boxed


def reject_constant(value):
    raise ValueError(f"Non-finite JSON number: {value}")


def main():
    files = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
    ).decode().split("\0")
    names = sorted(set(name for name in files if name))
    forbidden = ("models/", ".venv/", ".cache/", "research/", ".publication/")
    results = {}
    for name in names:
        path = ROOT / name
        if name.startswith(forbidden) or path.suffix in {".npz", ".safetensors", ".pem", ".key"}:
            raise ValueError(f"Local-only file would be published: {name}")
        if path.stat().st_size > 5_000_000:
            raise ValueError(f"Unexpected large source artifact: {name}")
        text = path.read_text()
        if "\u2014" in text:
            raise ValueError(f"Em dash in project text: {name}")
        if name.endswith(".json"):
            data = json.loads(text, parse_constant=reject_constant)
            if re.search(r"/(?:Users|home)/[^/\s]+/", text):
                raise ValueError(f"Personal filesystem path: {name}")
            if name.startswith("results/"):
                results[path.name] = data
    fixtures = json.loads((ROOT / "data/math_checks.json").read_text())
    answers = {case["id"]: case["answer"] for case in fixtures}
    for name in ("bf16-math-eos.json", "q8-math-eos.json"):
        evaluation = results[name]
        rows = evaluation["cases"]
        if {row["id"] for row in rows} != set(answers) or len(rows) != len(answers):
            raise ValueError(f"Incomplete or duplicate EOS evaluation: {name}")
        for row in rows:
            actual = is_correct(row["text"], answers[row["id"]], row["finish"])
            if (actual != row["correct"] or row["expected"] != answers[row["id"]] or
                    row["answer"] != boxed(row["text"])):
                raise ValueError(f"Saved grading mismatch: {name}, {row['id']}")
            if row["finish"] != "stop" or row["stop_token_id"] not in {151643, 151645}:
                raise ValueError(f"Unfinished EOS evaluation: {name}, {row['id']}")
        if evaluation["correct"] != sum(row["correct"] for row in rows):
            raise ValueError(f"Incorrect grading total: {name}")
    for name, data in results.items():
        if name.startswith("final-"):
            for row in data["runs"]:
                if row["generated_tokens"] != len(row["tokens"]) or row["decode_tps"] <= 0:
                    raise ValueError(f"Invalid benchmark token accounting: {name}")
            for key, median in data["median"].items():
                expected = statistics.median(row[key] for row in data["runs"])
                if not math.isclose(median, expected, rel_tol=1e-9, abs_tol=1e-12):
                    raise ValueError(f"Invalid stored benchmark median: {name}, {key}")
    for name in ("q8-fidelity-final.json", "q6-fidelity-final.json", "q4-hq-fidelity-final.json", "fusion-fidelity.json"):
        data = results[name]
        positions = sum(row["positions"] for row in data["cases"])
        if positions != data["positions"] or positions != 359:
            raise ValueError(f"Invalid fidelity position count: {name}")
        for key in ("kl_mean", "top1_agreement"):
            expected = sum(row[key] * row["positions"] for row in data["cases"]) / positions
            if not math.isclose(expected, data[key], rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f"Invalid weighted fidelity metric: {name}, {key}")
    correctness = results["decode-candidate-correctness.json"]
    if not correctness.get("all_exact") or sum(row["positions"] for row in correctness["checks"]) != 960:
        raise ValueError("Incomplete decode-candidate validation")
    for row in correctness["checks"]:
        if not row["prefill_exact"] or row["decode_max_abs"] != 0 or not row["cache_exact"]:
            raise ValueError("Decode candidate changed logits or cache")
    summary = results["decode-optimization-summary.json"]
    for name in summary["comparisons"] + summary["excluded_speed_files"]:
        if name not in results:
            raise ValueError(f"Missing referenced result: {name}")
    print(f"Checked {len(names)} source artifacts, {len(results)} JSON results and all 24 saved EOS answers.")


if __name__ == "__main__":
    main()
