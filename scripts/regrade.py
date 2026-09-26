"""Regrade saved completions against independently checked local answer keys."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    args = p.parse_args()
    source = Path("data/math_checks.json")
    keys = {x["id"]: x["answer"] for x in json.loads(source.read_text())}
    import re
    for file in args.files:
        path = Path(file)
        data = json.loads(path.read_text())
        for row in data["cases"]:
            row["expected"] = keys[row["id"]]
            answer = row["answer"]
            if answer is not None:
                answer = re.sub(r"\\[dt]?frac\{(-?\d+)\}\{(\d+)\}", r"\1/\2", answer)
                answer = re.sub(r"[\s,$]", "", answer)
            row["correct"] = row["finish"] == "stop" and answer == row["expected"]
        data["correct"] = sum(x["correct"] for x in data["cases"])
        data["answer_key_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        data["regraded"] = True
        path.write_text(json.dumps(data, indent=2)+"\n")
        print(file, data["correct"], "/", len(data["cases"]))


if __name__ == "__main__":
    main()
