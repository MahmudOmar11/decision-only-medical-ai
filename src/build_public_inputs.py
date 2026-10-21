"""Rebuild the published-source inputs from the original datasets.

The repository stores no third-party text. For each input, data/inputs_public.csv.gz holds the source record,
our edits to that record's text (character spans and the text we added) and the SHA-256 of the final input.
This script reads the downloaded sources, applies the edits and checks every hash.

    python src/build_public_inputs.py \
        --acuitybench acuitybench_transformed.csv --ramaswamy DataExpanded_FINAL.csv \
        --levine media-1.docx --sen2med dataset.csv --helpmed HELPMed/ \
        --out inputs_public.jsonl

Any source can be omitted; its inputs are skipped. See data/SOURCES.md for where to get each file.
"""
import argparse
import csv
import gzip
import hashlib
import json
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
csv.field_size_limit(sys.maxsize)


def acuitybench(path):
    """Text of the single user message in `conversational_prompt`, keyed dataset:source_id."""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            msgs = [m["content"] for m in json.loads(r["conversational_prompt"]) if m.get("role") == "user"]
            if len(msgs) == 1:
                out[f"{r['dataset']}:{r['source_id']}"] = msgs[0]
    return out


def ramaswamy(path):
    """`prompt_text`, keyed case_id:variant_code."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {f"{r['case_id']}:{r['variant_code']}": r["prompt_text"] for r in csv.DictReader(f)}


def levine(path):
    """Vignettes from eTable 2 of the supplement: age and sex, current problem, then the other details."""
    w = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    root = ET.fromstring(zipfile.ZipFile(path).read("word/document.xml"))
    text = lambda p: "".join(t.text or "" for t in p.iter(w + "t"))
    table = next(t for t in root.iter(w + "tbl")
                 if [text(c) for c in t.find(w + "tr").findall(w + "tc")][:1] == ["Case #"])
    out = {}
    for tr in table.findall(w + "tr")[1:]:
        cells = [[text(p) for p in tc.findall(w + "p")] for tc in tr.findall(w + "tc")]
        if len(cells) < 5 or not cells[0][0].strip().isdigit():
            continue
        problem = " ".join(cells[3]).strip()
        details = [d.strip() for d in cells[4] if d.strip()]
        out[cells[0][0].strip()] = ". ".join([details[0], problem] + details[1:]) + "."
    return out


def sen2med(path):
    """`clinical_context` of the baseline AskDocs version, keyed context_id."""
    with open(path, newline="", encoding="utf-8") as f:
        return {r["context_id"]: r["clinical_context"] for r in csv.DictReader(f)
                if r["dataset"] == "askdocs" and r["perturbation"] == "baseline"}


def helpmed(repo):
    """Scenario fields joined by blank lines, keyed file:index (original and third-person files)."""
    out = {}
    for name in ["data/main/scenarios.jsonl", "data/scenarios_3p.jsonl"]:
        text, dec, i = (Path(repo) / name).read_text(encoding="utf-8"), json.JSONDecoder(), 0
        while text.strip():
            obj, end = dec.raw_decode(text.lstrip())
            text = text.lstrip()[end:]
            out[f"{Path(name).name}:{i}"] = "\n\n".join(
                obj[k] for k in ["title", "description", "heading_1", "body_1", "heading_2", "body_2"])
            i += 1
    return out


LOADERS = {"acuitybench": acuitybench, "ramaswamy": ramaswamy, "levine": levine,
           "sen2med": sen2med, "helpmed": helpmed}


def apply(text, edits):
    for start, end, new in sorted(edits, reverse=True):
        text = text[:start] + new + text[end:]
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in LOADERS:
        ap.add_argument(f"--{name}")
    ap.add_argument("--table", default=ROOT / "data/inputs_public.csv.gz")
    ap.add_argument("--out", default="inputs_public.jsonl")
    a = ap.parse_args()
    sources = {n: f(getattr(a, n)) for n, f in LOADERS.items() if getattr(a, n)}
    if not sources:
        ap.error("give at least one source")
    done, missing, bad = 0, 0, []
    with gzip.open(a.table, "rt", newline="", encoding="utf-8") as f, open(a.out, "w", encoding="utf-8") as out:
        for r in csv.DictReader(f):
            if r["source"] not in sources:
                continue
            base = sources[r["source"]].get(r["record"])
            if base is None:
                missing += 1
                continue
            text = apply(base, json.loads(r["edits"]))
            if hashlib.sha256(text.encode("utf-8")).hexdigest() != r["sha256"]:
                bad.append(r["id"])
                continue
            out.write(json.dumps({"id": r["id"], "set": r["set"], "question_key": "triage_level", "text": text},
                                 ensure_ascii=False) + "\n")
            done += 1
    print(f"{done} inputs written to {a.out}; {missing} source records not found; {len(bad)} hash mismatches")
    if bad:
        print("mismatched:", ", ".join(bad[:10]), file=sys.stderr)
    sys.exit(1 if bad or missing else 0)


if __name__ == "__main__":
    main()
