"""Run one system over an input file, resumably, writing one JSON line per input.

  python src/run.py SYSTEM INPUTS OUTPUT [--limit N] [--workers N] [--dry-run]

SYSTEM: jev, decisions, llm, jev_team or decisions_team. INPUTS: data/inputs_controlled.jsonl.gz or the file written
by src/build_public_inputs.py. Jev answers once per input; Decisions three times (probabilities averaged); the LLM
three times (majority). --dry-run prints what would be sent for the first input and makes no calls.
"""
import argparse
import gzip
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import models as M
import workflow as W

ROOT = Path(__file__).resolve().parent.parent
TRIAGE = json.loads((ROOT / "data/triage_question.json").read_text())
THRESHOLDS = json.loads((ROOT / "data/thresholds.json").read_text())


def read(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for r in rows:
        r.setdefault("question", TRIAGE)
    return rows


def mean(vectors):
    if not vectors or any(v is None for v in vectors):
        return None
    return {k: sum(v[k] for v in vectors) / len(vectors) for k in vectors[0]}


def original(model, r):
    calls = 1 if model == "jev" else 3
    return mean([M.DECISION_MODELS[model](r["text"], r["question_key"], r["question"]) for _ in range(calls)])


def run_one(system, r):
    qk, q = r["question_key"], r["question"]
    if system in ("jev", "decisions"):
        probs = original(system, r)
        return {"decision": M.decide(probs, qk, q), "probs": probs}
    if system == "llm":
        picks = [M.llm(r["text"], q) for _ in range(3)]
        return {"decision": W.majority(picks, q), "picks": picks}
    model = system.removesuffix("_team")
    out = W.team(r, model, THRESHOLDS[model], original=original(model, r))
    return out or {"decision": None}


def dry_run(system, r):
    model = system.removesuffix("_team")
    if model == "llm":
        print(json.dumps(M.llm_messages(r["text"], r["question"]), indent=1))
        return
    print(json.dumps(M.REQUEST_BODIES[model](r["text"], r["question_key"], r["question"]), indent=1))
    if system.endswith("_team"):
        copies = W.deletions(r["text"])
        print(f"\n{len(copies)} deletion copies; first passage: {copies[0][0]!r}" if copies else "\nno deletion copies")
        print(json.dumps(M.finding_messages(copies[0][0] if copies else "..."), indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("system", choices=["jev", "decisions", "llm", "jev_team", "decisions_team"])
    ap.add_argument("inputs")
    ap.add_argument("output")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    rows = read(a.inputs)[:a.limit]
    if a.dry_run:
        return dry_run(a.system, rows[0])
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for line in out.open()} if out.exists() else set()
    pending = [r for r in rows if r["id"] not in done]
    with out.open("a") as f, ThreadPoolExecutor(a.workers) as pool:
        for r, res in zip(pending, pool.map(lambda r: run_one(a.system, r), pending)):
            f.write(json.dumps({"id": r["id"], **res}) + "\n")
            f.flush()
    print(f"{len(pending)} new, {len(done)} already in {out}")


if __name__ == "__main__":
    main()
