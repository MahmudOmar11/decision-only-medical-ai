"""The paired workflow (a decision model with the LLM), as frozen for the study.

1. The decision model answers the original message and every valid deletion copy: each paragraph of two or more
   sentences, each sentence, and, where a sentence cannot be removed, its function-word spans. A copy is valid when it
   only deletes words, keeps every number and keeps the age and sex phrase.
2. For each deletion that changes the answer, the LLM is asked three times whether the passage reports a patient
   finding; a majority "no" makes the deletion eligible.
3. The eligible deletion with the largest influence (change in cumulative probability across the urgency boundary it
   crosses; ties to the shorter passage) is accepted only if its influence exceeds the frozen threshold for the
   question and original answer level (data/thresholds.json). Otherwise the original answer stands.
4. Emergency check, four-option triage only: if the answer is not the ER, the decision model's original probability
   for "ER now" is at least 0.30, and all three LLM answers on the selected text are the ER, the answer becomes the ER.
"""
import re
from collections import Counter

import models as M

LEVEL = {"monitor_at_home": 0, "see_doctor_within_weeks": 1, "see_doctor_within_24_48_hours": 2, M.ER: 3,
         "false": 0, "true": 1, "not_indicated": 0, "recommended": 1}

_SENT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[A-Z])")
_AGE = re.compile(r"(\d+-year-old (?:man|woman|patient))")
_WORD = re.compile(r"[A-Za-z0-9]+(?:[.'/][A-Za-z0-9]+)*")
_NUM = re.compile(r"\d+(?:[.,]\d+)*")
_TOKEN = re.compile(r"\S+")
FUNCTION_WORDS = frozenset("""who whom whose which that with without and but or as after before since while when
because although though despite for from of to in on at by about during under over into through following""".split())


def sentences(text):
    return [s.strip() for line in text.split("\n") if line.strip() for s in _SENT.split(line.strip()) if s.strip()]


def tidy(text):
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +([,.;:!?])", r"\1", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def remove(text, piece):
    i = text.find(piece)
    if i < 0:
        return None
    lines = [ln for ln in (text[:i] + text[i + len(piece):]).split("\n") if ln.strip() not in ("-", "- :")]
    return tidy(re.sub(r"\n\n\n+", "\n\n", "\n".join(lines)))


def valid(copy, original):
    if not copy or not copy.strip() or copy == original:
        return False
    it = iter(_WORD.findall(original))
    if not all(any(w == b for b in it) for w in _WORD.findall(copy)):
        return False
    if not set(_NUM.findall(original)) <= set(_NUM.findall(copy)):
        return False
    m = _AGE.search(original)
    return not (m and m.group(1) not in copy)


def _spans(sentence):
    """Function-word runs of a sentence and each adjacent pair, skipping spans with digits or the age phrase."""
    toks = [(m.group(), m.start(), m.end()) for m in _TOKEN.finditer(sentence)]
    starts = [i for i, (t, _, _) in enumerate(toks) if t.lower().strip(",;:.()\"'") in FUNCTION_WORDS]
    runs = []
    for i in starts:
        j = i
        while j + 1 < len(toks) and (j + 1) not in starts and not re.search(r"[,;:.()]$", toks[j][0]):
            j += 1
        runs.append(sentence[toks[i][1]:toks[j][2]].rstrip(",;:."))
    if len(runs) < 2:
        return []
    cands = runs + [sentence[sentence.find(a):sentence.find(b, sentence.find(a)) + len(b)]
                    for a, b in zip(runs, runs[1:]) if sentence.find(b, sentence.find(a)) > 0]
    out = []
    for span in cands:
        if not (re.search(r"\d", span) or _AGE.search(span) or span in out or len(span.split()) < 2):
            out.append(span)
    return out


def deletions(text):
    """(passage, copy) for every valid deletion copy of a message."""
    out = []
    for sent in sentences(text):
        copy = remove(text, sent)
        if valid(copy, text):
            out.append((sent, copy))
            continue
        for span in _spans(sent):
            copy = tidy(text.replace(sent, tidy(sent.replace(span, "", 1)), 1))
            if valid(copy, text):
                out.append((span, copy))
    for block in re.split(r"\n[ \t]*\n", text):
        block = block.strip()
        if block and len(sentences(block)) >= 2:
            copy = remove(text, block)
            if valid(copy, text):
                out.append((block, copy))
    return out


_probes, _votes = {}, {}  # identical deletion copies and passages are asked once, as in the study


def _probe(ask, model, text, qk, q):
    key = (model, text, qk)
    if key not in _probes:
        _probes[key] = ask(text, qk, q)
    return _probes[key]


def _finding_votes(piece):
    if piece not in _votes:
        _votes[piece] = [M.is_finding(piece) for _ in range(3)]
    return _votes[piece]


def _above(probs, keys, level):
    return sum(float(probs.get(k, 0.0)) for k in keys if LEVEL[k] >= level)


def team(item, model, thresholds, original=None, llm_reads=None):
    """Team answer for one input. `original`: the decision model's probabilities for the unaltered message (computed
    if not given); `llm_reads`: the LLM's three answers to the unaltered message (reused when nothing is removed).
    Returns None when a required call is missing (for example a refusal)."""
    ask = M.DECISION_MODELS[model]
    text, qk, q = item["text"], item["question_key"], item["question"]
    keys = M.options(q)
    vec0 = original if original is not None else ask(text, qk, q)
    if vec0 is None:
        return None
    d0 = M.decide(vec0, qk, q)
    eligible = []
    for piece, copy in deletions(text):
        vec = _probe(ask, model, copy, qk, q)
        if vec is None:
            return None
        d = M.decide(vec, qk, q)
        if d == d0:
            continue
        votes = _finding_votes(piece)
        if None in votes:
            return None
        if votes.count("true") / 3 >= 0.5:
            continue
        hi = max(LEVEL[d0], LEVEL[d])
        eligible.append((abs(_above(vec0, keys, hi) - _above(vec, keys, hi)), piece, copy, vec))
    eligible.sort(key=lambda x: (-x[0], len(x[1])))
    limit = thresholds[item["set"]][f"{qk}|{LEVEL[d0]}"]
    chosen = eligible[0] if eligible and eligible[0][0] > limit else None
    vec = chosen[3] if chosen else vec0
    decision = M.decide(vec, qk, q)
    escalated = False
    if qk == "triage_level" and decision != M.ER:
        p_er = ask(text, "er_now", M.ER_QUESTION)
        if p_er is None:
            return None
        if p_er["true"] >= M.ER_YES_AT:
            reads = [M.llm(chosen[2], q) for _ in range(3)] if chosen else (llm_reads or [M.llm(text, q) for _ in range(3)])
            if all(r == M.ER for r in reads):
                decision, escalated = M.ER, True
    return {"decision": decision, "removed": chosen[1] if chosen else None, "escalated": escalated}


def majority(picks, question):
    """Majority of the LLM's answers; ties go to the earlier option."""
    keys = M.options(question)
    counts = Counter(p for p in picks if p is not None)
    return max(keys, key=lambda k: (counts[k], -keys.index(k))) if counts else None
