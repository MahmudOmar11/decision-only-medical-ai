"""Calls to the three systems, exactly as configured in the study, and the decision rule applied to their outputs.

Keys are read from the environment: OPENROUTER_API_KEY (Jev and the LLM) and OPENAI_API_KEY (OpenAI Decisions).
The LLM is called through OpenRouter; set LLM_PROVIDER=openai to call the same model directly, as the Decisions
team did in the study.
"""
import json
import os
import random
import re
import time
import urllib.error
import urllib.request

JEV_URL, JEV_MODEL = "https://openrouter.ai/api/alpha/decisions", "typesafe/jev-1.13"
DECISIONS_URL, DECISIONS_MODEL = "https://api.openai.com/v1/decisions", "gpt-6-luna"
LLM_URL, LLM_MODEL = "https://openrouter.ai/api/v1/chat/completions", "openai/gpt-6-luna"
RETRY = {408, 429, 500, 502, 503, 504, 520, 521, 522, 524, 529}
SYSTEM = "You are assisting with a clinical decision. Use only the information provided. Follow the answer format exactly."
FINDING_QUESTION = ("Does this sentence report a finding about the patient: a symptom, a sign, a vital sign, an "
                    "examination finding, a test result, or a medical condition, medication, allergy or procedure "
                    "in the patient's history?")
ER_QUESTION = {"type": "noul",
               "instructions": "Does this person need to go to the ER now? Base your answer only on the information in this message.",
               "criteria": {"true": "Needs to go to the emergency room now",
                            "false": "Does not need to go to the emergency room now"}}
ER = "go_to_er_now"
ER_YES_AT = 0.30


def _post(url, key_env, body, timeout=120, attempts=6):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST", headers={
        "Authorization": "Bearer " + os.environ[key_env], "Content-Type": "application/json"})
    delay = 2.0
    for i in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code not in RETRY or i == attempts - 1:
                raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}")
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if i == attempts - 1:
                raise
        time.sleep(delay + random.random())
        delay = min(delay * 2, 60)


def options(question):
    return ["true", "false"] if question["type"] == "noul" else list(question["criteria"])


def decide(probs, question_key, question):
    """Choice: the most probable option, ties to the earlier option. Yes/no: 'true' at or above the cut-off."""
    if not probs:
        return None
    if question["type"] == "noul":
        cut = ER_YES_AT if question_key == "er_now" else 0.5
        return "true" if float(probs.get("true", 0.0)) >= cut else "false"
    keys = options(question)
    return max(keys, key=lambda k: (float(probs.get(k, 0.0)), -keys.index(k)))


def jev_body(text, question_key, question):
    return {"model": JEV_MODEL, "state": text, "questions": {question_key: question}}


def jev(text, question_key, question):
    """Jev 1.13 through OpenRouter's decisions endpoint. Returns option probabilities."""
    raw = _post(JEV_URL, "OPENROUTER_API_KEY", jev_body(text, question_key, question))
    result = raw.get("result", raw)
    ans = next(box[question_key] for box in (result, result.get("answers"), result.get("output"), result.get("result"))
               if isinstance(box, dict) and isinstance(box.get(question_key), dict))
    if question["type"] == "noul":
        p = float(ans.get("noul", ans.get("probability")))
        return {"true": p, "false": 1 - p}
    return {k: float(v) for k, v in ans["probabilities"].items()}


def decisions_body(text, question_key, question):
    """Yes/no questions are sent as predicates with both definitions appended to the instructions."""
    q = {"name": question_key, "instructions": question["instructions"]}
    if question["type"] == "noul":
        c = question["criteria"]
        q.update(type="predicate", instructions=question["instructions"] + "\n\nTrue means: " + c["true"]
                 + "\nFalse means: " + c["false"])
    else:
        q.update(type="choice", choices=[{"value": k, "description": d} for k, d in question["criteria"].items()])
    return {"model": DECISIONS_MODEL, "input": text, "questions": [q]}


def decisions(text, question_key, question):
    """OpenAI Decisions. Returns option probabilities, or None for a refusal."""
    raw = _post(DECISIONS_URL, "OPENAI_API_KEY", decisions_body(text, question_key, question))
    ans = raw["answers"][0]
    if ans.get("type") == "refusal":
        return None
    if question["type"] == "noul":
        p = float(ans["probability"])
        return {"true": p, "false": 1 - p}
    return {str(e["value"]): float(e["probability"]) for e in ans["probabilities"]}


def _chat(messages):
    if os.environ.get("LLM_PROVIDER") == "openai":
        raw = _post("https://api.openai.com/v1/chat/completions", "OPENAI_API_KEY",
                    {"model": DECISIONS_MODEL, "messages": messages, "reasoning_effort": "low",
                     "max_completion_tokens": 2000})
    else:
        raw = _post(LLM_URL, "OPENROUTER_API_KEY", {"model": LLM_MODEL, "messages": messages, "max_tokens": 2000,
                                                     "reasoning": {"effort": "low"}})
    return raw["choices"][0]["message"].get("content") or ""


def llm_messages(text, question):
    if question["type"] == "choice":
        body = "\n".join(f"{'ABCD'[i]}. {o}" for i, o in enumerate(question["criteria"].values()))
        user = f"{text}\n\n{question['instructions']}\n\n{body}\n\nAnswer with the letter of one option only."
    else:
        user = f"{text}\n\n{question['instructions']}\n\nAnswer yes or no only."
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def _parse(reply, question):
    space = ({"ABCD"[i]: k for i, k in enumerate(question["criteria"])} if question["type"] == "choice"
             else {"YES": "true", "NO": "false"})
    t = (reply or "").strip()
    exact = re.fullmatch(r"[\s*(\[]*([A-Za-z]+)[\s*)\].:]*", t)
    if exact and exact.group(1).upper() in space:
        return space[exact.group(1).upper()]
    words = "|".join(space) if question["type"] == "noul" else "[" + "".join(space) + "]"
    m = re.search(rf"(?<![A-Za-z])({words})(?![A-Za-z])", t, flags=re.I)
    return space[m.group(1).upper()] if m else None


def llm(text, question):
    """One LLM answer mapped to an option key, with one retry when the reply cannot be read."""
    messages = llm_messages(text, question)
    for _ in range(2):
        pick = _parse(_chat(messages), question)
        if pick is not None:
            return pick
    return None


def finding_messages(piece):
    user = f"Sentence: \"{piece}\"\n\n{FINDING_QUESTION}\n\nAnswer yes or no only."
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def is_finding(piece):
    """The LLM's one question in the paired workflow: does this passage report a patient finding? ('true'/'false')"""
    for _ in range(2):
        pick = _parse(_chat(finding_messages(piece)), {"type": "noul"})
        if pick is not None:
            return pick
    return None


DECISION_MODELS = {"jev": jev, "decisions": decisions}
REQUEST_BODIES = {"jev": jev_body, "decisions": decisions_body}
