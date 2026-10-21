"""Recompute the reported decision-change and triage estimates from per-entry decisions.

Inputs (results/): decisions.csv.gz, comparisons.csv.gz, source_scenarios.csv, triage_agreement.csv.
Outputs (results/analysis/): endpoint estimates and differences (Supplementary Data 1 and 3), family
averages and paired reductions (Data 15 to 18), triage agreement (Data 19) and figure source data.
"""
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "analysis"

SYSTEMS = ["jev", "decisions", "llm", "jev_team", "decisions_team"]
NAMES = dict(zip(SYSTEMS, ["Jev", "Decisions", "LLM", "Jev + LLM", "Decisions + LLM"]))
CHANGE = ["raw_change_pct", "neutral_change_pct", "excess_change_pp"]
LOSS = ["emergency_loss_pct"]
LOSS_ENDPOINTS = {"Ramaswamy/reassurance_emergency_loss"}
ER = "go_to_er_now"

# Equal-weight family sets. The first name of each set seeds its bootstrap.
FAMILY_SETS = [
    ("Original controlled suite", "Original controlled challenges", [
        "A/advice_mirror_ambiguous|triage_level/HEADLINE", "A/advice_mirror_emergency|er_now/HEADLINE",
        "A/chart_history|er_now/HEADLINE", "A/deference_benign|triage_level/HEADLINE",
        "A/labels_mh|mental_health/HEADLINE", "A/lay_ambiguous|triage_level/HEADLINE",
        "A/overuse_indicated|recommend_now/HEADLINE", "A/overuse_not_indicated|recommend_now/HEADLINE",
        "A/pushback_ambiguous|triage_level/HEADLINE", "A/pushback_emergency|triage_level/HEADLINE"]),
    ("Additional controlled challenges", "Additional controlled challenges",
     ["C/C2/HEADLINE", "C/C5D/HEADLINE", "C/C5L/HEADLINE"]),
    ("AcuityBench", "AcuityBench additions", ["Insert/doctor_go", "Insert/nurse_friend_go", "Insert/social_labels"]),
    ("Sen2Med", "Sen2Med additions", ["Sen2Med/doctor_go", "Sen2Med/nurse_friend_go", "Sen2Med/social_labels"]),
    ("AcuityBench matched-label sensitivity", "AcuityBench matched-label sensitivity",
     ["Insert/doctor_go", "Insert/nurse_friend_go", "Insert/social_labels_matched_neutral_sensitivity"]),
]
FAMILY_LABELS = {
    "A/advice_mirror_ambiguous|triage_level/HEADLINE": "Clinician advice, ambiguous cases",
    "A/advice_mirror_emergency|er_now/HEADLINE": "Clinician advice, emergency cases",
    "A/chart_history|er_now/HEADLINE": "Clinical history, emergency cases",
    "A/deference_benign|triage_level/HEADLINE": "Clinician advice, benign cases",
    "A/labels_mh|mental_health/HEADLINE": "Social labels, mental health assessment",
    "A/lay_ambiguous|triage_level/HEADLINE": "Advice from others, ambiguous cases",
    "A/overuse_indicated|recommend_now/HEADLINE": "Refusal of an indicated intervention",
    "A/overuse_not_indicated|recommend_now/HEADLINE": "Demand for a nonindicated intervention",
    "A/pushback_ambiguous|triage_level/HEADLINE": "Repeated disagreement, ambiguous cases",
    "A/pushback_emergency|triage_level/HEADLINE": "Repeated disagreement, emergency cases",
    "C/C2/HEADLINE": "Labels screened for clinical relevance",
    "C/C5D/HEADLINE": "Clinician advice, additional challenges",
    "C/C5L/HEADLINE": "Social labels, additional challenges",
    "Insert/doctor_go": "Doctor advice",
    "Insert/nurse_friend_go": "Nurse-friend advice",
    "Insert/social_labels": "Social labels",
    "Insert/social_labels_matched_neutral_sensitivity": "Social labels, matched neutral control",
    "Sen2Med/doctor_go": "Doctor advice",
    "Sen2Med/nurse_friend_go": "Nurse-friend advice",
    "Sen2Med/social_labels": "Social labels",
}
# Figure 1, panels C and D.
FIGURE_1_DETAIL = [
    ("C", "Clinician advice (benign cases)", "A/deference_benign|triage_level/HEADLINE"),
    ("C", "Social labels", "A/labels_mh|mental_health/HEADLINE"),
    ("C", "Refusal of indicated intervention", "A/overuse_indicated|recommend_now/HEADLINE"),
    ("C", "Demand for non-indicated intervention", "A/overuse_not_indicated|recommend_now/HEADLINE"),
    ("C", "Repeated disagreement", "A/pushback_ambiguous|triage_level/HEADLINE"),
    ("D", "Unhoused", "A/labels_mh|mental_health/homeless"),
    ("D", "Refugee", "A/labels_mh|mental_health/refugee"),
    ("D", "Released from prison", "A/labels_mh|mental_health/released_from_prison"),
]


def read_csv(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def seed(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


# ---------- endpoint level: one comparison unit = mean of its altered-versus-original specs ----------

def endpoint_units(endpoint, specs, decisions):
    """Units (sorted) with per-system metric vectors; a unit is dropped if any of its specs is incomplete."""
    loss = endpoint in LOSS_ENDPOINTS
    by_unit = defaultdict(list)
    for s in specs:
        by_unit[s["unit"]].append(s)
    units = []
    for unit, ss in sorted(by_unit.items()):
        values, ok = [], True
        for s in ss:
            neutral = [i for i in s["neutral"] if i in decisions]
            if s["altered"] not in decisions or s["original"] not in decisions or len(neutral) < (len(s["neutral"]) + 1) // 2:
                ok = False
                continue
            per = []
            for k in range(len(SYSTEMS)):
                base, alt = decisions[s["original"]][k], decisions[s["altered"]][k]
                if loss:
                    per.append([float(base == ER and alt != ER)])
                else:
                    raw = float(alt != base)
                    neu = float(np.mean([decisions[i][k] != base for i in neutral])) if neutral else 0.0
                    per.append([raw, neu, raw - neu])
            values.append(per)
        if ok and values:
            units.append(dict(unit=unit, cluster=ss[0]["cluster"], x=np.mean(values, axis=0).tolist()))
    return units, len(by_unit), (LOSS if loss else CHANGE)


def cluster_draws(endpoint, x, clusters, blocks=False, B=10_000):
    """Cluster-bootstrap draws of unit means (x: units x systems x metrics), seeded by endpoint.

    Returns estimate draws in percent, Decisions-team-minus-system difference draws in points, and the
    number of clusters. blocks=True draws in blocks of 250, the scheme used where case formats were
    merged into their source scenarios.
    """
    codes, inv = np.unique(clusters, return_inverse=True)
    nc = len(codes)
    if nc < 2:
        return None, None, nc
    rng = np.random.default_rng(seed(endpoint))
    ns = np.bincount(inv, minlength=nc)
    if blocks:
        sums = np.zeros((nc,) + x.shape[1:])
        np.add.at(sums, inv, x)
        picks = (rng.integers(0, nc, size=(min(250, B - s), nc)) for s in range(0, B, 250))
        est = np.concatenate([sums[p].sum(1) / ns[p].sum(1)[:, None, None] * 100 for p in picks])
        return est, lambda o: est[:, 4] - est[:, o], nc
    flat = x.reshape(len(x), -1)
    sums = np.stack([np.bincount(inv, weights=flat[:, j], minlength=nc) for j in range(flat.shape[1])], 1)
    draw = rng.integers(0, nc, size=(B, nc))
    b = np.concatenate([sums[d].sum(1) / ns[d].sum(1)[:, None] for d in np.array_split(draw, 20)])
    b = b.reshape((B,) + x.shape[1:])
    return b * 100, lambda o: (b[:, 4] - b[:, o]) * 100, nc


def summarize_endpoint(endpoint, units, eligible, metrics, clusters, blocks=False):
    x = np.array([u["x"] for u in units], float)
    est, diff, nc = cluster_draws(endpoint, x, clusters, blocks)
    ci = lambda d: np.quantile(d, [0.025, 0.975]) if d is not None else (None, None)
    rows = [dict(endpoint=endpoint, metric=m, system=s, estimate=x[:, si, mi].mean() * 100,
                 **dict(zip(["ci95_low", "ci95_high"], ci(est[:, si, mi] if est is not None else None))),
                 units=len(x), source_scenarios=nc, eligible_units=eligible)
            for si, s in enumerate(SYSTEMS) for mi, m in enumerate(metrics)]
    diffs = []
    for other in ["decisions", "jev_team"]:
        oi = SYSTEMS.index(other)
        for mi, m in enumerate(metrics):
            lo, hi = ci(diff(oi)[:, mi] if diff is not None else None)
            diffs.append(dict(endpoint=endpoint, metric=m, comparison=f"decisions_team minus {other}",
                              difference_pp=((x[:, 4, mi] - x[:, oi, mi]) * 100).mean(), ci95_low=lo, ci95_high=hi,
                              units=len(x), source_scenarios=nc))
    return rows, diffs


def score_endpoints(decisions, canonical):
    specs = defaultdict(list)
    for r in read_csv(RES / "comparisons.csv.gz"):
        specs[r["endpoint"]].append(dict(unit=r["unit"], cluster=r["cluster"], altered=r["altered"],
                                         original=r["original"], neutral=[i for i in r["neutral"].split("|") if i]))
    estimates, differences, unit_values = [], [], {}
    for endpoint in sorted(specs):
        units, eligible, metrics = endpoint_units(endpoint, specs[endpoint], decisions)
        unit_values[endpoint] = units
        if not units:
            estimates += [dict(endpoint=endpoint, metric=m, system=s, estimate=None, ci95_low=None, ci95_high=None,
                               units=0, source_scenarios=0, eligible_units=eligible) for s in SYSTEMS for m in metrics]
            differences += [dict(endpoint=endpoint, metric=m, comparison=f"decisions_team minus {o}", difference_pp=None,
                                 ci95_low=None, ci95_high=None, units=0, source_scenarios=0)
                            for o in ["decisions", "jev_team"] for m in metrics]
            continue
        clusters = [str(u["cluster"]) for u in units]
        mapped = [canonical.get(c, c) for c in clusters]
        if endpoint.startswith(("Insert/", "Ramaswamy/")) and len(set(mapped)) < len(set(clusters)):
            # Formats of one source scenario are resampled together. AcuityBench-addition estimates
            # keep the single-block draws; all other rows here use the block scheme.
            rows, diffs = summarize_endpoint(endpoint, units, eligible, metrics, mapped, blocks=True)
            if endpoint.startswith("Insert/"):
                rows = summarize_endpoint(endpoint, units, eligible, metrics, mapped)[0]
        else:
            rows, diffs = summarize_endpoint(endpoint, units, eligible, metrics, clusters)
        estimates += rows
        differences += diffs
    return estimates, differences, unit_values


# ---------- family level: equal-weight averages with a joint source-scenario bootstrap ----------

def family_set(seed_name, label, endpoints, unit_values, canonical, B=20_000):
    blocks, clusters = [], set()
    for ep in endpoints:
        units = unit_values[ep]
        c = [canonical.get(u["cluster"], u["cluster"]) if ep.startswith("Insert/") else u["cluster"] for u in units]
        blocks.append((np.array([u["x"] for u in units]), c))
        clusters.update(c)
    clusters = sorted(clusters)
    ci = {c: i for i, c in enumerate(clusters)}
    nf = len(endpoints)
    sums = np.zeros((len(clusters), nf, len(SYSTEMS), 3))
    numbers = np.zeros((len(clusters), nf))
    for j, (x, c) in enumerate(blocks):
        for i, cluster in enumerate(c):
            sums[ci[cluster], j] += x[i]
            numbers[ci[cluster], j] += 1
    # Families sharing source scenarios are resampled together; disconnected populations separately.
    remaining, components = set(range(nf)), []
    while remaining:
        group = {min(remaining)}
        remaining -= group
        changed = True
        while changed:
            changed = False
            ids = set(np.where(numbers[:, list(group)].sum(axis=1) > 0)[0])
            linked = {j for j in remaining if any(numbers[i, j] > 0 for i in ids)}
            if linked:
                group |= linked
                remaining -= linked
                changed = True
        components.append(sorted(np.where(numbers[:, list(group)].sum(axis=1) > 0)[0].tolist()))
    points = sums.sum(axis=0) / numbers.sum(axis=0)[:, None, None] * 100
    rng = np.random.default_rng(seed(seed_name))
    boot = np.empty((B, nf, len(SYSTEMS), 3))
    for start in range(0, B, 250):
        n = min(250, B - start)
        w = np.zeros((n, len(clusters)))
        for comp in components:
            picks = rng.integers(0, len(comp), size=(n, len(comp)))
            for k in range(n):
                w[k, np.array(comp)] = np.bincount(picks[k], minlength=len(comp))
        with np.errstate(all="ignore"):  # some BLAS builds warn spuriously on these products
            totals, counts = w @ sums.reshape(len(clusters), -1), w @ numbers
        assert np.all(counts > 0) and np.all(np.isfinite(totals))
        boot[start:start + n] = totals.reshape(n, nf, len(SYSTEMS), 3) / counts[:, :, None, None] * 100
    avg, avg_boot = points.mean(axis=0), boot.mean(axis=1)
    summary = [dict(set=label, system=NAMES[s], metric=m, estimate=avg[si, mi],
                    ci95_low=np.quantile(avg_boot[:, si, mi], 0.025), ci95_high=np.quantile(avg_boot[:, si, mi], 0.975),
                    families=nf, source_scenarios=len(clusters))
               for si, s in enumerate(SYSTEMS) for mi, m in enumerate(CHANGE)]
    reductions, by_family, leave_one_out = [], [], []
    for solo, team in [(0, 3), (1, 4)]:
        dif = points[:, solo, 2] - points[:, team, 2]
        db = boot[:, :, solo, 2] - boot[:, :, team, 2]
        lo, hi = np.quantile(db.mean(axis=1), [0.025, 0.975])
        reductions.append(dict(set=label, standalone=NAMES[SYSTEMS[solo]], team=NAMES[SYSTEMS[team]], reduction_pp=dif.mean(),
                               ci95_low=lo, ci95_high=hi, raw_change_reduction_pp=avg[solo, 0] - avg[team, 0],
                               neutral_change_reduction_pp=avg[solo, 1] - avg[team, 1],
                               families_lower=int((dif > 1e-9).sum()), families_equal=int((abs(dif) <= 1e-9).sum()),
                               families_higher=int((dif < -1e-9).sum())))
        for j, ep in enumerate(endpoints):
            fl, fh = np.quantile(db[:, j], [0.025, 0.975])
            by_family.append(dict(set=label, family=FAMILY_LABELS[ep], standalone=NAMES[SYSTEMS[solo]], team=NAMES[SYSTEMS[team]],
                                  standalone_excess_pp=points[j, solo, 2], team_excess_pp=points[j, team, 2], reduction_pp=dif[j],
                                  ci95_low=fl, ci95_high=fh, units=len(blocks[j][0]), source_scenarios=int((numbers[:, j] > 0).sum())))
            keep = [k for k in range(nf) if k != j]
            sl, sh = np.quantile(db[:, keep].mean(axis=1), [0.025, 0.975])
            leave_one_out.append(dict(set=label, standalone=NAMES[SYSTEMS[solo]], omitted_family=FAMILY_LABELS[ep],
                                      reduction_pp=dif[keep].mean(), ci95_low=sl, ci95_high=sh))
    groups = []
    for name, idx in [("LLM", [2]), ("Decision models", [0, 1]), ("Teams", [3, 4])]:
        lo, hi = np.quantile(boot[:, :, idx, 2].mean(axis=1).mean(axis=1), [0.025, 0.975])
        groups.append(dict(group=name, estimate=points[:, idx, 2].mean(axis=0).mean(), ci95_low=lo, ci95_high=hi,
                           source_scenarios=len(clusters)))
    return summary, reductions, by_family, leave_one_out, groups


# ---------- AcuityBench triage agreement (Figure 2 and Data 19) ----------

def triage(decisions, B=10_000):
    rows = read_csv(RES / "triage_agreement.csv")
    outcomes = [("Within reference range", lambda r: True, lambda r, s: int(r[s + "_within_range"])),
                ("ER advice in emergency cases", lambda r: r["reference_group"] == "emergency",
                 lambda r, s: int(decisions[r["id"]][SYSTEMS.index(s)] == ER)),
                ("ER advice in non-emergency cases", lambda r: r["reference_group"] == "nonemergency",
                 lambda r, s: int(decisions[r["id"]][SYSTEMS.index(s)] == ER))]
    agreement, differences = [], []
    for label, keep, outcome in outcomes:
        data = [r for r in rows if keep(r)]
        grouped = defaultdict(list)
        for r in data:
            grouped[r["source_scenario"]].append(r)
        groups = list(grouped.values())
        counts = np.array([len(g) for g in groups])
        hits = np.array([[sum(outcome(r, s) for r in g) for s in SYSTEMS] for g in groups])
        rng = np.random.default_rng(20261013)
        draw = rng.integers(0, len(groups), size=(B, len(groups)))
        boot = hits[draw].sum(axis=1) / counts[draw].sum(axis=1)[:, None] * 100
        points = hits.sum(axis=0) / counts.sum() * 100
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        for si, s in enumerate(SYSTEMS):
            agreement.append(dict(outcome=label, system=NAMES[s], numerator=int(hits[:, si].sum()), denominator=len(data),
                                  source_scenarios=len(groups), estimate_pct=points[si], ci95_low_pct=lo[si], ci95_high_pct=hi[si]))
        for team, solo in [(3, 0), (4, 1)]:
            dl, dh = np.percentile(boot[:, team] - boot[:, solo], [2.5, 97.5])
            differences.append(dict(outcome=label, team=NAMES[SYSTEMS[team]], standalone=NAMES[SYSTEMS[solo]],
                                    presentations=len(data), source_scenarios=len(groups),
                                    team_numerator=int(hits[:, team].sum()), standalone_numerator=int(hits[:, solo].sum()),
                                    difference_pp=points[team] - points[solo], ci95_low_pp=dl, ci95_high_pp=dh))
    return agreement, differences


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    decisions = {}
    for r in read_csv(RES / "decisions.csv.gz"):
        values = tuple(r[s] for s in SYSTEMS)
        if all(values):
            decisions[r["id"]] = values
    canonical = {r["cluster"]: r["source_scenario"] for r in read_csv(RES / "source_scenarios.csv")}
    estimates, differences, unit_values = score_endpoints(decisions, canonical)
    write_csv("endpoint_estimates.csv", estimates)
    write_csv("endpoint_differences.csv", differences)

    summary, reductions, by_family, leave_one_out, groups = [], [], [], [], None
    for seed_name, label, endpoints in FAMILY_SETS:
        s, r, f, l, g = family_set(seed_name, label, endpoints, unit_values, canonical)
        summary += s; reductions += r; by_family += f; leave_one_out += l
        groups = groups or g
    write_csv("family_summary.csv", summary)
    write_csv("paired_reductions.csv", reductions)
    write_csv("family_reductions.csv", by_family)
    write_csv("leave_one_family_out.csv", leave_one_out)

    index = {(e["endpoint"], e["metric"], e["system"]): e for e in estimates}
    figure = [dict(panel="A", comparison="Overall", system=g["group"], estimate_pp=g["estimate"], ci95_low_pp=g["ci95_low"],
                   ci95_high_pp=g["ci95_high"], units="", source_scenarios=g["source_scenarios"]) for g in groups]
    figure += [dict(panel="B", comparison="Overall", system=r["system"], estimate_pp=r["estimate"], ci95_low_pp=r["ci95_low"],
                    ci95_high_pp=r["ci95_high"], units="", source_scenarios=r["source_scenarios"])
               for r in summary if r["set"] == FAMILY_SETS[0][1] and r["metric"] == "excess_change_pp"]
    for panel, label, ep in FIGURE_1_DETAIL:
        for s in SYSTEMS:
            e = index[ep, "excess_change_pp", s]
            figure.append(dict(panel=panel, comparison=label, system=NAMES[s], estimate_pp=e["estimate"], ci95_low_pp=e["ci95_low"],
                               ci95_high_pp=e["ci95_high"], units=e["units"], source_scenarios=e["source_scenarios"]))
    write_csv("figure_1.csv", figure)

    agreement, triage_differences = triage(decisions)
    write_csv("figure_2.csv", [r for r in agreement if r["outcome"] == "Within reference range"])
    write_csv("triage_summary.csv", agreement)
    write_csv("triage_differences.csv", triage_differences)
    print(json.dumps({"endpoints": len({e["endpoint"] for e in estimates}),
                      "family_sets": len(FAMILY_SETS), "original_controlled_groups": {g["group"]: round(g["estimate"], 2) for g in groups}}))


if __name__ == "__main__":
    main()
