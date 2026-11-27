"""Summarize paired physical trials. Input JSONL has one row per controller and pair."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


def summarize(rows):
    pairs = defaultdict(dict)
    counts = defaultdict(Counter)
    for r in rows:
        key = (r["condition"], r["pair"])
        arm = r["arm"]
        if arm not in ("incumbent", "candidate") or arm in pairs[key]:
            raise ValueError(f"Invalid or duplicate arm for {key}")
        status = r["status"]
        if status not in ("success", "timeout", "abort"):
            raise ValueError(f"Invalid status: {status}")
        if status == "success" and not 0 <= r["reacquisition_ms"] <= r["timeout_ms"]:
            raise ValueError("Invalid reacquisition time")
        if status != "success" and r.get("reacquisition_ms") is not None:
            raise ValueError("Only successful trials have a reacquisition time")
        if status == "abort" and not r.get("reason"):
            raise ValueError("Aborted trial requires a reason")
        pairs[key][arm] = r
        totals = counts[(key[0], arm)]
        totals[status] += 1
        totals["switches"] += r["switches"]
        totals["missed_deadlines"] += r["missed_deadlines"]
    deltas = defaultdict(list)
    outcomes = defaultdict(Counter)
    for (condition, pair), arms in pairs.items():
        if len(arms) < 2:
            outcomes[condition]["incomplete"] += 1
            continue
        a, b = arms["incumbent"], arms["candidate"]
        if a["timeout_ms"] != b["timeout_ms"]:
            raise ValueError(f"Unmatched timeout for {pair}")
        outcomes[condition][a["status"] + "/" + b["status"]] += 1
        if a["status"] == b["status"] == "success":
            deltas[condition].append(b["reacquisition_ms"] - a["reacquisition_ms"])
    return {"arms": [{"condition": c, "arm": a, **n,
                      "success_rate_completed": n["success"] / (n["success"] + n["timeout"])
                      if n["success"] + n["timeout"] else None} for (c,a),n in counts.items()],
            "pairs": [{"condition": c, "outcomes": n,
                       "median_candidate_minus_incumbent_ms": median(deltas[c]) if deltas[c] else None}
                      for c,n in outcomes.items()]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trials", type=Path)
    args = parser.parse_args()
    with args.trials.open() as stream:
        print(json.dumps(summarize(json.loads(line) for line in stream), indent=2))
