"""Export the original data, audit switches, and replay the independent fixtures."""
import argparse
import json
from pathlib import Path

import numpy as np

from episodes import HISTORY, FEATURES, acceptable, dataset, make_session

NAMES = ("track", "search", "wait")
ROOT = Path(__file__).resolve().parent


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            print(json.dumps(row, separators=(",", ":")), file=stream)


def annotation(window):
    accepted = [name for name, flag in zip(NAMES, acceptable(window)) if flag]
    if not accepted:
        reason = "missing"
    elif window[-1] == 2:
        reason = "confirmed" if accepted == ["track"] else "isolated"
    elif len(accepted) == 2:
        reason = "boundary"
        accepted = ["wait", "search"]
    else:
        reason = "brief" if accepted == ["wait"] else "lost"
    return accepted, reason


def read_table():
    text = (ROOT / "edge/scores.inc").read_text()
    return np.array([[float(s.strip().removesuffix("f")) for s in line.strip("{},").split(",")]
                     for line in text.splitlines()])


def prediction(table, window):
    if window[-1] == 0:
        return "defer"
    index = sum(symbol * 3 ** power for power, symbol in enumerate(reversed(window)))
    return NAMES[int(table[index].argmax())]


def export(output, table):
    switches = []
    for split, start, count in [("train", 0, 48), ("validation", 1000, 12), ("test", 2000, 24)]:
        sessions = []
        for seed in range(start, start + count):
            frames = make_session(seed).tolist()
            ticks, previous = [], None
            for tick in range(HISTORY - 1, len(frames)):
                window = frames[tick - HISTORY + 1:tick + 1]
                accepted, reason = annotation(window)
                ticks.append(dict(tick=tick, accepted=accepted, reason=reason))
                choice = prediction(table, window)
                if split == "test" and choice != previous and previous in accepted:
                    switches.append(dict(session=seed, tick=tick, window=window, previous=previous,
                                         choice=choice, accepted=accepted, reason=reason))
                previous = choice
            sessions.append(dict(session=seed, split=split, frames=frames, ticks=ticks))
        write_rows(output / "data" / (split + ".jsonl"), sessions)
    write_rows(output / "reports/switches.jsonl", switches)


def review(output, table, include_mlp):
    if include_mlp:
        import torch
        from train import fit
        torch.set_num_threads(1)
        mlp, selection = fit("mlp", dataset(0, 48), dataset(1000, 12))
        print("MLP selection:", selection)
    results = []
    for line in (ROOT / "data/review-cases.jsonl").read_text().splitlines():
        case = json.loads(line)
        frames, labels = case["frames"], case["labels"]
        assert len(labels) == len(frames) - HISTORY + 1
        previous = None
        for tick, label in enumerate(labels, HISTORY - 1):
            window = frames[tick - HISTORY + 1:tick + 1]
            options, _ = annotation(window)
            rule = previous if previous in options else next(iter(options), "defer")
            previous = rule
            age = next((i for i, value in enumerate(reversed(window)) if value == 2), HISTORY)
            timeout = "track" if age == 0 else "search" if age >= 3 else "wait"
            row = dict(id=case["id"], tick=tick, accepted=label.split("|"), rule=rule,
                       timeout=timeout if options else "defer", transformer=prediction(table, window))
            if include_mlp:
                with torch.no_grad():
                    scores = mlp(torch.tensor(FEATURES[np.asarray(window)[None, :]]))
                row["mlp"] = NAMES[int(scores.argmax())] if options else "defer"
            results.append(row)
    write_rows(output / "reports/review-replay.jsonl", results)
    for key in ("rule", "timeout", "transformer") + (("mlp",) if include_mlp else ()):
        print(key, sum(row[key] in row["accepted"] for row in results), "/", len(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--with-mlp", action="store_true")
    args = parser.parse_args()
    scores = read_table()
    export(args.output, scores)
    review(args.output, scores, args.with_mlp)
