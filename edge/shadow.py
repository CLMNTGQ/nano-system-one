"""Read-only scorer sidecar. Read resampled upstream JSON events from stdin."""
import argparse
from choice import ChoiceScorer
import json
import sys
import time
from collections import deque
from pathlib import Path



def run(library, max_age_ms):
    scorer = ChoiceScorer(library)
    frames = deque(maxlen=6)
    last_session = last_tick = last_generation = last_capture = None
    for line in sys.stdin:
        event = json.loads(line)
        session, tick, generation = event["session"], event["tick"], event["ownership_generation"]
        capture = event["capture_mono_ns"]
        symbol, mask = event["symbol"], event["admissible_mask"]
        if symbol not in (0, 1, 2) or type(mask) is not int or not 0 <= mask <= 7:
            raise ValueError("Invalid frame or admissibility mask")
        if last_tick is None or session != last_session or generation != last_generation or tick != last_tick + 1 or capture <= last_capture:
            frames.clear()
        frames.append(symbol)
        last_session, last_tick, last_generation, last_capture = session, tick, generation, capture
        start = time.monotonic_ns()
        age = start - capture
        blocked = None
        if event["autonomous"] is not True:
            blocked = "ownership_lost"
        elif not 0 <= age <= max_age_ms * 1_000_000:
            blocked = "stale_snapshot"
        decision = scorer.decide(frames, mask, blocked)
        finish = time.monotonic_ns()
        if decision["status"] == "choice" and finish - capture > max_age_ms * 1_000_000:
            decision = scorer.decide(frames, mask, "stale_snapshot")
        print(json.dumps(dict(session=session, tick=tick, capture_mono_ns=capture,
                              symbol=symbol, history=list(frames), admissible_mask=mask,
                              ownership_generation=generation, autonomous=event["autonomous"],
                              trace_version=2, decision=decision, proposal=decision.get("selected", "defer"),
                              executed=event["executed"], age_ms=(finish-capture)/1e6,
                              processing_us=(finish-start)/1e3)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    parser.add_argument("--max-age-ms", type=float, required=True)
    args = parser.parse_args()
    if not 0 < args.max_age_ms < float("inf"):
        parser.error("max-age-ms must be finite and positive")
    run(args.library, args.max_age_ms)
