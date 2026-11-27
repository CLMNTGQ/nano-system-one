"""Subscribe to robot state and detector results; record and replay read-only OODA traces."""
import argparse
import asyncio
import ctypes as ct
import json
import math
import sys
import time
import uuid
from collections import deque
from pathlib import Path

PERIOD = 200_000_000
NAMES = ("track", "search", "wait")
PHASES = {"approaching": "track", "lining up": "track", "searching": "search",
          "getting ready": "wait", "waiting for the policy": "wait"}


class Observer:
    def __init__(self, library):
        self.library = ct.CDLL(str(library.resolve()))
        self.select = self.library.nso_select
        self.select.argtypes = [ct.POINTER(ct.c_uint8), ct.c_uint, ct.c_int, ct.POINTER(ct.c_float)]
        self.select.restype = ct.c_int
        self.frames = deque(maxlen=6)
        self.previous = None
        self.owner = None
        self.last_valid = self.last_positive = None

    def step(self, row):
        identity = (row["session"], row["eligible"], row["source"])
        if self.owner != identity or self.previous != row["tick"] - 1:
            self.frames.clear()
            self.last_valid = self.last_positive = None
        if row["symbol"]:
            self.last_valid = row["detector_rx_ns"]
            if row["symbol"] == 2:
                self.last_positive = self.last_valid
        self.previous, self.owner = row["tick"], identity
        self.frames.append(row["symbol"])
        if len(self.frames) < 6 or not row["eligible"]:
            return "defer"
        choice = self.select((ct.c_uint8 * 6)(*self.frames), row["mask"], 1, None)
        return "defer" if choice < 0 else NAMES[choice]


    def timing(self, now):
        return {"valid_age_ms": None if self.last_valid is None else (now-self.last_valid)/1e6,
                "positive_age_ms": None if self.last_positive is None else (now-self.last_positive)/1e6}


async def subscribe(path, method, params, event, latest, key):
    reader, writer = await asyncio.open_unix_connection(path)
    try:
        request = dict(jsonrpc="2.0", id=1, method=method, params=params)
        writer.write((json.dumps(request) + "\n").encode())
        await writer.drain()
        while line := await reader.readline():
            msg = json.loads(line)
            if msg.get("error"):
                raise RuntimeError(f"{method}: {msg['error']}")
            if msg.get("method") == event:
                latest[key] = (time.monotonic_ns(), msg["params"])
        raise ConnectionError(f"{method} subscription closed")
    finally:
        writer.close()
        await writer.wait_closed()


async def capture(args):
    latest = {}
    observer = Observer(args.library)
    session = str(uuid.uuid4())
    start = time.monotonic_ns()
    next_tick = 1
    async with asyncio.TaskGroup() as group:
        group.create_task(subscribe(args.robot_socket, "robot.subscribe", {"hz": 10}, "robot.state", latest, "state"))
        group.create_task(subscribe(args.media_socket, "media.ball", json.loads(args.detector.read_text()),
                                    "media.ballSeen", latest, "ball"))
        while True:
            await asyncio.sleep(max(0, (start + next_tick * PERIOD - time.monotonic_ns()) / 1e9))
            now = time.monotonic_ns()
            tick = max(next_tick, (now - start) // PERIOD)
            next_tick = tick + 1
            state_rx, state = latest.get("state", (0, {}))
            ball_rx, ball = latest.pop("ball", (0, {}))
            valid = now - PERIOD < ball_rx <= now and type(ball.get("seen")) is bool
            symbol = (2 if ball["seen"] else 1) if valid else 0
            soccer, move = state.get("soccer") or {}, state.get("move") or {}
            source = move.get("source")
            phase = soccer.get("phase")
            eligible = bool(state_rx and now - state_rx <= PERIOD
                            and state.get("autonomous") is True and soccer.get("autonomous") is True
                            and state.get("policy") in ("walk", "stand")
                            and (state.get("safety") or {}).get("fallen") is False
                            and source != "padd" and phase in PHASES)
            row = dict(session=session, tick=tick, decision_ns=now, symbol=symbol,
                       detector_rx_ns=ball_rx or None, state_rx_ns=state_rx or None,
                       state_sensor_ns=state.get("t_ns"), phase=phase,
                       incumbent=PHASES.get(phase), head=soccer.get("head",False),
                       walk=soccer.get("walk",False), source=source, eligible=eligible,
                       mask=7 if eligible else 0, missed=(state.get("loop") or {}).get("missed"))
            row["proposal"] = observer.step(row)
            row.update(observer.timing(now))
            row["complete_ns"] = time.monotonic_ns()
            print(json.dumps(row), flush=True)


def replay(args):
    observer = Observer(args.library)
    ages, processing = [], []
    previous = {}
    misses = comparable = mismatches = count = 0
    with args.trace.open() as stream:
        for line in stream:
            row = json.loads(line)
            count += 1
            proposal = observer.step(row)
            timing = observer.timing(row["decision_ns"])
            mismatches += proposal != row["proposal"] or any(
                key in row and row[key] != value for key, value in timing.items())
            if row["symbol"] and row["detector_rx_ns"] is not None:
                ages.append((row["complete_ns"] - row["detector_rx_ns"]) / 1e6)
            processing.append((row["complete_ns"] - row["decision_ns"]) / 1e6)
            key, counter = row["session"], row["missed"]
            if counter is not None:
                if key in previous and counter >= previous[key]:
                    misses += counter - previous[key]
                    comparable += 1
                previous[key] = counter
    def percentile(values, p):
        return sorted(values)[math.ceil(p * len(values)) - 1] if values else None
    print(json.dumps(dict(rows=count, mismatches=mismatches,
                          detector_receive_to_proposal_p95_ms=percentile(ages,.95),
                          detector_receive_to_proposal_p99_ms=percentile(ages,.99),
                          processing_p99_ms=percentile(processing,.99),
                          missed_deadlines=misses if comparable else None)))
    if not count or mismatches:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    modes = parser.add_subparsers(dest="mode", required=True)
    live = modes.add_parser("capture")
    live.add_argument("--robot-socket", required=True)
    live.add_argument("--media-socket", required=True)
    live.add_argument("--detector", required=True, type=Path)
    offline = modes.add_parser("replay")
    offline.add_argument("trace", type=Path)
    args = parser.parse_args()
    if args.mode == "capture":
        asyncio.run(capture(args))
    else:
        replay(args)
