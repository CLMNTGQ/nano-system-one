"""Follow one person using local MicroDuck camera, depth and motion services."""
import argparse
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import selectors
import socket
import subprocess
import sys
import time
import uuid

from .model import Network
from .policy import DecisionPolicy, MAX_AGE

ZERO = (0., 0., 0.)
STATE_TTL = .25
DEPTH_TTL = .35
TICK = .05


def rpc(method, params=None, request_id=None):
    msg = dict(jsonrpc="2.0", method=method)
    if params is not None:
        msg["params"] = params
    if request_id is not None:
        msg["id"] = request_id
    return (json.dumps(msg, allow_nan=False) + "\n").encode()


def move(twist):
    return rpc("robot.move", dict(zip(("vx", "vy", "vyaw"), twist)))


def fresh(stamp, now, limit):
    return math.isfinite(stamp) and -1e-9 <= now - stamp <= limit


def clear_ahead(frame):
    """Conservative central forward corridor for the fixed, forward-facing head."""
    if (frame["rows"], frame["cols"]) != (8, 8):
        return False
    ranges, status = frame["distance_mm"], frame["status"]
    if len(ranges) != 64 or len(status) != 64:
        return False
    return all(status[i] in (5, 9) and ranges[i] >= 400
               for row in range(2, 6) for i in range(row*8+2, row*8+6))


class Controller:
    def __init__(self, model, name):
        self.policy = DecisionPolicy(model)
        self.name = name
        self.state = None
        self.depth_t = self.expiry = -math.inf
        self.clear = False
        self.target = None
        self.decision = None
        self.sent_motion = False
        self.revoked = False

    def observe(self, row, wall, now):
        age = wall - row["t"]
        if age < 0:
            raise ValueError("Camera timestamp is in the future")
        row = dict(row, age_s=max(age, row["age_s"]))
        self.expiry = now + MAX_AGE - row["age_s"]
        self.decision, _ = self.policy.step(row)
        self.target = row["target"]

    def robot(self, state):
        self.state = state
        motion = state["move"]
        live = "deadman" not in motion.get("limited_by", [])
        if live and motion.get("source") != self.name:
            self.revoked = True
        if (state.get("autonomous") or state.get("soccer") or state.get("laser")
                or state["policy"] not in ("walk", "stand")
                or state["safety"]["fallen"] or state["safety"]["limp"]
                or state["safety"].get("picked_up") or state["safety"].get("held")):
            self.revoked = True

    def depth(self, frame):
        self.depth_t = frame["t_ns"] / 1e9
        self.clear = clear_ahead(frame)

    def command(self, now):
        if self.revoked or self.state is None:
            return None, "unavailable_or_taken_over"
        if not fresh(self.state["t_ns"]/1e9, now, STATE_TTL):
            return None, "stale_robot_state"
        if self.decision is None or now > self.expiry:
            return ZERO, "expired_observation"
        if not fresh(self.depth_t, now, DEPTH_TTL) or not self.clear:
            return ZERO, "depth_blocked_or_stale"
        head = self.state["head"]
        if len(head) != 4 or not abs(head[2]) <= .18:
            return ZERO, "head_off_axis"
        action = self.decision["selected"]
        if action not in ("follow", "hold_distance") or self.target is None:
            return ZERO, self.decision["reason"]
        x, _, w, h = self.target
        error = 2*x + w - 1
        yaw = 0. if abs(error) < .08 else max(-.35, min(.35, -.7*error))
        vx = min(.10, max(0., .5*(.50-h))) if action == "follow" and abs(error) < .45 else 0.
        return (vx, 0., yaw), action


def connect(stack, path):
    sock = stack.enter_context(socket.socket(socket.AF_UNIX))
    sock.settimeout(1)
    sock.connect(path)
    return sock


def request(sock, method, params, ident):
    sock.sendall(rpc(method, params, ident))
    # Read exactly one line. Streaming notifications may already follow the ack.
    data = bytearray()
    deadline = time.monotonic()+2
    while len(data) < 65536 and time.monotonic() < deadline:
        ch = sock.recv(1)
        if not ch:
            break
        if ch == b"\n":
            msg = json.loads(data)
            if msg.get("id") != ident or "result" not in msg:
                raise RuntimeError(f"{method} refused: {msg}")
            return msg["result"]
        data.extend(ch)
    raise ConnectionError(f"Incomplete {method} reply")


def control_loop(robot, depth, camera, controller, log):
    streams = {"robot": robot, "depth": depth, "camera": camera}
    buffers = {key: bytearray() for key in streams}
    with selectors.DefaultSelector() as sel:
        for key, file in streams.items():
            sel.register(file, selectors.EVENT_READ, key)
        next_tick = time.monotonic()
        last_log = None
        last_drive_state = None
        try:
            while True:
                for entry, _ in sel.select(max(0., next_tick-time.monotonic())):
                    key = entry.data
                    data = os.read(entry.fd, 65536)
                    if not data:
                        raise ConnectionError(f"{key} stream closed")
                    buf = buffers[key]
                    buf.extend(data)
                    if len(buf) > 2**20:
                        raise ValueError(f"Oversized {key} message")
                    while b"\n" in buf:
                        line, _, tail = buf.partition(b"\n")
                        buf[:] = tail
                        msg = json.loads(line)
                        if key == "camera":
                            controller.observe(msg, time.time(), time.monotonic())
                        elif key == "robot" and msg["method"] == "robot.state":
                            controller.robot(msg["params"])
                        elif key == "depth" and msg["method"] == "tof.frame":
                            controller.depth(msg["params"])
                if controller.revoked:
                    raise RuntimeError("Control released: other driver, mode or posture")
                now = time.monotonic()
                if now < next_tick:
                    continue
                next_tick = now + TICK
                twist, reason = controller.command(now)
                # Never claim the velocity slot just to send idle heartbeats.
                state_id = (controller.state or {}).get("t_ns")
                drive = twist is not None and twist != ZERO and state_id != last_drive_state
                release = twist == ZERO and controller.sent_motion
                if drive or release:
                    robot.sendall(move(twist))
                    controller.sent_motion = drive
                    last_drive_state = state_id
                marker = (controller.decision or {}).get("t"), reason
                if marker != last_log:
                    state = controller.state or {}
                    print(json.dumps(dict(t=now, decision=controller.decision, control=reason,
                                          twist=twist, reported_move=state.get("move"))), file=log, flush=True)
                    last_log = marker
        finally:
            # Leave a new owner's slot alone. A broken link falls back to robotd's deadman.
            if controller.sent_motion and controller.command(time.monotonic())[0] is not None:
                try:
                    robot.sendall(move(ZERO))
                except OSError:
                    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-socket", default="/run/robotd.sock")
    parser.add_argument("--media-socket", default="/run/mediad/media.sock")
    parser.add_argument("--tof-socket", default="/run/tofd/tof.sock")
    parser.add_argument("--model", type=Path, default=Path(__file__).parent/"checkpoints"/"transformer.json")
    args = parser.parse_args()
    name = "nano-follow-"+uuid.uuid4().hex[:10]
    controller = Controller(Network.load(args.model), name)
    with ExitStack() as stack:
        robot = connect(stack, args.robot_socket)
        hello = request(robot, "hello", dict(api_version=60, client=name), 1)
        if hello["api_version"] < 60:
            raise RuntimeError("robotd API v60+ required for movement source telemetry")
        if not request(robot, "robot.subscribe", {"hz": 25}, 2)["accepted"]:
            raise RuntimeError("robotd refused state subscription")
        depth = connect(stack, args.tof_socket)
        if not request(depth, "tof.stream", None, 1)["accepted"]:
            raise RuntimeError("tofd refused depth subscription")
        camera = stack.enter_context(subprocess.Popen(
            [sys.executable, "-m", "person_follow.capture", "--socket", args.media_socket],
            stdout=subprocess.PIPE))
        stack.callback(camera.terminate)
        control_loop(robot, depth, camera.stdout, controller, sys.stdout)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
