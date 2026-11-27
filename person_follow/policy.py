"""Decision contract and causal replay for the single-person shadow trial."""
from collections import deque
import json
import math

import numpy as np

VERSION = "person-follow-v1"
ACTIONS = ("follow", "hold_distance", "stop")
WINDOW = 6
FEATURES = ("center_x", "height", "quality", "visible", "interval_s", "age_s")
MAX_AGE = 0.5


class DecisionPolicy:
    """One causal session, shared by replay and the live controller."""

    def __init__(self, model=None):
        self.model = model
        self.history = deque(maxlen=WINDOW)
        self.previous_time = None
        self.following = False

    def step(self, row):
        if row["version"] != VERSION:
            raise ValueError("Unsupported observation version")
        t, age = row["t"], row["age_s"]
        dt = 0 if self.previous_time is None else t - self.previous_time
        if not all(map(math.isfinite, (t, age, dt))) or age < 0 or dt < 0 or (self.previous_time is not None and dt == 0):
            raise ValueError("Expected increasing timestamps and nonnegative age")
        self.previous_time = t
        box = row["target"]
        if box is not None:
            x, y, w, h = box
            if not (0 <= x < x+w <= 1.000001 and 0 <= y < y+h <= 1.000001):
                raise ValueError("Expected normalized xywh")
        q = row["quality"]
        if not 0 <= q <= 1:
            raise ValueError("Expected normalized quality")
        valid = box is not None and q >= 0.5 and age <= MAX_AGE
        if dt > MAX_AGE or not valid:
            self.history.clear()
            self.following = False
        if valid:
            self.history.append([2*(x+w/2)-1, h, q, 1, min(dt, 1), min(age, 1)])
        scores = None
        mask = [False, False, True]
        if row.get("override", False):
            reason, selected = "operator_override", None
            self.history.clear()
            self.following = False
        elif not valid:
            reason, selected = "unreliable_target", "stop"
        elif len(self.history) < WINDOW:
            reason, selected = "warming_up", "stop"
        else:
            mask = [h < 0.65, True, True]
            if self.model is None:
                self.following = h < (0.51 if self.following else 0.42)
                selected = "stop" if h >= 0.65 else ("follow" if self.following else "hold_distance")
                reason = "height_hysteresis"
            else:
                scores = self.model(np.array(self.history)[None])[0].tolist()
                if len(scores) != len(ACTIONS) or not all(map(math.isfinite, scores)):
                    selected, reason = None, "invalid_scores"
                else:
                    selected = ACTIONS[max((i for i in range(3) if mask[i]), key=scores.__getitem__)]
                    reason = "masked_choice"
        return dict(t=t, status="choice" if selected else "defer", selected=selected,
                   reason=reason, mask=mask, scores=scores), list(self.history)


def replay(rows, model=None):
    policy = DecisionPolicy(model)
    for row in rows:
        yield policy.step(row)


def observations(path):
    with open(path) as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)
