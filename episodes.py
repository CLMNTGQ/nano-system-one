"""Synthetic session fixtures for a 5 Hz behavior selector (oldest frame first)."""
import numpy as np

HISTORY = 6
# Symbol order is also the native runtime ABI: missing, absent, visible.
FEATURES = np.array([[0, 0], [0, 1], [1, 1]], dtype=np.float32)
OPTIONS = np.array([[1, 0], [0, 1], [0, 0]], dtype=np.float32)
TRACK, SEARCH, WAIT = range(3)


def make_session(seed):
    rng = np.random.default_rng(seed)
    symbols = []
    for _ in range(4):
        symbols.extend([2] * int(rng.integers(4, 11)))
        symbols.extend([1] * int(rng.integers(1, 12)))
    symbols = np.array(symbols)
    noise = rng.random(len(symbols))
    symbols[(symbols == 1) & (noise < 0.12)] = 2  # isolated false detections
    start = rng.integers(0, len(symbols) - 4)
    symbols[start:start + rng.integers(1, 5)] = 0
    return symbols


def acceptable(window):
    """Provisional design labels; zero target mass denotes controller deferral."""
    target = np.zeros(3, dtype=bool)
    if window[-1] == 0:
        return target
    hits = np.flatnonzero(window == 2)
    age = len(window) - 1 - hits[-1] if len(hits) else len(window)
    if age == 0:
        target[TRACK if np.count_nonzero(window[-3:] == 2) >= 2 else WAIT] = True
    elif age <= 2:
        target[WAIT] = True
    elif age == 3:
        target[[WAIT, SEARCH]] = True
    else:
        target[SEARCH] = True
    return target


def dataset(first_seed, count):
    windows, labels, sessions = [], [], []
    for seed in range(first_seed, first_seed + count):
        for window in np.lib.stride_tricks.sliding_window_view(make_session(seed), HISTORY):
            windows.append(window.copy())
            labels.append(acceptable(window))
            sessions.append(seed)
    return np.array(windows), np.array(labels), np.array(sessions)
