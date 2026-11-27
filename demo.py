"""Fit three hand-labeled snapshots to illustrate the offline interface."""

import numpy as np

from model import CandidateScorer, softmax


def main():
    names = ("track_target", "search_target", "wait")
    # Columns: uses a visible target, actively searches.
    candidates = np.array([[1, 0], [0, 1], [0, 0]])
    # State: current visibility, seconds since last seen / 3 (clipped to 1).
    examples = [([1, 0], 0), ([0, 0.05], 2), ([0, 1], 1)]
    model = CandidateScorer(state_size=2, candidate_size=2)
    for _ in range(400):
        for state, label in examples:
            model.train_step(state, candidates, label, learning_rate=0.08)
    for state, label in examples:
        probabilities = softmax(model.logits(state, candidates))
        scores = ", ".join(f"{name}={p:.3f}" for name, p in zip(names, probabilities))
        print(f"state={state} expected={names[label]} {scores}")


if __name__ == "__main__":
    main()
