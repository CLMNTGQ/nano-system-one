"""Lesson 1: trace a Transformer encoder from robot observations to choice logits."""

import numpy as np


def layer_norm(x):
    centered = x - x.mean(axis=-1, keepdims=True)
    return centered / np.sqrt((centered ** 2).mean(axis=-1, keepdims=True) + 1e-5)


def softmax_rows(x):
    exp = np.exp(x - x.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


class MicroDuckEncoder:
    """Score one decision tick. Each observation and behavior has two features."""

    def __init__(self, width=16, heads=2, max_history=16, seed=7):
        if min(width, heads, max_history) < 1 or width % heads:
            raise ValueError("Use positive dimensions and a width divisible by heads.")
        self.heads = heads
        self.width = width
        rng = np.random.default_rng(seed)

        def matrix(inputs, outputs):
            return rng.standard_normal((inputs, outputs)) / np.sqrt(inputs)

        self.observation_projection = matrix(2, width)
        self.behavior_projection = matrix(2, width)
        self.age_embedding = matrix(max_history, width)
        self.qkv = matrix(width, 3 * width)
        self.mix_heads = matrix(width, width)
        self.expand = matrix(width, 4 * width)
        self.contract = matrix(4 * width, width)
        self.readout = matrix(width, 1)

    def forward(self, observations, behaviors):
        observations = np.asarray(observations, dtype=float)
        behaviors = np.asarray(behaviors, dtype=float)
        for rows in (observations, behaviors):
            if rows.ndim != 2 or rows.shape[1] != 2 or not len(rows) or not np.isfinite(rows).all():
                raise ValueError("Expected a nonempty, finite array with two features per row.")
        history = len(observations)
        if history > len(self.age_embedding):
            raise ValueError("Observation window exceeds the age embedding table.")

        # Oldest observation first. Age zero always identifies the newest sample.
        ages = np.arange(history - 1, -1, -1)
        state = observations @ self.observation_projection + self.age_embedding[ages]
        options = behaviors @ self.behavior_projection
        x = np.concatenate([state, options])
        length = len(x)

        # A row reads every observation. Each option can also read itself.
        allowed = np.eye(length, dtype=bool)
        allowed[:, :history] = True
        qkv = (layer_norm(x) @ self.qkv).reshape(length, 3, self.heads, -1)
        query, key, value = qkv.transpose(1, 2, 0, 3)
        affinities = query @ key.swapaxes(-1, -2) / np.sqrt(query.shape[-1])
        attention = softmax_rows(np.where(allowed, affinities, -np.inf))
        context = (attention @ value).transpose(1, 0, 2).reshape(length, self.width)
        x = x + context @ self.mix_heads
        x = x + np.maximum(layer_norm(x) @ self.expand, 0) @ self.contract
        logits = (layer_norm(x[history:]) @ self.readout).ravel()
        return logits, attention


if __name__ == "__main__":
    encoder = MicroDuckEncoder()
    # Visibility and frame validity at four equally spaced observation ticks.
    window = [[1, 1], [1, 1], [0, 1], [0, 1]]
    # Uses a visible target, actively searches.
    choices = {"track_target": [1, 0], "search_target": [0, 1], "wait": [0, 0]}
    logits, attention = encoder.forward(window, list(choices.values()))
    print("Random initialization. Scores illustrate the computation.")
    print("First head, track_target attention:", attention[0, len(window)].round(3))
    for name, logit, probability in zip(choices, logits, softmax_rows(logits)):
        print(f"{name:14} logit={logit: .3f}  probability={probability:.3f}")
