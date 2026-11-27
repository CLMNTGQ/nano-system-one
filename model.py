"""Shared candidate scorer and one-example SGD update, using NumPy."""

import numpy as np


class CandidateScorer:
    def __init__(self, state_size, candidate_size, hidden_size=16, seed=0):
        self.state_size = state_size
        self.candidate_size = candidate_size
        width = state_size + candidate_size
        rng = np.random.default_rng(seed)
        self.input_weights = rng.normal(0, width ** -0.5, (width, hidden_size))
        self.hidden_bias = np.zeros(hidden_size)
        self.output_weights = rng.normal(0, hidden_size ** -0.5, hidden_size)

    def _encode(self, state, candidates):
        state = np.asarray(state, dtype=float)
        candidates = np.asarray(candidates, dtype=float)
        if state.shape != (self.state_size,):
            raise ValueError("Expected one state vector")
        if candidates.ndim != 2 or candidates.shape[1] != self.candidate_size or not len(candidates):
            raise ValueError("Expected a nonempty matrix of candidate descriptors")
        if not np.isfinite(state).all() or not np.isfinite(candidates).all():
            raise ValueError("Features must be finite")
        repeated_state = np.broadcast_to(state, (len(candidates), self.state_size))
        inputs = np.concatenate((repeated_state, candidates), axis=1)
        hidden = np.tanh(inputs @ self.input_weights + self.hidden_bias)
        return inputs, hidden

    def logits(self, state, candidates):
        _, hidden = self._encode(state, candidates)
        return hidden @ self.output_weights

    def train_step(self, state, candidates, label, learning_rate):
        """Minimize categorical cross-entropy over the supplied candidates."""
        inputs, hidden = self._encode(state, candidates)
        if not isinstance(label, (int, np.integer)) or not 0 <= label < len(hidden):
            raise ValueError("Label must index an available candidate")
        gradient = softmax(hidden @ self.output_weights)
        gradient[label] -= 1
        hidden_gradient = gradient[:, None] * self.output_weights * (1 - hidden ** 2)
        self.output_weights -= learning_rate * (hidden.T @ gradient)
        self.input_weights -= learning_rate * (inputs.T @ hidden_gradient)
        self.hidden_bias -= learning_rate * hidden_gradient.sum(axis=0)


def softmax(logits):
    weights = np.exp(logits - np.max(logits))
    return weights / weights.sum()
