"""Train on a workstation, compare on episode splits, and export the edge scorer."""
import argparse
import copy
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from episodes import FEATURES, HISTORY, OPTIONS, SEARCH, TRACK, WAIT, dataset
from transformer import MicroDuckEncoder


def norm(x):
    return F.layer_norm(x, (x.shape[-1],), eps=1e-5)


class TemporalScorer(nn.Module):
    def __init__(self, seed):
        super().__init__()
        ref = MicroDuckEncoder(max_history=HISTORY, seed=seed)
        for name, value in vars(ref).items():
            if isinstance(value, np.ndarray):
                setattr(self, name, nn.Parameter(torch.tensor(value, dtype=torch.float32)))
        self.register_buffer("options", torch.tensor(OPTIONS))

    def forward(self, obs):
        batch, ticks, _ = obs.shape
        state = obs @ self.observation_projection + self.age_embedding[:ticks].flip([0])
        candidates = (self.options @ self.behavior_projection).expand(batch, -1, -1)
        x = torch.cat([state, candidates], dim=1)
        length = x.size(1)
        qkv = (norm(x) @ self.qkv).reshape(batch, length, 3, 2, 8)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        allowed = torch.eye(length, dtype=torch.bool, device=x.device)
        allowed[:, :ticks] = True
        scores = (q @ k.transpose(-1, -2)) / 8 ** 0.5
        weights = scores.masked_fill(~allowed, -torch.inf).softmax(dim=-1)
        context = (weights @ v).transpose(1, 2).reshape(batch, length, 16)
        x = x + context @ self.mix_heads
        x = x + F.relu(norm(x) @ self.expand) @ self.contract
        return (norm(x[:, ticks:]) @ self.readout).squeeze(-1)


class SummaryScorer(nn.Module):
    """Same shared MLP design as the snapshot baseline, with window summaries."""
    def __init__(self):
        super().__init__()
        self.register_buffer("options", torch.tensor(OPTIONS))
        self.score = nn.Sequential(nn.Linear(6, 16), nn.Tanh(), nn.Linear(16, 1))

    def forward(self, obs):
        visible, valid = obs.unbind(-1)
        age_grid = torch.arange(HISTORY - 1, -1, -1)
        age = torch.where(visible.bool(), age_grid, HISTORY).min(dim=1).values
        state = torch.stack([visible[:, -1], age / HISTORY, visible.mean(1), valid.mean(1)], 1)
        state = state[:, None].expand(-1, 3, -1)
        options = self.options.expand(len(obs), -1, -1)
        return self.score(torch.cat([state, options], -1)).squeeze(-1)


def loss(logits, accepted):
    return (logits.logsumexp(-1) - logits.masked_fill(~accepted, -torch.inf).logsumexp(-1)).mean()


def tensors(split):
    windows, labels, _ = split
    keep = labels.any(1)
    return torch.tensor(FEATURES[windows[keep]]), torch.tensor(labels[keep])


def fit(kind, train, valid, seeds=(7, 17, 27)):
    x, y = tensors(train)
    vx, vy = tensors(valid)
    best_loss = float("inf")
    runs = {}
    for seed in seeds:
        torch.manual_seed(seed)
        model = TemporalScorer(seed) if kind == "transformer" else SummaryScorer()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
        seed_best = float("inf")
        for epoch in range(12):
            for idx in torch.randperm(len(x)).split(128):
                optimizer.zero_grad()
                loss(model(x[idx]), y[idx]).backward()
                optimizer.step()
            with torch.no_grad():
                score = loss(model(vx), vy).item()
            seed_best = min(seed_best, score)
            if score < best_loss:
                best = copy.deepcopy(model).eval()
                best_loss, selected = score, (seed, epoch + 1)
        runs[seed] = round(seed_best, 6)
    return best, {"selected_seed_epoch": selected, "validation_loss_by_seed": runs}


def timeout(windows, threshold):
    ages = np.where(windows == 2, np.arange(HISTORY)[::-1], HISTORY).min(1)
    return np.where(ages == 0, TRACK, np.where(ages >= threshold, SEARCH, WAIT))


def metrics(split, predictions):
    windows, labels, sessions = split
    usable = labels.any(1)
    correct = labels[np.arange(len(labels)), predictions]
    adjacent = (sessions[1:] == sessions[:-1]) & usable[1:] & usable[:-1]
    redundant = (predictions[1:] != predictions[:-1]) & labels[np.arange(1, len(labels)), predictions[:-1]]
    required = labels[:, SEARCH] & ~labels[:, WAIT]
    delays = []
    missed = 0
    for start in np.flatnonzero(required & ~np.r_[False, required[:-1] & (sessions[:-1] == sessions[1:])]):
        end = start + 1
        while end < len(required) and required[end] and sessions[end] == sessions[start]:
            end += 1
        found = np.flatnonzero(predictions[start:end] == SEARCH)
        if len(found):
            delays.append(int(found[0]))
        else:
            missed += 1
    return {"acceptable_rate": round(float(correct[usable].mean()), 4),
            "premature_search_ticks": int(np.sum(usable & ~labels[:, SEARCH] & (predictions == SEARCH))),
            "unnecessary_switches": int(np.sum(adjacent & redundant)),
            "search_delay_ticks_when_found": float(np.mean(delays)) if delays else None,
            "missed_search_intervals": missed, "search_intervals": len(delays) + missed,
            "deferred_ticks": int((~usable).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    train, valid, test = (dataset(start, n) for start, n in [(0, 48), (1000, 12), (2000, 24)])
    ref = MicroDuckEncoder(max_history=HISTORY, seed=7)
    sample = FEATURES[test[0][0]]
    np.testing.assert_allclose(TemporalScorer(7)(torch.tensor(sample[None])).detach()[0],
                               ref.forward(sample, OPTIONS)[0], atol=2e-5)
    threshold = max(range(1, HISTORY + 1), key=lambda t: metrics(train, timeout(train[0], t))["acceptable_rate"])
    report = {"timeout": {"threshold": threshold, **metrics(test, timeout(test[0], threshold))}}
    for kind in ("mlp", "transformer"):
        model, selection = fit(kind, train, valid)
        with torch.no_grad():
            obs = torch.tensor(FEATURES[test[0]])
            logits = model(obs)
            keep = test[1].any(1)
            report[kind] = {**selection, **metrics(test, logits.argmax(1).numpy()),
                            "test_set_loss": float(loss(logits[keep], torch.tensor(test[1][keep])))}
            if kind == "transformer":
                ablated = obs[:, -1:].expand_as(obs)
                report["last_frame_repeated"] = metrics(test, model(ablated).argmax(1).numpy())
                domain = np.array(list(itertools.product(range(3), repeat=HISTORY)))
                table = model(torch.tensor(FEATURES[domain])).numpy()
                np.save(out / "logits.npy", table)
                np.savez(out / "weights.npz", **{k: v.numpy() for k, v in model.state_dict().items()})
                with (out / "scores.inc").open("w") as file:
                    for row in table:
                        file.write("{" + ", ".join(f"{v:.9e}f" for v in row) + "},\n")
    result = json.dumps(report, indent=2)
    (out / "metrics.json").write_text(result + "\n")
    print(result)


if __name__ == "__main__":
    main()
