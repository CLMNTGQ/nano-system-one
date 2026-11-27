"""Train off-device on separate episode manifests or generated proxy episodes."""
import argparse
import json
from pathlib import Path

import numpy as np
from .model import Network
from .policy import ACTIONS, VERSION, WINDOW, observations, replay


def synthetic(seed):
    rng = np.random.default_rng(seed)
    height = rng.uniform(.28,.62)
    x = 0.
    rows = []
    for i in range(50):
        if i % 10 == 0:
            velocity = rng.choice([-.013, 0., .013])
        height = np.clip(height + velocity,.24,.75)
        x = np.clip(x+rng.normal(0,.03),-.5,.5)
        h = np.clip(height+rng.normal(0,.012),.2,.8)
        visible = i % 23 != 21
        # Labels use latent proximity and approach trend, before sensor noise.
        action = "stop" if not visible or height > .66 else (
            "follow" if height+3*velocity < .445 else "hold_distance")
        rows.append(dict(version=VERSION,t=i/5,age_s=.015,quality=.9 if visible else 0.,
                         target=[.5+x/2-h/5,.08,h*.4,h] if visible else None,label=action))
    return rows


def load_splits(manifest):
    if not manifest:
        return {name:[synthetic(seed) for seed in seeds] for name,seeds in
                (("train",range(80)),("val",range(100,120)),("test",range(200,220)))}
    splits = {name:[] for name in ("train","val","test")}
    owners = {}
    for ep in json.loads(manifest.read_text()):
        split = ep["split"]
        path = (manifest.parent/ep["path"]).resolve()
        for field,value in [("path",str(path))]+[(k,ep[k]) for k in ("session","person","site")]:
            if owners.setdefault((field,value),split) != split:
                raise ValueError(f"{field} leaks across splits: {value}")
        splits[split].append(list(observations(path)))
    return splits


def samples(episodes):
    x,y = [],[]
    for rows in episodes:
        for row,(decision,hist) in zip(rows,replay(rows)):
            if len(hist) == WINDOW and row.get("label") in ACTIONS:
                x.append(hist)
                y.append(ACTIONS.index(row["label"]))
    if not x:
        raise ValueError("Split has no eligible labeled windows")
    return np.array(x),np.array(y)


def fit(kind, train, val):
    x,y = train
    net = Network(kind)
    rng = np.random.default_rng(99)
    m = {k:np.zeros_like(v) for k,v in net.p.items()}
    v = {k:np.zeros_like(w) for k,w in net.p.items()}
    best_loss, step = np.inf, 0
    for epoch in range(35):
        for batch in np.array_split(rng.permutation(len(x)),max(1,len(x)//64)):
            logits, cache = net.forward(x[batch])
            prob = np.exp(logits-logits.max(1,keepdims=True))
            prob /= prob.sum(1,keepdims=True)
            prob[np.arange(len(batch)),y[batch]] -= 1
            step += 1
            for k,g in net.gradients(cache,prob/len(batch)).items():
                m[k] = .9*m[k] + .1*g
                v[k] = .999*v[k] + .001*g*g
                net.p[k] -= .003*(m[k]/(1-.9**step))/(np.sqrt(v[k]/(1-.999**step))+1e-8)
        z = net(val[0])
        z -= z.max(1,keepdims=True)
        loss = np.mean(np.log(np.exp(z).sum(1))-z[np.arange(len(z)),val[1]])
        if loss < best_loss:
            best_loss,best = loss,{k:w.copy() for k,w in net.p.items()}
    net.p = best
    return net


def evaluate(episodes, model):
    total = correct = advances = switches = 0
    eligible = eligible_correct = 0
    stop_delays = []
    for rows in episodes:
        loss_start = None
        previously_visible = False
        previous = None
        for row,(d,history) in zip(rows,replay(rows,model)):
            visible = row["target"] is not None
            if previously_visible and not visible:
                loss_start = row["t"]
            if loss_start is not None and d["selected"] == "stop":
                stop_delays.append(row["t"] - loss_start)
                loss_start = None
            previously_visible = visible
            label = row.get("label")
            if label not in ACTIONS:
                continue
            selected = d["selected"]
            correct += selected == label
            total += 1
            if len(history) == WINDOW:
                eligible += 1
                eligible_correct += selected == label
            advances += selected == "follow" and label != "follow"
            switches += {selected,previous} == {"follow","hold_distance"}
            previous = selected
    return dict(labeled_ticks=total,accuracy=correct/total,eligible_accuracy=eligible_correct/eligible,
                unwanted_follow_ticks=advances,follow_hold_switches=switches,
                loss_to_stop_completed=len(stop_delays),
                max_observation_loss_to_stop_s=max(stop_delays,default=None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    splits = load_splits(args.manifest)
    train,val = samples(splits["train"]),samples(splits["val"])
    samples(splits["test"])
    args.out.mkdir(parents=True,exist_ok=True)
    report = dict(source="annotated manifest" if args.manifest else "synthetic proxy",
                  episodes={k:len(v) for k,v in splits.items()},test={})
    for kind in ("rule","mlp","transformer"):
        model = None if kind=="rule" else fit(kind,train,val)
        if model:
            model.save(args.out/(kind+".json"))
        report["test"][kind] = evaluate(splits["test"],model)
    (args.out/"results.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
