# nano-system-one

A small decision model built from scratch for an edge robot. Given a short observation history and a set of available behaviors, it scores each candidate in one forward pass. MicroDuck ball tracking is the first experiment.

## Run the first example

With Python 3.10+ and NumPy installed (`python -m pip install numpy`), run:

```sh
python demo.py
```

The example trains a shared MLP on three illustrative snapshots and prints the probability of each behavior for those same snapshots. It demonstrates the training and inference contract. The scores are in-sample and uncalibrated. The example runs entirely on a development machine.

- [Problem and proposed solution](docs/problem.md)
- [Architecture and evaluation plan](docs/architecture.md)
- [Model and learning rule](model.py)

## Build the Transformer

Read [Lesson 1: From a missed detection to three logits](docs/lesson-01.md) alongside [the encoder](transformer.py). Run `python transformer.py` to trace a two-head attention block and the choice distribution at random initialization. The MLP above provides the first comparison point.
