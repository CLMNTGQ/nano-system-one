# nano-system-one

An experimental decision model for edge robotics. The first application is MicroDuck ball tracking. Given recent observations and the behaviors currently available, the model scores each behavior in one forward pass. The robot controller decides whether to act.

## Design question

When the ball disappears, does the sequence of detections and missed frames contain enough information to improve the choice between tracking, searching, and waiting? The project compares a small temporal Transformer with a 96-parameter MLP and a timeout rule. The deciding evidence will be behavior on held-out episodes and execution cost on the target device.

| Component | Current status |
|---|---|
| [MLP baseline](model.py) | Manual SGD update and a three-snapshot fitting example |
| [Temporal Transformer](transformer.py) | NumPy forward pass with an explicit attention mask and shared candidate readout |
| MicroDuck integration | Proposed asynchronous interface; episode replay and device benchmarks are pending |

## Design documents

- [Problem and decision contract](docs/problem.md)
- [Encoder design decisions and validation criteria](docs/encoder-design.md)
- [Runtime architecture and evaluation plan](docs/architecture.md)

## Run locally

With Python 3.10+ and NumPy installed, run the two prototypes from the repository root:

```sh
python demo.py
python transformer.py
```

The MLP fits three illustrative snapshots. The encoder prints an attention row, logits, and choice probabilities at random initialization. These outputs make the computation inspectable. Assessing behavior quality requires episode data.
