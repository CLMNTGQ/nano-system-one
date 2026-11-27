# nano-system-one

![Three MicroDuck robots in a harbor training hangar, with a red ball beside the central robot.](images/nano-system-one-banner.webp)

nano-system-one is a from-scratch experiment in **structured decision output on an edge robot**. Each inference answers a bounded question: given recent observations, which allowed behavior should the MicroDuck propose? The controller consumes a behavior identifier and retains authority to act.

## What comes out of the model?

The first output type is **Choice**. MicroDuck defines three identifiers in a fixed order: `track`, `search`, `wait`. The trained Transformer supplies one logit per identifier. The native selector applies the caller's admissibility mask and returns the highest-scoring allowed identifier. When a decision cannot be issued, the caller gets `NSO_DEFER`. A valid choice can still be wrong for the situation; episode replay evaluates that separate question.

For example, after four sightings and two valid misses, the exported model scores `track` at −6.490, `search` at −6.331 and `wait` at 0.739 (rounded). With all three behaviors allowed, the C call returns `NSO_WAIT`. With the same history and an empty mask it returns `NSO_DEFER`. **Waiting is a behavior. Deferring returns control to the caller.** See the [output contract and native identifiers](docs/problem.md#decision-contract).

## Design question

When the ball disappears, does the sequence of detections and missed frames contain enough information to improve the choice between tracking, searching, and waiting? The project compares a small temporal Transformer with a 96-parameter MLP and a timeout rule. The deciding evidence will be behavior on held-out episodes and execution cost on the target device.

| Component | Current status |
|---|---|
| [MLP baseline](model.py) | Manual SGD update and a three-snapshot fitting example |
| [Temporal Transformer](transformer.py) | NumPy reference and a trained PyTorch encoder with shared candidate readout |
| MicroDuck integration | Synthetic replay completed; native AArch64 scorer cross-compiled; physical integration pending |

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

## Train off-device and run on the robot CPU

The [training and deployment record](docs/edge-training.md) describes the synthetic episode splits, comparisons, and the C inference interface. The first trained model is shipped as an exhaustive table of Transformer logits for six ternary sensor observations. Its 8,748-byte payload lets an ARM64 caller score an admissible subset of behaviors locally. The native component can be built without Python. Robot integration and physical evaluation are pending.

## Inspect the episodes and decisions

[Training episodes](data/train.jsonl) · [Validation episodes](data/validation.jsonl) · [Test episodes](data/test.jsonl) · [Independent design fixtures](data/review-cases.jsonl). See the [replay audit, OODA telemetry contract and paired robot trials](docs/replay-and-trials.md) for reproduction and integration details.

## Field validation

[Transition criteria, contextual replay results and live OODA capture](docs/ooda-field-validation.md) · [Paired decision traces](reports/context-replay.json). The live subscriber and closed-loop analysis await measurements from a physical robot.

## Timing and architecture ablation

[Matched architecture comparison and field gates](docs/timing-ablation.md) · [Held-out timing episodes](data/timing-holdout.jsonl) · [Decision-level results](reports/timing-ablation.json).
