# Ball tracking · Original experiment

The following quick start and experiments reproduce the original ball decision contract. Its native lookup table and training data apply to ball tracking.

## Run System One on your MicroDuck

This quick start runs the trained model **on the robot CPU in shadow mode**. It reads ball detections and robot state, proposes `track`, `search` or `wait`, and records the controller's reported phase alongside the proposal. MicroDuck's existing controller continues to drive the robot. The shipped integration supports observation and replay; using the proposal to drive the motors would require a command adapter.

### 1. Copy the edge runtime from your computer

Open a terminal at the root of your local `nano-system-one` checkout. Replace the example SSH destination with the account and address you normally use for your duck. The board needs Python **3.11 or newer**, a C compiler, and read/connect access to its service sockets. The trained scores are already in `edge/scores.inc`; the Python adapter uses the standard library.

```sh
DUCK=microduck@192.168.1.42
ssh "$DUCK" 'mkdir -p ~/nano-system-one'
scp -r edge "$DUCK:nano-system-one/"
ssh "$DUCK"
```

### 2. Build and check one decision on the duck

Run the remaining commands **in the robot's SSH shell**:

```sh
cd ~/nano-system-one
python3 --version
cc -std=c99 -Os -shared -fPIC edge/scorer.c -o libnso.so
PYTHONPATH=edge python3 - <<'PY'
from pathlib import Path
from choice import ChoiceScorer
model = ChoiceScorer(Path("libnso.so"))
print(model.decide([2, 2, 2, 2, 1, 1], 7))
PY
```

Look for `'status': 'choice'` and `'selected': 'wait'`. This checks the actual library, model table and Python binding on your board, before the camera or robot services enter the picture. If `cc` is missing, install a C toolchain suitable for the board image or build the shared library with a matching AArch64 Linux toolchain and copy it to the same directory.

### 3. Record a ball-tracking session

Use the duck's usual autonomous ball-play mode. The scorer becomes eligible when the robot reports both autonomy and autonomous soccer, a supported controller phase, and a walking or standing policy. The operator guide describes how to check or change these modes. [MicroDuck operator guide](https://github.com/pollen-robotics/microduck/blob/main/docs/robot/cheatsheet.md)

```sh
ROBOT_SOCKET=/run/robotd.sock
MEDIA_SOCKET=/run/mediad/media.sock
ls -l "$ROBOT_SOCKET" "$MEDIA_SOCKET"
printf '{}\n' > detector-defaults.json
TRACE="microduck-$(date +%Y%m%d-%H%M%S).jsonl"
python3 edge/robot_trace.py --library ./libnso.so capture \
  --robot-socket "$ROBOT_SOCKET" --media-socket "$MEDIA_SOCKET" \
  --detector detector-defaults.json > "$TRACE"
```

The empty detector configuration requests the firmware defaults. If your detector is tuned, use its settings for comparable results. Let the duck see the ball, briefly hide it, then show it again. Stop recording with **Ctrl-C**. The shell redirects output to the trace, so a quiet terminal during capture is expected.

### 4. Verify and inspect the trace

```sh
python3 edge/robot_trace.py --library ./libnso.so replay "$TRACE"
python3 - "$TRACE" <<'PY'
import json, sys
from collections import Counter

reasons = Counter()
with open(sys.argv[1]) as trace:
    for line in trace:
        row = json.loads(line)
        decision = row["decision"]
        if decision["status"] == "choice":
            print("Model proposal:", decision, "\nController phase:", row["phase"])
            break
        reasons[decision["reason"]] += 1
    else:
        print("No eligible choice in this session. Deferrals:", dict(reasons))
PY
```

A successful first session has **rows greater than zero, zero replay mismatches, and at least one choice**. A trace containing only deferrals can also replay perfectly. `warmup` means the adapter is collecting six consecutive samples. For `missing_observation`, check the detector stream; for `operator_override` or `ineligible_controller`, check the pad and reported operating mode. A valid frame showing no ball counts as a valid observation.

If capture reports a missing socket, check `systemctl status robotd mediad` and inspect listening Unix sockets with `ss -xl`. Update the two path variables for your firmware. For a permission error, check your SSH account's access to both sockets. [Upstream socket configuration](https://github.com/pollen-robotics/microduck/blob/main/deploy/robotd.toml) · [Capture and replay contract](docs/ooda-field-validation.md)

## What comes out of the model?

The first output type is **Choice**. MicroDuck defines three identifiers in a fixed order: `track`, `search`, `wait`. The trained Transformer supplies one logit per identifier. The native selector applies the caller's admissibility mask and returns the highest-scoring allowed identifier. When a decision cannot be issued, the caller gets `NSO_DEFER`. A valid choice can still be wrong for the situation; episode replay evaluates that separate question.

For example, after four sightings and two valid misses, the exported model scores `track` at −6.490, `search` at −6.331 and `wait` at 0.739 (rounded). With all three behaviors allowed, the C call returns `NSO_WAIT`. With the same history and an empty mask it returns `NSO_DEFER`. **Waiting is a behavior. Deferring returns control to the caller.** See the [output contract and native identifiers](docs/problem.md#decision-contract).

## Understand the solution

[Follow six sensor observations through the Transformer and its structured output](docs/structured-choice.md). This design record connects tensor shapes, attention, training loss and the controller mask to three different decisions on the same window.

## Design question

When the ball disappears, does the sequence of detections and missed frames contain enough information to improve the choice between tracking, searching, and waiting? The project compares a small temporal Transformer with a 96-parameter MLP and a timeout rule. The deciding evidence will be behavior on held-out episodes and execution cost on the target device.

| Component | Current status |
|---|---|
| [MLP baseline](model.py) | Manual SGD update and a three-snapshot fitting example |
| [Temporal Transformer](transformer.py) | NumPy reference and a trained PyTorch encoder with shared candidate readout |
| MicroDuck integration | Read-only socket capture and versioned decision replay implemented; physical validation pending |

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
