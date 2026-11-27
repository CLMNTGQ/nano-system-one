# Embedded behavior selection, experiment 01

## Operating contract

The proposed behavior worker samples at 5 Hz. Each window contains six ticks in chronological order, with symbols for a missing observation, a valid miss, and a valid sighting. An upstream adapter must resample by capture timestamp, construct complete windows, check snapshot age and ownership, and recheck the selected action at dispatch. The worker is separate from the 50 Hz motor controller. No observation from a later tick is available to the scorer.

The complete input domain has 729 windows and three known candidate descriptors. The encoder isolates each candidate from the others, so pruning options preserves each surviving logit. This permits an exhaustive export: run the trained Transformer once per possible window at build time and store three float32 logits per window. The native CPU routine indexes the table in base three and selects the best admissible behavior. A tie selects the lowest admissible index. The storage cost grows exponentially with history length. New features, timing, behavior descriptors or weights require a new export. This is a CPU deployment design for the RK3566. It does not depend on NPU or RKNN availability.

## Data and training

The generator creates four tracking/loss cycles per session, variable durations, isolated detections, and an outage. Entire sessions are assigned to train (48), validation (12), or test (24) using disjoint seed ranges. A provisional rule annotates each window: a repeated recent sighting supports tracking, an isolated one supports waiting, and a sustained loss supports searching. The transition at three missed ticks accepts either waiting or searching. When the newest frame is missing, the caller defers and that tick is excluded from the loss. These labels are synthetic specifications, not human reviews or measured consequences of a behavior.

On the workstation, PyTorch trains all encoder weights with AdamW. The loss is the negative logarithm of the total softmax mass on acceptable behaviors. Validation loss selects the seed and epoch; all three seed results are retained. The MLP uses the same window reduced to four statistics. The timeout rule uses the age of the last sighting and a threshold fitted on train. The NumPy forward pass supplies the initial parity check.

## Reproduction

The recorded run used Python 3.12, NumPy 2.3.5, and PyTorch 2.15.0 CPU. From the repository root:

```sh
python -m pip install numpy torch
python train.py --output build/run
cc -std=c99 -Os -shared -fPIC edge/scorer.c -o build/libnso.so
python verify_edge.py build/libnso.so build/run/weights.npz
```

The checked-in scores.inc is the recorded export. To deploy a newly trained checkpoint, copy its generated scores.inc into edge/, rebuild, and run the verifier against that checkpoint. On AArch64 Linux, compile the same C file with the native compiler or cross-compile with a matching target. For example:

```sh
aarch64-linux-gnu-gcc -std=c99 -Os -c edge/scorer.c -o build/scorer.o
```

Link this object with the behavior worker. The C header defines the mask bits and the defer result. The call emits raw logits; apply softmax over admissible candidates only if the consumer needs a distribution. Scores are uncalibrated. The runtime owns no heap memory or mutable global state.

## Evidence and next gate

See [the full metrics](../reports/synthetic.json). The Transformer matched an acceptable label on all eligible synthetic test ticks. It made 82 switches where keeping the previous behavior would also have been acceptable, versus 48 for the MLP. Repeating only the latest frame reduced its acceptable rate to 70.64%. Those numbers measure imitation of the synthetic rubric. The episode generator and annotator share assumptions that require independent review.

For physical evaluation, collect complete sessions covering brief occlusion, sustained loss, isolated detections and stale streams. Reserve whole held-out sessions before tuning. Record capture timestamps, validity, admissibility, ownership, proposed and executed behavior, and actual reacquisition events. Review ambiguous transitions independently. Run the scorer in shadow mode first and measure end-to-end decision age, peak memory and motor deadline misses alongside normal perception load. Actual reacquisition time requires a closed-loop trial. The C export has been compiled for AArch64 and exhaustively checked on the development host; on-device latency, integration and closed-loop behavior remain unmeasured.
