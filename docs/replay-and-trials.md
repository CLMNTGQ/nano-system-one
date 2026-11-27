# MicroDuck decision audit and physical trial design

The JSONL files in data/ contain all 84 original synthetic sessions, split before windowing. One record contains a session's raw ternary frames and tick-level annotations. To recover a labeled input at tick t, take frames[t-5:t+1]. The symbol order is missing, valid miss, valid sighting (0, 1, 2). Reason codes identify the branch of the training rubric. An empty accepted list denotes controller deferral. The short names track, search and wait correspond to the C enum order.

## What the 82 switches mean

All 82 original test switches counted by the metric take the same form. The previous proposal was wait. At the third missed tick the acceptable set admits wait and search; the model selects search. The 82 rows in [the audit](../reports/switches.jsonl) include the exact window and session. None is evidence of repeated back-and-forth physical movement. The metric counts a switch while the previous behavior was still acceptable. Whether early search helps depends on how often the ball reappears and what the search costs.

## Independent design fixtures

[Six handwritten scenarios](../data/review-cases.jsonl) specify 34 decisions including three deferrals. These are author-proposed cases, independent of the label function in code; human annotation and real sessions are pending. Each tick has an explicit list of acceptable behaviors. The rule baseline uses the exact training acceptable set, retaining its previous action when allowed and otherwise preferring wait on an ambiguous boundary. The timeout threshold of three ticks is the recorded training selection. Each predictor starts each episode from scratch.

The [tick-level replay](../reports/review-replay.jsonl) permits review of every disagreement. The Transformer passes 26 of 34, the rule 25, and the timeout 24 (including three straightforward deferrals for each). On isolated-flash and noisy-return, the distinction is whether one sighting should interrupt an existing search. Following an outage, elapsed ticks and valid negative observations also encode different evidence. Resolve these choices with independent reviewers before treating this small fixture as a product requirement. Keep a separate blind test when iterating.

To regenerate the splits, switch audit, and review results with the original summary MLP refitted on its original training and validation splits:

```sh
python audit_replay.py --output build/audit --with-mlp
```

The saved model export contains only Transformer weights in table form. Refitting the MLP uses the same PyTorch version and CPU settings as the [training experiment](edge-training.md). The replay workflow attaches its generated files as an artifact and prints the four counts.

## OODA and observation-only integration

Observe: the adapter supplies resampled observations on the original 5 Hz grid. Orient: the sidecar builds the six-frame history. Decide: it logs the scorer proposal. Act: the existing controller remains the command writer; the adapter records what it executed. Join consecutive records by session, ownership generation and tick to observe what followed each action. Track proposal versus execution separately.

Build the native library on the target, then run the sidecar in a separate process:

```sh
cc -Os -shared -fPIC edge/scorer.c -o libnso.so
python3 edge/shadow.py ./libnso.so --max-age-ms 200 < telemetry.jsonl > shadow.jsonl
```

Use a live telemetry stream for latency measurements. For each tick the adapter must emit session, integer tick, capture_mono_ns (same host monotonic clock), symbol (0/1/2), admissible_mask (bits track/search/wait), ownership_generation, autonomous (boolean), and executed (actual behavior). The example age budget is a configurable trial parameter. Stored timestamps may be stale during offline playback. An episode boundary, gap, timestamp reversal or ownership change clears the window. Missing frames are explicit symbols. The adapter should use bounded nonblocking telemetry outside the motor loop and increment tick across dropped samples. The sidecar has no command interface. Connecting actual upstream telemetry and validating clock domains require a MicroDuck installation.

## Paired closed-loop trials

Compare the incumbent autonomous controller and candidate selector on matched starting pose, ball trajectory, lighting, occlusion duration and operator procedure. Alternate controller order within pairs. Include brief occlusions, sustained loss, isolated detections and camera outages. Define a trial timeout and what constitutes confirmed reacquisition before running. Record controller and firmware versions, elapsed time from return of the ball to confirmed reacquisition, unsuccessful trials at timeout, behavior switches, and motor deadline misses. Preserve aborted trials with their reason. Compare per-condition success rates and paired time differences for pairs where both succeed; report censored and discordant pairs separately. In shadow mode report sensor-to-proposal p95/p99 and missed motor deadlines relative to a worker-disabled run. Closed-loop conclusions require actual dispatch and observations of its consequences.
