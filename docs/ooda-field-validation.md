# OODA field validation

## Transition criteria for review

These provisional criteria address all eight previous Transformer disagreements, indexed by the tick of the last observation. A field reviewer should judge the sequence without seeing the model output. Preserve dissent or an acceptable transition interval rather than forcing a label.

| Development case and tick | Criterion under review |
|---|---|
| brief-occlusion, 8 | After a previously stable track and two valid misses, resume on the first credible return. |
| isolated-flash, 7 and 8 | After one isolated sighting during search, resume searching at the next valid miss. |
| camera-outage, 9 | At the first valid negative after camera recovery, allow one wait tick to gather evidence. Missing frames provide no evidence of absence. |
| noisy-return, 7 and 9 | An unconfirmed detection followed by a valid miss permits immediate resumption of search. |
| noisy-return, 8 and 10 | Alternating detections do not establish stable reacquisition; waiting and searching are acceptable. |

The original six authored cases are now development data. Five additional hand-authored episodes were set aside before fitting the context head. They share the author and design assumptions; reserve independently annotated real sessions for the final selection.

## Offline design comparison

Run `node experiments/context-replay.mjs` from a workstation. The frozen Transformer supplies three logits. A linear residual adds an intercept, age since last detection (sample ticks at 5 Hz, censored at six), count of valid negatives since that detection, recent sightings, latest visibility, missingness and prior incumbent behavior. The base logits are scaled by 0.25. Train only the residual on development fixtures with acceptable-set loss. This is a prototype context variant: the six-frame encoder itself stays frozen. Offline previous behavior is the previous **rule** output, even during candidate evaluation. Live equivalent evidence comes from the previous observed controller phase. Physical behavior or ownership cannot be inferred from the synthetic proxy.

The same episodes are evaluated by the exact rubric with continuity, a newly fitted summary MLP, the frozen Transformer, and the context variant. The MLP uses the original feature and hidden dimensions, manually implemented SGD, and trains on the original 48 synthetic sessions. Its weights differ from the original AdamW baseline. The committed report includes per-tick predictions and context weights. On held-out authored episodes the rule accepted 25/28, MLP 24/28, frozen Transformer 24/28 and contextual head 26/28. These counts include explicit deferrals. The small difference is insufficient to promote the context variant to the robot.

## Live observation and reproducible replay

Upstream autod enables soccer and yields while robotd runs the soccer behavior. Subscribe to the state that autod consumes and the detector notifications that feed robotd. The adapter only subscribes; it never sends motion or mode commands. The media subscription can itself add detection load, so collect a telemetry-only comparison with the same detector settings. [autod](https://github.com/pollen-robotics/microduck/blob/4024f6ee4373d38ac0a60a79157bcfb57642be9c/autod/src/main.rs) · [soccer implementation](https://github.com/pollen-robotics/microduck/blob/4024f6ee4373d38ac0a60a79157bcfb57642be9c/robotd/src/soccer.rs) · [wire types](https://github.com/pollen-robotics/microduck/blob/4024f6ee4373d38ac0a60a79157bcfb57642be9c/duck-ipc-proto/src/lib.rs)

Requires Linux, Python 3.11+, access to the two service sockets, and the native C scorer. Supply socket paths from the installed robot configuration. Use the installed soccer detector settings for the field test; an empty JSON object requests upstream defaults.

```sh
cc -std=c99 -Os -shared -fPIC edge/scorer.c -o libnso.so
python3 edge/robot_trace.py --library ./libnso.so capture \
  --robot-socket "$ROBOT_SOCKET" --media-socket "$MEDIA_SOCKET" \
  --detector detector.json > session.jsonl
python3 edge/robot_trace.py --library ./libnso.so replay session.jsonl
```

At 5 Hz the trace records the newest look received within the period, or an explicit missing symbol. It stores robotd phase, incumbent interpretation, head/walk flags, source, eligibility, loop misses, proposal and monotonic timestamps. Subscribe to independent looks: target_m can persist during occlusion. The detector protocol supplies no capture timestamp or frame identity, so timing starts at local detector message receipt. The reported state sensor timestamp belongs to robotd. A phase is a reported controller state; head/walk indicate actuation involvement, not proof of physical ball reacquisition. Unknown phases, including kicking, defer. Shadow eligibility is conservative and advisory; it is not dispatch authorization. Transport termination ends the session. Replay uses recorded eligibility and symbols, independent of wall-clock time.

Accept a trace only if replay has zero mismatches, a loss/return is visible, and the trace includes valid phase coverage. Report ingress-to-proposal p95/p99 and counter deltas for comparable durations. Capture a baseline motor deadline counter without the observer as well. True exposure-to-decision latency requires an upstream camera timestamp.

## Paired physical trials

Use the [paired design](replay-and-trials.md). Agree a timeout and confirmed reacquisition criterion with the operator before running either arm (e.g. three consecutive fresh visible looks following physical reappearance). Preserve the trial outcome, operator abort reason, controller and firmware versions, and synchronized trace. Alternate arm order. The shadow adapter has no dispatch path. Controlled candidate trials require a separately integrated action adapter and an operator with physical access.

One JSONL row per arm contains condition, pair, arm (incumbent/candidate), status (success/timeout/abort), timeout_ms, reacquisition_ms (null for unsuccessful trials), switches, missed_deadlines and reason for abort. Run `python3 edge/paired_trials.py trials.jsonl` for per-condition results, discordant and incomplete pairs, and paired latency differences. Compare actual success and timing before selecting a deployed policy.
