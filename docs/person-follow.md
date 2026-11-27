# Following one person with structured decisions

The first trial starts with one consenting participant in view. They walk away, stop, turn sideways and leave the frame. The operator starts and ends the session. A successful eventual behavior would resume following as the gap opens, hold position when the gap closes and cancel motion when the target is lost. Physical stopping distance and distance to the person require measurements on the robot.

## OODA and the controller boundary

| Stage | Current implementation |
|---|---|
| Observe | OpenCV's built-in upright-person HOG detector reads recorded constant-frame-rate video or grayscale luminance from MicroDuck's local UYVY camera socket. Its sigmoid-transformed SVM margin is an uncalibrated quality indicator. |
| Orient | Require one candidate, reject large horizontal jumps between recent boxes, and normalize the box using upright image dimensions. Box height is a proximity proxy. Posture and camera pitch can both change it. After a loss, six consecutive valid samples are needed again. Identity persistence across people is outside the single-person trial. |
| Decide | Replay six ordered samples with either the hysteresis rule, the MLP or a one-block Transformer. Produce a structured choice with a reason and an admissibility mask. |
| Act | Shadow output only. A future controller must accept a fresh proposal before issuing a bounded motion intent. It owns obstacle checks, manual priority, motion expiration and cancellation on disconnect. |

The upstream protocol exposes camera frames via local `media.frame`, a JSON header followed by raw bytes. The future controller can investigate `robot.move` and `robot.stop` once ownership and expiration have been validated on the installed firmware. An absent observation stream needs an independent controller watchdog: the replay function only runs when a row arrives. [Pollen IPC protocol](https://github.com/pollen-robotics/microduck/blob/main/duck-ipc-proto/src/lib.rs)

## Capture and inspect the first physical episode

Run from the repository root with Python 3 and NumPy. Capture additionally requires Python OpenCV with HOG support (for Debian-based images the packages are commonly named `python3-numpy` and `python3-opencv`). Use the Python interpreter where both imports succeed. On the robot, use the socket configured by its installed firmware and an account with read access. The camera client fetches a frame per iteration. Its achieved rate includes inference and socket time.

```sh
python3 -c 'import cv2, numpy; print(cv2.__version__)'
python3 -m person_follow.capture --socket /run/mediad/media.sock > session.jsonl
# Press Ctrl-C after the participant has stopped and left the frame.
python3 -m person_follow session.jsonl > rule-decisions.jsonl
python3 -m person_follow session.jsonl --model person_follow/checkpoints/transformer.json > model-decisions.jsonl
```

On a workstation use `python3 -m person_follow.capture --video session.mp4 --show > session.jsonl` to inspect candidate boxes. Video uses source time and zero transport age. Live input uses the capture clock and includes detector processing time in age. Start with a stationary robot, record the participant's approximate distances separately, then repeat while an operator drives and turns. Compare a run with and without capture for CPU, memory and control-loop timing. Check detected box heights, frame intervals and age before interpreting decisions. Multiple boxes, substantial jumps and missed detections break the observation history. HOG performance from duck height and on-board throughput are open field questions.

## A trace from observation to choice

Each JSONL file is one episode. A row carries `version`, increasing `t` in seconds, `age_s`, normalized `target=[x,y,width,height]` or null, and `quality`. Optional `override=true` marks an operator takeover during replay. The current camera recorder does not subscribe to pad state. In the synthetic example the sixth sample completes a window. At t = 1.0 s the supplied Transformer gives approximately 7.28, 4.87 and −21.01 in action order. All three are allowed, so it selects `follow`. At t = 4.2 s the target is missing; the explicit loss rule selects `stop` before evaluating the network.

| Output | Meaning |
|---|---|
| follow | Propose approach. The future motion controller will decide heading and speed. |
| hold_distance | Keep tracking the target while suspending forward motion. |
| stop | Request cancellation of follow motion. |
| defer | Return no proposal, for example on a recorded manual override. The controller owns the handover. |

The token contains horizontal center, height, quality, visibility, sample interval and age. The current loss guard admits only visible histories; visibility is therefore constant within learned windows. The vector sequence is projected to width 12 and receives learned positions. The last token queries all six tokens. Only its output is needed for this single decision. The attention and residual feed-forward block are implemented and differentiated in NumPy. In row-vector notation:

```text
E = X W_embed + P                          # 6 × 12
q = E[last] W_q; K = E W_k; V = E W_v
a = softmax(K qᵀ / sqrt(12))               # 6 attention weights
u = E[last] + aᵀ V
z = u + tanh(u W_ff) W_project
scores = z W_out + bias                    # follow, hold_distance, stop
choice = argmax(scores[i] for allowed i)
```

A stale or unreliable observation requests stop. A normalized box height of at least 0.65 masks follow. These provisional geometry thresholds need field calibration. An empty history or a gap over 0.5 s restarts warm-up. Decision replay and stored checkpoints are versioned independently of the original ball runtime.

## Train and evaluate

```sh
python3 -m person_follow.train --out /tmp/person-follow-run
python3 -m unittest person_follow.test_policy
```

The reproducible generator in `person_follow/train.py` labels simulated episodes using latent proximity and trend before adding observation noise. Seed ranges 0–79, 100–119 and 200–219 generate independent train, validation and test episodes. The MLP and Transformer minimize cross-entropy with Adam; validation loss selects the checkpoint. All candidates receive the same test episodes and admissibility rules. See the [machine-readable report](../person_follow/checkpoints/results.json). All-tick accuracy includes warm-up. Eligible accuracy covers complete reliable windows. Count unwanted forward proposals and follow/hold transitions as well: a high agreement score can hide oscillations. Loss-to-stop is measured at the observation timestamps, excluding camera latency and physical braking.

To train on real traces, label the acceptable decision at each reviewed timestamp with a `label` field. Omit the label for an ambiguous interval. Retain the full observation sequence for context. Make a JSON manifest with one entry per episode, using paths relative to that manifest:

```json
{"split":"train","session":"walk-01","person":"p01","site":"room-a","path":"walk-01.jsonl"}
```

Wrap entries in a JSON array and populate all three splits (`train`, `val`, `test`). The loader rejects reuse of a session, person or site across splits. Run `python3 -m person_follow.train --manifest manifest.json --out real-run`. Keep the test episodes for final comparison. In active trials, add external distance measurements, time in the desired band, loss-to-physical-stop latency and operator intervention counts. Release an active mode only after reviewing actual captures, measuring board load and verifying controller arbitration.
