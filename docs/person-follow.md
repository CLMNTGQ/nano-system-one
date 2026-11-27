# Following one person with structured decisions

The first trial starts with one consenting participant in view. They walk away, stop, turn sideways and leave the frame. The operator starts and ends the session. A successful eventual behavior would resume following as the gap opens, hold position when the gap closes and cancel motion when the target is lost. Physical stopping distance and distance to the person require measurements on the robot.

## OODA and the controller boundary

| Stage | Current implementation |
|---|---|
| Observe | OpenCV's built-in upright-person HOG detector reads recorded constant-frame-rate video or grayscale luminance from MicroDuck's local UYVY camera socket. Its sigmoid-transformed SVM margin is an uncalibrated quality indicator. |
| Orient | Require one candidate, reject large horizontal jumps between recent boxes, and normalize the box using upright image dimensions. Box height is a proximity proxy. Posture and camera pitch can both change it. After a loss, six consecutive valid samples are needed again. Identity persistence across people is outside the single-person trial. |
| Decide | Replay six ordered samples with either the hysteresis rule, the MLP or a one-block Transformer. Produce a structured choice with a reason and an admissibility mask. |
| Act | The active client converts each fresh accepted choice to bounded forward and yaw velocities, checks forward depth and robot state, and renews the motion intent using fresh feedback. |

The upstream protocol exposes camera frames via local `media.frame`, a JSON header followed by raw bytes. The client identifies its command connection with `hello`, subscribes to `robot.state` and `tof.frame`, and sends `robot.move` notifications. A zero twist releases active motion. The independent control tick continues even if the camera stalls. [Pollen IPC protocol](https://github.com/pollen-robotics/microduck/blob/main/duck-ipc-proto/src/lib.rs)

## Install and check on the robot

From a local checkout on your computer, substitute your duck's usual SSH destination. These commands copy the person-follow package and both checkpoints.

```sh
DUCK=microduck@192.168.1.42
ssh "$DUCK" 'mkdir -p ~/nano-system-one'
scp -r person_follow "$DUCK:nano-system-one/"
ssh "$DUCK"
```

Run the remaining commands **in the robot's SSH shell**. On a Debian-based image, install missing dependencies with `sudo apt install python3-numpy python3-opencv` if package management is available. Use the system Python with these packages.

```sh
cd ~/nano-system-one
python3 -c 'import cv2, numpy; print(cv2.__version__, numpy.__version__); assert len(cv2.HOGDescriptor_getDefaultPeopleDetector())'
ls -l /run/mediad/media.sock /run/tofd/tof.sock /run/robotd.sock
python3 -m person_follow person_follow/example.jsonl --model person_follow/checkpoints/transformer.json | sed -n '6p;22p'
```

Expect `follow` at 1.0 s and `stop` at 4.2 s. This checks the checkpoint and inference path. The next step checks the actual camera. If a socket is missing, inspect `ss -xl` and the corresponding service; use the paths configured on your firmware. An access error means the SSH account needs permission to connect to the service socket.

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

## Run the active loop

Copy the `person_follow/` directory, including `checkpoints/`, to the duck. Run from the directory containing it with Python 3, NumPy and OpenCV with HOG support. Check that camera capture yields reliable boxes first. Start the stock walking policy in a standing posture with the head centered. Use a level, open area and one participant. Switch off native autonomy, soccer and laser tracking. The client uses local read access to mediad and tofd and command access to robotd (API v60 or later).

```sh
python3 -m person_follow.active > follow-trial.jsonl
```

The first six consecutive valid observations warm up the model. Walk away, pause, step slightly to one side and leave the frame. Inspect `decision.selected`, `control`, `twist` and the subsequently reported `reported_move.applied` in the trace. The requested twist and the robot's telemetry may differ. Measure distance and actual stopping time during this trial. Press Ctrl-C to end it. To use custom firmware paths see `python3 -m person_follow.active --help`.

**Feedback law.** Left/right image error controls signed yaw up to 0.35 rad/s. A target right of center requests negative yaw in robotd's trunk frame. While following, forward speed decreases as the box approaches half the image height and is capped at 0.10 m/s. Large lateral errors trigger turning in place. Holding distance suspends forward motion and can continue centering. Actual distance depends on the person and camera pose; this version uses apparent size.

**Motion and handover.** A 20 Hz timer checks inputs independently of the camera process. Motion requires robot state within 250 ms, depth within 350 ms and a camera observation within 500 ms of exposure. The 4×4 central depth cells must all report a valid return at least 400 mm away. Unknown returns can keep the robot stationary. This local clearance check covers the forward sensor corridor; it cannot detect drop-offs or all side obstacles. The head must face forward. Motion expires on target loss or camera silence. With fresh robot state the client sends zero; after loss of the command link or robot telemetry the daemon's configured deadman bounds continued motion.

The upstream twist slot is shared and uses the latest writer. A fresh state reporting another live driver (including a pad) ends the session; restart explicitly to follow again. The client never renews a nonzero command twice from the same state sample. Ownership is observed through telemetry and has a race between a state sample and a command. Use a single active movement producer for the trial; an atomic command-priority guarantee would require firmware support.

### Read the active log

Each row carries the latest model decision, the controller's reason in `control`, the computed velocity in `twist`, and the latest robot telemetry in `reported_move`. A nonzero `twist` is a computed request. Compare it with **later** `reported_move.applied` samples to see what the daemon applied; measuring the physical response requires observing the duck.

| What you see | What to check |
|---|---|
| `warming_up` or `unreliable_target` | Inspect the camera trace for six consecutive fresh, reliable boxes on one person. |
| `expired_observation` | Compare capture age and intervals with the 500 ms freshness limit. |
| `depth_blocked_or_stale` | Check the forward clearance, valid central returns and depth stream freshness. |
| `head_off_axis` | Center the head before resuming the trial. |
| Session exits with `Control released` | Check the active driver, posture and built-in modes. Restart explicitly once the robot is available. |

## A trace from observation to choice

Each JSONL file is one episode. A row carries `version`, increasing `t` in seconds, `age_s`, normalized `target=[x,y,width,height]` or null, and `quality`. Optional `override=true` marks an operator takeover during replay. The current camera recorder does not subscribe to pad state. In the synthetic example the sixth sample completes a window. At t = 1.0 s the supplied Transformer gives approximately 7.28, 4.87 and −21.01 in action order. All three are allowed, so it selects `follow`. At t = 4.2 s the target is missing; the explicit loss rule selects `stop` before evaluating the network.

| Output | Meaning |
|---|---|
| follow | Approach using box height error for forward speed and horizontal offset for yaw. |
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
choice = allowed action with the highest score
```

A stale or unreliable observation requests stop. A normalized box height of at least 0.65 masks follow. These provisional geometry thresholds need field calibration. An empty history or a gap over 0.5 s restarts warm-up. Decision replay and stored checkpoints are versioned independently of the original ball runtime.

## Train and evaluate

```sh
python3 -m person_follow.train --out /tmp/person-follow-run
python3 -m unittest person_follow.test_policy person_follow.test_active
```

The reproducible generator in `person_follow/train.py` labels simulated episodes using latent proximity and trend before adding observation noise. Seed ranges 0–79, 100–119 and 200–219 generate independent train, validation and test episodes. The MLP and Transformer minimize cross-entropy with Adam; validation loss selects the checkpoint. All candidates receive the same test episodes and admissibility rules. See the [machine-readable report](../person_follow/checkpoints/results.json). All-tick accuracy includes warm-up. Eligible accuracy covers complete reliable windows. Count unwanted forward proposals and follow/hold transitions as well: a high agreement score can hide oscillations. Loss-to-stop is measured at the observation timestamps, excluding camera latency and physical braking.

To train on real traces, label the acceptable decision at each reviewed timestamp with a `label` field. Omit the label for an ambiguous interval. Retain the full observation sequence for context. Make a JSON manifest with one entry per episode, using paths relative to that manifest:

```json
{"split":"train","session":"walk-01","person":"p01","site":"room-a","path":"walk-01.jsonl"}
```

Wrap entries in a JSON array and populate all three splits (`train`, `val`, `test`). The loader rejects reuse of a session, person or site across splits. Run `python3 -m person_follow.train --manifest manifest.json --out real-run`. Keep the test episodes for final comparison. In active trials, add external distance measurements, time in the desired band, loss-to-physical-stop latency and operator intervention counts. Record board load and controller arbitration during the first supervised active trials.
