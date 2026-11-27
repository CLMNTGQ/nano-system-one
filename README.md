# nano-system-one

![Three MicroDuck robots in a harbor training hangar, with a red ball beside the central robot.](images/nano-system-one-banner.webp)

nano-system-one is a from-scratch experiment in **structured decision output on an edge robot**. Each inference answers a bounded question: given recent observations, which allowed behavior should the MicroDuck propose? The controller consumes a behavior identifier and retains authority to act.

## Current use case · Follow one person

MicroDuck should follow one participant who walks away, hold position within a chosen distance band and stop when the target is lost. The new `person-follow-v1` prototype reads a short history of detected person boxes and proposes **follow**, **hold_distance** or **stop**. The controller owns motion. The active runtime feeds accepted choices to the stock MicroDuck motion service and uses the next camera frame to correct heading and approach speed. Shadow capture and offline replay are also available.

**Run a decision trace** from the repository root with Python 3 and NumPy:

```sh
python3 -m person_follow person_follow/example.jsonl
python3 -m person_follow person_follow/example.jsonl --model person_follow/checkpoints/transformer.json
```

The example and trained weights come from synthetic episodes. On 20 held-out synthetic episodes, agreement on eligible windows was 93.2% for the rule, 95.3% for the MLP and 96.4% for the Transformer. The Transformer issued fewer unwanted forward proposals but switched between follow and hold more often. Physical target tracking, distance calibration and on-board timing await the first robot trial.

**Run active following on the robot** from the repository root with the robot standing, the head facing forward and built-in autonomy, laser and soccer modes off:

```sh
python3 -m person_follow.active
```

This command can move the robot. It uses the included Transformer, the local camera, the forward depth sensor and the existing walking policy. Ctrl-C ends the session. The active loop has passed simulated feedback tests; the first physical trial is pending. [Setup, ownership and active trial procedure](docs/person-follow.md#run-the-active-loop)

[Capture your first MicroDuck person-follow session, inspect the math and reproduce training](docs/person-follow.md) · [Test results](person_follow/checkpoints/results.json)

## How a decision becomes motion

OpenCV detects the person. Six timestamped observations describe their apparent size and horizontal position over time. The Transformer scores three named behaviors. The controller selects an allowed behavior and checks the latest robot state and forward depth before sending a velocity request. The next camera frame shows how the person's position changed.

For example, a person who appears small and slightly to the right can produce `follow`. The controller requests a right turn and forward motion. As the person becomes centered, the turn decreases. As their apparent size increases, forward speed decreases. Losing the target requests `stop`. The trial log records the decision, the requested velocity and the velocity reported by the walking controller.

[Decision contract and worked numerical example](docs/person-follow.md#a-trace-from-observation-to-choice) · [Install and verify on your MicroDuck](docs/person-follow.md#install-and-check-on-the-robot)

## Earlier experiment · Ball tracking

The original ball experiment uses a separate action vocabulary, C runtime and dataset. Its installation commands, design records and results are collected in the [ball-tracking guide](BALL-TRACKING.md).

<a id="run-system-one-on-your-microduck"></a>
[Current MicroDuck setup](docs/person-follow.md#install-and-check-on-the-robot) · [Original ball quick start](BALL-TRACKING.md#run-system-one-on-your-microduck)
