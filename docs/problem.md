# Choosing a behavior after losing sight of a ball

MicroDuck is tracking a ball when a detection disappears. A single missed frame and several seconds of valid frames without a detection both show an absent target. The appropriate response depends on the history. Missing camera updates introduce a third case: the robot has no current evidence about visibility.

## Decision contract

**Question.** Which admissible behavior fits the latest six observations? The application defines the vocabulary. TypeSafe's Jev inspired this bounded interface. The first implemented primitive is a categorical choice for ball tracking. [TypeSafe AI, Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)

| Model or controller concept | Trace label | Native result | Mask bit |
|---|---|---|---|
| Follow the target (`track_target`) | `track` | `NSO_TRACK` = 0 | 1 |
| Look for the target (`search_target`) | `search` | `NSO_SEARCH` = 1 | 2 |
| Hold the current activity as specified by the wait behavior | `wait` | `NSO_WAIT` = 2 | 4 |
| Return control without proposing a behavior | `defer` | `NSO_DEFER` = −1 | — |

On success, `nso_select` returns an identifier enabled by the mask and optionally writes all three raw logits in enum order, including those of masked candidates. A caller that needs probabilities can apply softmax over just the allowed candidates. Those probabilities require field calibration before they can support a confidence threshold. Equal logits select the lowest allowed enum index.

On deferral the optional logit buffer stays untouched; callers must branch on the result before reading it. Causes include missing current observation, invalid input, empty mask, or a caller reporting stale state or loss of ownership. The native result groups those causes under one value. The controller rechecks admissibility at dispatch and records actual behavior separately from the proposal. [C interface](../edge/scorer.h) · [Implementation](../edge/scorer.c)

| Example situation | Proposed behavior | Meaning |
|---|---|---|
| Fresh detection, tracking permitted | track_target | Continue following the detected ball |
| Brief occlusion | wait | Allow the detector to reacquire the target |
| Sustained loss, fresh observations, search permitted | search_target | Start the search routine |
| Stale sensor stream or operator takeover | Controller handles the event | Discard pending autonomous decisions |

## Proposed solution

Start with an offline behavior scorer and compare it with the existing autonomous controller on identical recorded observations. Use sensor timestamps and a short history summary as features. Assign training labels from episode reviews that document the intended behavior and acceptable transition timing. Keep all samples from each physical run together when splitting data.

The first experiment asks whether the scorer can reduce unnecessary transitions while preserving timely reacquisition. A later closed-loop trial is needed to establish whether its choices improve actual tracking: replay records only the trajectory taken by the original controller.
