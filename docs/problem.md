# Choosing a behavior after losing sight of a ball

MicroDuck is tracking a ball when a detection disappears. A single missed frame and several seconds of valid frames without a detection both show an absent target. The appropriate response depends on the history. Missing camera updates introduce a third case: the robot has no current evidence about visibility.

## Decision contract

At each decision tick, the caller supplies an observation and a nonempty set of admissible behaviors. The model returns one logit per candidate. The caller can normalize the scores and select a behavior, defer to the existing controller, or drop an expired result. Candidate identifiers travel with their descriptors so that scores map back to behaviors.

| Example situation | Proposed behavior | Meaning |
|---|---|---|
| Fresh detection, tracking permitted | track_target | Continue following the detected ball |
| Brief occlusion | wait | Allow the detector to reacquire the target |
| Sustained loss, fresh observations, search permitted | search_target | Start the search routine |
| Stale sensor stream or operator takeover | Controller handles the event | Discard pending autonomous decisions |

## Proposed solution

Start with an offline behavior scorer and compare it with the existing autonomous controller on identical recorded observations. Use sensor timestamps and a short history summary as features. Assign training labels from episode reviews that document the intended behavior and acceptable transition timing. Keep all samples from each physical run together when splitting data.

The first experiment asks whether the scorer can reduce unnecessary transitions while preserving timely reacquisition. A later closed-loop trial is needed to establish whether its choices improve actual tracking: replay records only the trajectory taken by the original controller.
