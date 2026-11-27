# Temporal encoder design

**Status.** The forward pass is implemented in [transformer.py](../transformer.py). A trained instance and its native CPU export are described in the [embedded experiment](edge-training.md). Physical deployment awaits validation on robot sessions.

## Context and influences

A missing ball detection can mean a brief occlusion, sustained loss, or a missing camera update. Those cases can require different behaviors even when the latest visibility flag is identical. The proposed encoder gives each behavior access to the recent sequence before the controller makes its decision.

Daniel Kahneman describes System 1 as fast, automatic judgment and System 2 as slower, effortful deliberation in *Thinking, Fast and Slow* (2011). The engineering analogy motivates a bounded judgment at a single decision tick. The caller owns admissibility, deferral, and action dispatch. Deliberative planning is outside the current prototype. [Kahneman, *Thinking, Fast and Slow*](https://us.macmillan.com/books/9780374533557/thinkingfastandslow/)

TypeSafe's Jev inspired the interface: an application declares the question and candidate answers, then consumes a structured result. MicroDuck fixes the question to behavior selection and uses numeric observations and behavior descriptors. Andrej Karpathy's GPT-2 reproduction inspired the implementation style: a compact forward pass whose tensor operations can be inspected directly. [TypeSafe AI, *Introducing System One Models & Jev*](https://typesafe.ai/blog/introducing-system-one-models-and-jev) · [Karpathy, build-nanogpt](https://github.com/karpathy/build-nanogpt)

## Decisions and trade-offs

| Decision | Reason | Consequence to evaluate |
|---|---|---|
| Encode visibility and frame validity separately | A valid frame without a ball and a missing frame provide different evidence | Training and replay must include both cases |
| Add an age embedding to each observation | Detection order distinguishes loss from reacquisition | Age in ticks assumes regular sampling. Measure timing jitter before using this representation on hardware |
| Use one bidirectional encoder block | Every observation in the window is available at decision time. Each behavior can retrieve evidence using its own query | Attention cost grows quadratically with sequence length in this dense implementation. Compare quality and latency with the MLP and timeout rule |
| Isolate candidates with an attention mask and share the readout | Candidate identity comes from its descriptor, so reordering admissible behaviors should only reorder logits | Two behaviors with identical descriptors receive the same score. Extend descriptors when introducing behaviors that need different treatment |
| Start with width 16 and two heads | This supplies two attention distributions per candidate at a size convenient for a reference implementation | Replay and target-device measurements must justify the final dimensions |

The encoder uses scaled dot-product attention, residual connections, normalization, and a position-wise feed-forward network. [Vaswani et al., *Attention Is All You Need*](https://arxiv.org/abs/1706.03762)

## Inspectable forward pass

For four observations and three candidate behaviors, the input sequence has seven rows of width 16. Observation rows may read the whole history. Each behavior row may read the history and itself. Each head therefore produces a 7 by 7 attention matrix, with prohibited entries set to zero after softmax. The two heads return eight features each. The shared readout projects the three final behavior vectors onto three logits. A softmax across admissible behaviors gives their relative probabilities.

This structure makes two properties checkable before training. Permuting behavior rows permutes logits. Removing one behavior preserves the surviving logits, although normalization changes their probabilities. Attention weights expose how the implementation mixes values. Establishing which observations influence a learned choice requires interventions such as changing frame validity or reversing the order of detections and measuring the change in logits.

## Evidence needed to retain the Transformer

Use held-out, complete episodes with matched windows that end in the same frame but have different histories. Give each baseline access to the same sensor evidence and document its derived features. Compare transition errors, unnecessary switches, and delay to a search decision against the MLP and timeout rule. Measure latency and memory on the RK3566 with other services active. If the simpler scorer achieves equivalent behavior within the deployment budget, retain it for the robot. The [runtime evaluation plan](architecture.md#from-snapshots-to-episodes) defines the path from replay to closed-loop trials.
