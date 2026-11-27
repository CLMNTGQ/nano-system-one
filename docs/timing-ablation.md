# Temporal inputs and the orientation block

## Decision and evidence

The production encoder remains the shadow reference. On five new authored episodes (29 labeled ticks, including explicit deferrals), an eight-wide one-head candidate-to-history attention model (A) accepts 25 decisions; adding receipt-time features accepts 25. Prepending a trainable history self-attention block (B) accepts 23 with or without the same timing head. The existing production Transformer accepts 23 with or without a timing head. The timed summary MLP and timed rule each accept 24. [Per-tick results](../reports/timing-ablation.json)

These are simulated event times and author-defined expectations. The equal aggregate scores do not establish general equivalence. Two matched loss histories intentionally use different arrival offsets and transition targets. Revisit the plausible transition interval with a reviewer before using the cases for deployment selection. Keep whole physical sessions blind. No change is promoted from these scores.

## Experiment

Run `node experiments/timing-ablation.mjs build/timing`. There are two references: the deployed A table, and a smaller trainable A control matched to B in width, readout and training procedure. Both experimental models start from matching seeded parameters and train against acceptable-set cross-entropy on the original synthetic training sessions. Both project an observation symbol plus its age embedding; a candidate query reads the resulting history. B alone first contextualizes observations with self-attention and a residual. These small matched models use one head and tanh residuals; they are a design probe rather than weight-identical copies of the production two-head block. Seed 17 and optimization settings are fixed. Finite-seed and small-fixture uncertainty remain.

The timing variants freeze the discrete forward pass and fit an additive linear logit correction on four separate authored development sessions. Inputs are a bias, elapsed seconds since the last valid receipt and last positive receipt (capped at two seconds), and never-observed indicators. All models defer on missing current looks. The timed MLP uses the same receipts and frame evidence. The fixed rule uses a 650 ms search threshold and two-of-three confirmation. Training fixtures without sub-tick timing use nominal receive ages of zero. Code and data are in English.

The runtime can still carry three float32 logits for each of 729 histories. The experimental B [table](../experiments/exports/B-scores.inc) has 2,187 entries; its binary export is 8,748 bytes. The timing correction requires additional scalar coefficients and receipt-time state. The exporter produces both B.f32le and a JSON report. Continuous time values are not enumerated into the table.

## Robot acceptance gate

Follow [the live socket commands and paired trial protocol](ooda-field-validation.md). Both stored ages measure local receipt, with null for never seen, and reset across gaps and ownership changes. Replay checks ages as well as proposals on new traces and remains compatible with earlier traces. The socket fixture exercises stable tracking, five ticks of valid misses, five ticks of detector silence, return, pad takeover, and warm-up on resume.

For physical evaluation, do a worker-disabled and a shadow run with matched detector settings and durations. Retain raw traces, p95/p99 receiver-to-proposal latency, missed-loop deltas and firmware identifier. Have a reviewer mark accepted actions and transition intervals on complete sessions without seeing proposals. Select on premature searches and switch delay first, then evaluate the chosen policy in paired closed-loop trials with a separately integrated dispatch adapter. No host replay or simulated timestamp substitutes for these measurements.

To compare every variant, the rule, summary MLP and reported incumbent phase on the same new physical trace, run `node experiments/timing-ablation.mjs build/timing session.jsonl annotations.json`. The optional annotation JSON maps session IDs to tick numbers to arrays of acceptable actions. The runner checks every recorded production proposal before writing field-replay.jsonl; unannotated ticks retain a null label.
