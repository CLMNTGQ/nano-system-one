# From detector evidence to a typed choice

MicroDuck sees its ball four times, then receives two valid misses. This record follows that window through the deployed encoder and shows why the same scores can produce three different controller outcomes.

## Build the nine input rows

The history `(2,2,2,2,1,1)` runs oldest to newest. Features encode (visible, valid): a sighting is `(1,1)`, a valid miss `(0,1)`, and a missing frame `(0,0)`. A valid miss is evidence about the scene; a missing frame describes the sensor stream. For observation `x[t]` and behavior descriptor `c[i]`:

```text
u[t] = x[t] @ W_obs + age[5 − t]
v[i] = c[i] @ W_opt
```

Here `@` denotes matrix multiplication. Both projections map two features to width 16. The learned age vector `age[0]` identifies the newest observation. Track, search and wait use descriptors `(1,0),(0,1),(0,0)`. Six observations and three candidates form `X` of shape `9 × 16`. Age applies only to observations. Candidate identity comes from its descriptor. [Implementation](../transformer.py)

## Retrieve evidence for each behavior

Each of two heads projects normalized rows into queries, keys and values of width eight:

```text
Q, K, V = split(LN(X) @ W_head)
scores  = (Q @ transpose(K)) / sqrt(8)
A       = row_softmax(scores + M)
H       = A @ V
```

`LN` denotes layer normalization. Each head computes its own `Q`, `K`, `V` and `H`. Layer normalization centers and rescales each row across its 16 features. A candidate query describes the evidence it seeks. Keys determine match strength; values supply the information retrieved. Dividing by the square root of head width controls the scale of the dot products. The attention mask `M` adds zero on readable connections and negative infinity elsewhere. Every row may read the six observations; each candidate may also read itself. The tracking row of each `9 × 9` attention matrix therefore distributes unit mass over seven positions. Those weights explain mixing. Testing the influence of a particular missed sighting requires changing that input and repeating the forward pass.

Concatenate the two heads, mix them and apply a feed-forward network independently to each row:

```text
Y = X + concat(H1, H2) @ W_out
U = Y + ReLU(LN(Y) @ W1) @ W2
```

The feed-forward dimensions are `16 → 64 → 16`. The residual additions preserve a path for the incoming representation. The same learned vector `w` scores each final candidate row: `z[i] = LN(U[i]) @ w`. One forward pass produces all three logits. Candidate isolation and the shared readout preserve each candidate's raw score when the available set changes.

## Where System One gets its output type

The application declares a vocabulary and binds its identifiers to candidate descriptors. The controller supplies an admissible set `S`. This second mask acts **after** the Transformer. For a usable observation and nonempty `S`, the C selector returns the identifier with the largest allowed score. Equal scores use enum order.

```text
selected = candidate in S with the largest z[i]
weight[i] = exp(z[i])                    (i in S)
p[i] = weight[i] / sum(weight[j] for j in S)
```

Disallowed candidates receive zero probability. The C selector needs only the argmax. During training, when the annotation accepts a set `G` of behaviors, the loss is `loss = −log(sum(p[i] for i in G))`. A single acceptable label gives ordinary cross-entropy. Multiple labels allow a transition window when waiting and searching both make sense. The normalized probability is conditional on the options; measuring field correctness and calibration requires representative episodes. [Training objective](../train.py) · [Native selector](../edge/scorer.c)

## One window, three outcomes

The base-three index of the six symbols is **724**. Its exported raw logits are **−6.490017, −6.331248, 0.738609** in track, search, wait order. The controller changes the outcome without changing those scores:

| Mask | Allowed | Result | What happened |
|---|---|---|---|
| 7 | track, search, wait | choice / wait | Wait receives approximately 0.99843 of the allowed mass. |
| 3 | track, search | choice / search | The controller excludes wait. Search receives approximately 0.53961 of the remaining mass. |
| 0 | none | defer / empty_mask | The caller receives no proposed action or logits. |

The middle case produces this decision object (logits rounded for display):

~~~json
{
  "model": "choice-v1/726023a92444dc6267ebe816188e998e70cac459",
  "allowed": ["track", "search"],
  "status": "choice",
  "selected": "search",
  "logits": {"track": -6.490017, "search": -6.331248, "wait": 0.738609}
}
~~~

The three named scores let the reader see that the top unconstrained candidate was excluded. On deferral the object carries model, allowed options, status and reason. Waiting is a behavior; deferral returns control to the caller. The outer live trace separately records the incumbent controller phase and head/walk flags. These are observations of controller activity. The stdin adapter can record the executed behavior supplied by its caller.

## Why the trained model fits on the robot

There are `3⁶ = 729` discrete histories. Exporting three float32 logits per history requires `729 × 3 × 4 = 8,748` bytes. Training learns the scores; the exporter enumerates the inputs; the robot looks up a row and selects an allowed behavior. A continuous sensor input or a change of vocabulary would change that trade-off.

When investigating a proposal, check the observation history, model identifier, allowed set and score ordering, then inspect the observed incumbent action and subsequent measurements. Version 2 replay recomputes the full decision object. Older traces support comparison of the original proposal and timing fields. [Replay](../edge/robot_trace.py) · [Deployment and limits](edge-training.md)
