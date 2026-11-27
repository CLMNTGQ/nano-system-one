# Lesson 1 · From a missed detection to three logits

A duck sees its ball twice, then misses it twice. Should it keep tracking, search, or wait? Before training anything, build the computation that could learn to answer. You should be able to point to the input that records a miss, the operation that shares evidence across time, and the scalar that scores waiting.

## Why call it System One?

In *Thinking, Fast and Slow* (2011), Daniel Kahneman uses System 1 for fast, automatic judgments and System 2 for slower, effortful deliberation. We borrow that distinction as an engineering analogy. A single forward pass fits the first role: estimate which available behavior suits the current evidence. A slower process could collect more observations or plan several steps. This lesson implements only the fast judgment. The controller owns the final action, including the option to defer. [Daniel Kahneman, *Thinking, Fast and Slow*](https://us.macmillan.com/books/9780374533557/thinkingfastandslow/)

TypeSafe's Jev inspires the interface: the application supplies an observation and declares the possible answers, and the model returns scores that software can consume. Our first task fixes the question to “Which behavior fits now?” and gives each behavior a short numeric descriptor. The encoder below is our own design. [TypeSafe AI, *Introducing System One Models & Jev*](https://typesafe.ai/blog/introducing-system-one-models-and-jev)

Andrej Karpathy's *Let's reproduce GPT-2 (124M)* and its companion repository motivate the teaching method: start with an executable example, make the tensor shapes visible, then add one mechanism at a time. GPT-2 predicts the next token. Here all observations through the decision tick are already available and the output vocabulary is the supplied set of options. That changes both the attention mask and the readout. [Karpathy, build-nanogpt](https://github.com/karpathy/build-nanogpt) · [Video lecture](https://www.youtube.com/watch?v=l8pRSuU81PU)

## 1. Give time a representation

Run `python transformer.py` with NumPy. Each observation is a pair of numbers, **visible** and **frame valid**, sampled on a regular clock. A valid frame with a missing ball is [0, 1]. A missing frame is [0, 0]. Each row becomes a 16-dimensional vector through a shared projection. Add an embedding for its age in ticks. This gives the model the information needed to distinguish losing a ball from reacquiring it, even when the two windows contain the same rows in reverse order.

The existing 96-parameter MLP is a good baseline when elapsed time since detection already summarizes the observation history. We introduce a Transformer to investigate whether a behavior can learn which observations matter in a window with intermittent misses or missing frames. A measured gain over a baseline given the same information would justify the added computation. The tiny settings serve the lesson; hardware sizing comes after replay measurements.

## 2. Let each candidate ask for evidence

Project the two features of each behavior into the same width using a separate matrix. Stack four observation rows and three candidate rows to form X of shape (7, 16). In each of two heads, the query of a row is multiplied by the keys of the rows it may read. Dividing by the square root of head width (8 here) controls the scale. Row-wise softmax turns the resulting affinities into weights. Their weighted sum of the value vectors is the retrieved evidence. [Vaswani et al., *Attention Is All You Need* (2017)](https://arxiv.org/abs/1706.03762)

| Query row | May read | Reason |
|---|---|---|
| Observation | Every observation in the window | The complete window precedes this decision |
| Candidate | Every observation and itself | Each behavior gathers evidence using its own descriptor |

Inspect the printed attention row for the tracking candidate. The last two entries are zero because those columns belong to searching and waiting. Shuffle the candidates: the corresponding logits should follow them. No candidate has a rank embedding, all share a readout, and each is isolated from the others. The resulting probabilities still depend on the set of admissible options through their common denominator.

## 3. Read out a decision

After attention, mix the two heads, add the residual, and run a two-layer feed-forward network independently on each row with another residual. Normalization before each sublayer controls the scale it receives. One shared vector projects each final candidate row to a scalar. The shapes along this path are (2, 7, 8) for per-head context, (7, 16) after mixing, (3, 16) for the selected rows, and (3,) for the logits. Softmax across these three logits produces the distribution printed by the example.

The initial scores come from seeded random weights. They demonstrate information flow. The next learning milestone is to minimize the negative log probability of the labeled admissible behavior, then compare against the MLP and a timeout rule on held-out episodes. Use matched windows with the same last frame and different histories. Give the baselines access to the same sensor evidence. Evaluate whole episodes before interpreting a probability as calibrated or choosing a deployment threshold.
