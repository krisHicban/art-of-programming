# The Rise of Neural Networks Over Sequential Rule-Based Algorithms

## 1. Why **sequential, rule-based algorithms** fall short

* Traditional programming assumes you can **explicitly model the world**:
   * If X happens, do Y.
   * If condition A and B, then outcome C.
* But the world is **messy, non-linear, and high-dimensional**:
   * Countless variables interact in unknown ways.
   * Many influences are hidden (you don't see all the causes).
   * Noise, randomness, and context-dependence make exact rules brittle.

Sequential logic can work in **closed systems** (like a calculator or physics simulation), but in **open, real-world systems** (language, vision, social behavior, biology), the variables explode.

## 2. The "everything is connected" problem

* Any single outcome (e.g., a word you'll say next, or the weather tomorrow) depends on **a web of interactions**.
* You can't reduce it to a small fixed chain of rules — because the "causal graph" is too wide, too deep, and often **unknown**.
* If you tried to hand-craft rules for every possibility, the system would collapse under combinatorial explosion.

This is similar to Gödel's and Heisenberg's insights: you can't ever have a **perfect formal system** that fully contains reality.

## 3. So what's the alternative?

Instead of trying to **explicitly capture all rules**, Neural Nets take inspiration from how the **human mind processes information**:

* We don't calculate every factor when recognizing a face.
* We **learn patterns implicitly** from exposure.
* Our brain filters, compresses, and generalizes without ever "knowing all the rules."

Machine Learning adopts the same philosophy:

* **Neural networks don't "understand" all causes**.
* They **approximate functions** that map input → output based on experience (data).
* They absorb complex dependencies automatically, without us writing them down.

## 4. Why **large** neural networks?

* The bigger and more interconnected the network, the more **capacity** it has to capture subtle relationships.
* Reality is **multi-scale**: small local patterns + global structures interact.
* Large models act like giant "function approximators" that can pick up on both.
* Instead of coding causality ourselves, we let the model **emerge representations** through training.

## 5. The core conceptual shift

* **Old view**: If we try hard enough, we can write the rules.
* **ML/NN view**: The rules are too vast, hidden, and entangled. Let's learn them statistically, the way brains do.

It's not about predicting the **entire world**. It's about learning compressed, useful **representations** of reality — just enough to make effective predictions.

## Key Insight

The concept you're asking about is often described as the **curse of complexity and entanglement** in open systems, and the **solution** is to approximate reality via **data-driven learning** rather than explicit sequential rules.

## Formal Terms

Some formal terms tied to this idea:

* **Universal Approximation Theorem** (NNs can approximate any function, given enough capacity).
* **Distributed representation** (knowledge isn't in one rule but spread across weights).
* **Statistical learning** vs. **symbolic reasoning**.