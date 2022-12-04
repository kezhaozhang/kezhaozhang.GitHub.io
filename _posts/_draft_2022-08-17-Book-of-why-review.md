---
title: "Review of The Book of Why"
---

**The Book of Why:  The New Science of Cause and Effect**

**Judea Pearl and Dana Mackenzie**

**Basic Books (2018)**



This is an excellent introductory book on causal inference. It presents a unified, systematic, and axiomatic approach. 



First, the book lists the three rungs of the causal inference:

| Rung |                |           |      |
| ---- | -------------- | --------- | ---- |
| 1st  | Data           | Seeing    |      |
| 2nd  |                | Doing     |      |
| 3rd  | Counterfactual | Imagining |      |

three  rungs

1. Data: Seeing:  Correlation in data
2. Doing: 
3. Counterfactual

The first rung only reveals the correlation patterns in the data. Even the powerful deep learning is in this category: "animal abilities, not human intelligence."

One key idea the book emphasizes repeatedly is that data alone is insufficient in causal inference. Instead, one must have a model of the variables based on domain knowledge. In other words, causal inference is a scientific investigation, not a mere data reduction.

Behind the causal diagram is probability. However, the authors conjecture that the causal structure is more fundamental than probability and  "that human intuition is organized around causal, not statistical, relations."  

Causation cannot be reduced to probability. The definition of causation based on probability is therefore not complete. For example, $P(Y\mid X) >P(Y)$ is only an observation that if we see $X$, then the probability of $Y$ increases. This could be caused by some other parameter which is the cause of both $X$ and $Y$. 



The model of the causal relationships is best represented by directed acyclic graphs (DAGs), whose nodes are the variables, and the arrows point from the causes to the effects. The book gives a vivid history of using causal inference with graphs. It was XXX  who first used the graph model to estimate the gene effect. It also explains the difference between the causal graph and Bayesian networks in that the connections in the former are directional but directionless in the latter.



The causal diagram is used to encode our knowledge of the system.

Using causal diagrams to eliminate the confounding effect, the emphasis is shifted from confounders to deconfunders. The two sets may overlap, but they don't have to.  If we have data on a sufficient set of deconfounders, it does not matter if we ignore some or even all confounders. This is possible through the back-door criterion, which identifies which variables in a causal diagram are deconfounders.



Deconfounding: three elements: causal diagram and sufficient deconfounders, backdoor criterion



Removing of confounders, randomly controlled trials.

With the causal model in place, the book describes the logic and algorithms for causal inference:

- The backdoor criterion. This method removes the confounding effects by inspecting the graph structure and conditioning variables to shut off the paths that lead to confounding. All the connections can be classified into three types: fork, collider, and pipe. To shut off a path with an arrow pointing to the causal variable, one can condition on the fork but not the collider. Using some examples, the book convincingly demonstrates the simplicity and logic of this systematic approach compared to the ad hoc methods in previous practice. One application is the understanding of randomized controlled tests (RCT).

- Frontdoor criteria
- do calculus: transform do calculus to conditional probability. Conditions
- The direct and indirect effect
- Counterfactual



In this sense, all of our models are approximations of the truth; the Bayesian approach is the most logical tool that helps us refine our understanding and makes our models closer to reality.

In summary, *Bernoulli's Fallacy* is well researched and easy to understand. The arguments for the Bayesian approach are convincing. It's a beneficial book before reading Jaynes' *Probability Theory*.