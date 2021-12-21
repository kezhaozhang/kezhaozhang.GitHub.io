---
title: Solution to An Example in Bernoulli's Fallacy
---

# Problem Statement

Your friend rolls a six-sided die and secretly records the outcome; this
number becomes the target $T$. You then put on a blindfold and roll the
same six-sided die over and over. You're unable to see how it lands so,
each time, your friend (under the watchful eye of a judge, to prevent
any cheating) tells you *only* whether the number you just rolled was
greater than, equal to, or less than $T$. After some number of rolls,
say 10, you must guess what the target was. What would your strategy for
guessing, and how confident would you be?

Suppose the sequences of the outcomes is

::: center
G, G, L, E, L, L, L, E, G, L
:::

with G representing a greater roll, L a lesser roll, and E an equal
roll.

p.36 of Bernoulli's Fallacy.

# Solution

Because each roll in the sequence is independent, the order of the
outcomes is unimportant. The number of each type of rolls are important
and in the sequence of 10 outcomes, the number of greater rolls, equal
rolls and less rolls are

$$\begin{aligned}
N_G &= 3, \\
N_E & = 2, \\
N_L & = 5.\end{aligned}$$

## Maximum Likelihood

Maximum likelihood finds the best target value $T$ that maximizes the
likelihood:
$$p(\mathrm{sequence}|T) = \left(\frac{6-T}{6}\right)^{N_G} \left(\frac{1}{6}\right)^{N_E} \left(\frac{T-1}{6}\right)^{N_L},
\label{eqn_likelihood}$$ where $\frac{T-1}{6}$, $\frac{1}{6}$, and
$\frac{T-1}{6}$ are the probability to get a greater roll, an equal roll
and a less roll, respectively.

We can calculate the value of
Equation ([\[eqn_likelihood\]](#eqn_likelihood){reference-type="ref"
reference="eqn_likelihood"}) for each of possible values of $T$, 1, 2,
3, 4, 5, and 6 in this case, and find the $T$ corresponding to the
largest value.

::: center
   $T$        Likelihood
  ----- ----------------------
    1             0
    2     $64\times 6^{-10}$
    3    $864\times 6^{-10}$
    3    $1944\times 6^{-10}$
    5    $1024\times 6^{-10}$
    6             0
:::

From the table we can see that the likelihood is the largest when $T=4$.
Therefore, $T=4$ is the best guess. However, maximum likelihood does not
provide confidence or variation estimate of $T$. We will use Bayes'
theorem for that.

## Bayes' Theorem

Using Bayes' theorem

## Probabilistic Programming using PyMC3

``` {.python language="Python"}
import numpy as np
import pymc3 as pm

nL = 5
nE = 2
nG = 3
with pm.Model() as model:
    #Discrete uniform prior
    T = pm.DiscreteUniform('T', 1, 6) 
    #log-likelihood
    logp = nL*np.log((T-1)/6) + nE*np.log(1/6) + nG*np.log((6-T)/6) 
    potential = pm.Potential('potential', logp)
    
with model:
    trace = pm.sample(1000, tune=1000)
```
