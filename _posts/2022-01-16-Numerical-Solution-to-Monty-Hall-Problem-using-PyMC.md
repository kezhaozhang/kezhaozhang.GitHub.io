---
title: "Numerical Solution to Monty Hall Problem using PyMC"
typora-root-url: ../../kezhaozhang.GitHub.io
---

We numerically solved the [Monty Hall Problem](https://en.wikipedia.org/wiki/Monty_Hall_problem) with PyMC3, a probabilistic programming package in Python. The PyMC code is adapted from [Austin Rochford](https://austinrochford.com)'s [Introduction to Probabilistic Programming with PyMC](Intro to Probabilistic Programming with PyMC).



Since the Python array is 0 based, we change the Monty Hall Problem to 0 based: The game player picked Door 0, the game host opened Door 2, and the game player is asked to decide whether to switch to Door 1. 



## Variables

There are two variables in this problem:

- The observed: $D$
  The door that the game host opened. Possible values include 0, 1, and 2

- Parameter to be inferred: $L$

  The prize location, or the door behind which the prize is located.  Possible values include 0, 1, 2 

The inference is to find the posterior probability $P(L\mid D)$, which is the probability of the prize being located behind Door $L$, given that the game host opened Door $D$. In the calculation, we assume that the game player picked Door 0. 



## Prior and Likelihood

The prior for $L$ is uninformative: $\frac{1}{3}$ for each of the three possible values.

The likelihood $P(D\mid L)$ depends on $L$ and $D$. For each $L$, the probability for the host to open the door $D$ is listed in the table below.

|       | $D=0$ |     $D=1$     |     $D=2$     |
| :---: | :---: | :-----------: | :-----------: |
| $L=0$ |   0   | $\frac{1}{2}$ | $\frac{1}{2}$ |
| $L=1$ |   0   |       0       |       1       |
| $L=2$ |   0   |       1       |       0       |



##  PyMC Code and Numerical Result

The parameter $L$ is a `Categorical` random variable with three possible values. The uninformative prior specifies that the probability is $\frac{1}{3}$ for each value.



The observed variable, $D$ is also a `Categorical`random variable. The pmf of its distribution is a function of $L$, as listed in the table in the previous section. The dependence on $L$ is coded using `pm.math.switch` function. 



```python
with pm.Model() as monty:
    L = pm.Categorical('L', p=np.array([1/3, 1/3, 1/3])) #Prior
    p = pm.math.switch(tt.eq(L,0), np.array([0, 0.5, 0.5]), 
                       pm.math.switch(tt.eq(L,1), np.array([0, 0, 1]), 
                                      np.array([0, 1, 0])) ) #Likelihood
    D = pm.Categorical('D', p=p, observed=2) #Observed data: Door 2 is opened
    trace = pm.sample(10_000, return_inferencedata=True)
```



The MCMC result agrees with the theoretical calculation. Switching the door (from  $L=0$ to $L=1$ if Door 2 is opened ($D=2$) or to $L=2$ if Door 1 is opened ($D=1$)) will increase the probability of getting the prize form $\frac{1}{3}$ to $\frac{2}{3}$.

<figure>
  <center>
  <img src="/assets/images/D=2.svg" width="600">
  </center>
  <center>
  <captionfigure>
    MCMC result agrees with the theoretical result (D=2). Switching the door from L=0 to L=2 increases the prize probability from 1/3 to 2/3.
  </captionfigure>
  </center>
</figure>

<figure>
  <center>
  <img src="/assets/images/D=1.svg" width="600">
  </center>
  <center>
  <captionfigure>The result for a different opened door (D=1), the same result is obtained as in the case with D=2.
  </captionfigure>
  </center>
</figure>



## Discussion

The key to solving the problem with PyMC is to identify the observed ($D$) and the model parameters ($L$) and the likelihood.

In addition, the distribution of $L$ is important. In Austin Rochford's implementation, the distribution for $L$ is `DiscreteUniform`. Even though a similar result is obtained as the `Categorical` distribution used in this note, there are two advantages of using `Categorical` distribution:

1. More robust MCMC sampling. The sampling with `DiscreteUniform` distribution leads to less effective samples (e.g., `The number of effective samples is smaller than 25% for some parameters`). In addition, when changing the opened Door from 1 to 2, the initial sampling value must be set to be different from 2; otherwise, the sampling is aborted. This likely is because the distribution of $L$ is not truly `DiscteteUniform`.
2. With `Categorical` distribution, the prior of $L$ can be non-uniform. It might be possible that the prize is more likely to be behind one door than others. This can be encoded with `Categorical` distribution but not with `DiscreteUniform`.