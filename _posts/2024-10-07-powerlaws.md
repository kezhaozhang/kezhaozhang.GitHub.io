---

title: "Power Laws Distribution: First Return Time of a Random Walk "
date: 2024-10-07
typora-root-url: ./..
---



Power law distributions are prevalent in various fields. This note derives the probability of a random walk returning to its starting point for the first time (the first return time), which can be approximated by a power law distribution over sufficiently long time scales. Simulation results closely align with both the exact and approximate probabilities.



### Introduction

In a one-dimensional random walk, a walker has two possible step values, $-1$ and $+1$, each with equal probability. Starting at the origin ($0$), the walker may return to this point after a certain number of steps.  What is the distribution of the time, or steps, for the random walker to return to $0$ for the first time?



Figure 1 illustrates the displacement of a random walk over time. Notably, the random walker returns to the starting point at two even steps: $10$ and $16$. The return at step 10 marks the first return time for this random walk.

<figure>
  <center>
  <img src="/assets/images/power_laws_random_walk_displacement_vs_t.svg" width="500">
   </center>
  <center>
    <figcaption> Figure 1. Displacement of random walk over time. The red dots signify the points where the walker returns to the starting position (0). 
    </figcaption>
  </center>
</figure>



### Calculation

To analyze the random walk, we denote the position at time $t$ as:


$$
y = \sum_{i=1}^t s_t, \notag
$$



where $s_i$ represents the step taken at time $t$. For the walker to return to the starting point at the time $t$, it must satisfy $y=0$, which occurs when $t$ is even (i.e.,  $t=2 n$) and consists of  $n$ steps of $+1$ and $n$ steps of $-1$.

At step $t=2n$, the total number of combinations of $+1$ and $-1$ steps is $2^{2n}$. The specific combinations that result in a return to the origin can be calculated as the number of ways to choose $n$ steps of $+1$ from a total of $2n$ steps:


$$
\left(
\begin{array}{c}
{2 n} \\
n\\
\end{array}\right) = \frac{(2n)!}{n! n!}. \notag
$$



Thus, the probability of returning to the origin at time $t=2n$ is given by:


$$
p_{t=2n} = \frac{(2n)!}{n! n! 2^{2 n}}. \label{eqn:exact}
$$



Using [Sterling's approximation](https://en.wikipedia.org/wiki/Stirling's_approximation) for factorials:


$$
n! \approx \sqrt{2\pi n}\left(\frac{n}{e}\right)^n, \notag
$$



we can derive the approximate probability:


$$
p_{t=2n} \approx \frac{1}{\sqrt{\pi n}}. \label{eqn:approx}
$$


This suggests that the distribution of $t$ behaves like a power law, specifically with an exponent of $\frac{1}{2}$. 



Figure 2 compares the exact probability (Equation $\ref{eqn:exact}$) with the approximation (Equation $\ref{eqn:approx}$) and shows excellent agreement for $n>5$.

<figure>
  <center>
  <img src="/assets/images/power_laws_random_walk_return_time_formula.svg" width="630">
   </center>
  <center>
    <figcaption> Figure 2. Comparison of exact and approximate probabilities of first return time as a function of <i>n</i>.
    </figcaption>
  </center>
</figure>



### Simulations

To validate the theoretical results, we performed simulations of multiple random walks and calculated the probability of first returning to the origin over time. Below is the Wolfram Language code used for the simulation.

```mathematica
SeedRanom[1234];
nsteps = 2010;
nsamples = 1000;
r = RandomChoice[{1, -1}, {nsamples, nsteps}];
rw = FoldList[Plus, 0, #]&/@r;
p = Table[{n, Count[rw[[All, 2 n+1]], 0]/nsamples}, {n, 1, 1000}];
```



Figure 3 illustrates the results of our simulations, showing that they agree well with both the exact and approximate formulas for the probability of first return time.



<figure>
  <center>
  <img src="/assets/images/power_laws_random_walk_return_time_formula_with_simulations.svg" width="630">
   </center>
  <center>
    <figcaption> Figure 3. Comparison of exact and approximate probabilities with simulation results.
    </figcaption>
  </center>
</figure>






### Conclusion

The first return time of a random walk serves as a mechanism to generate the power law distributions. This is demonstrated by approximating the exact probability using Sterling's approximation and numerical simulations, which corroborate the theoretical findings.


