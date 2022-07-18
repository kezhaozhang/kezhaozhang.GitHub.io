---
title: "Bias in Poetntial Outcomes in Causal Inference"
typora-root-url: ../../kezhaozhang.GitHub.io
---





This note summarizes my understanding of the bias in potential outcomes while reading the book *Causal Inference: The Mixtape* (https://mixtape.scunning.com).



In a binary treatment case, the population is divided into two groups: with treatment ($D=1$) and without treatment ($D=0$). The goal is to estimate the causal effect using the outcomes of these two groups. The difference between this estimate and the "true" causal effect is also estimated.



First, some notations: $D=1$ and $D=0$ denote the two sub-populations.  The proportions in $D=1$ and $D=0$ groups are $\pi$ and $1-\pi$, respectively. The  outcome $Y^1$ denotes the result if the population has gone through the treatment and outcome $Y^0$ if the population does not have the treatment. An illustration of the subpopulations is shown in Figure 1.



<figure>
  <center>
  <img src="/assets/images/mixtape_ate_old.svg" width="300">
   </center>
  <center>
  <figurecaption>
  Figure 1. Population diagram. The black blocks are the actual measurements, and the gray blocks are the counterfactuals.
  </figurecaption>
  </center>
</figure>



The average result of the sub-populations are


$$
\begin{align*}
a &= E[Y^1\mid D=1]\\
b & = E[Y^1\mid D=0] \\
c & = E[Y^0\mid D=1] \\
d & = E[ Y^0 \mid D=0],
\end{align*}
$$
where $b$ and $c$ are counterfactuals because they are not observed.



In an ideal case, the whole population goes through two treatment cases: with and without treatment. The corresponding outcomes are $Y^1$ and $Y^0$. The average treatment effect (ATE) is the average of $Y^1 - Y^0$, or


$$
\begin{align*}
\mathrm{ATE} 
&= \pi a + (1-\pi)b  - \left[ \pi c +(1-\pi) d \right]\\
& = (a-d) - (1-\pi) (a-b) -\pi (c-d),
\end{align*}
$$


where $a-d = E[Y^1\mid D=1]-E[Y^0\mid D=0]$ is the observed result. The $a-b$ and $c-d$ terms are the selection bias terms.  For the observed estimate $a-d$ to be equal to the $\mathrm{ATE}$ if there is no selection bias, i.e.,
$$
\begin{align*}
a-b &= E[Y^1\mid D=1] - E[Y^1\mid D=0] = 0,  \\
c-d &= E[Y^0\mid D=1] -E[Y^0\mid D=0] = 0.
\end{align*}
$$
Therefore, no selection bias requires the average outcomes of the two subpopulations to be the same for both with treatment and without treatment. This helps to understand that randomized experiments reduce or remove selection bias.

