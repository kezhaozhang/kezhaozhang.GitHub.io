---
title: Estimation of Variability From Observed Data: A Bayesian Perspective
---

The uncertainty of estimates from data is a direct result of the posterior distribution of the model for the data generation. This note uses the Bayesian approach to discuss two cases, one of which explains the bootstrap method. 



## Uncertainty of Estimation from Data

Estimates of functions of the data, e.g., the mean, have uncertainty—the uncertainty results from the posterior distribution of the model that generates the data. 

The data model can be expressed as the likelihood probability:


$$
p(x \vert \theta)
$$
where $\theta$ is the model parameters and $x$ the data.



Let $f(x)$ be a function of the data $x$.  Its estimate is the average over the likelihood:
$$
F(\theta) = \int f(x) p(x\vert\theta)dx.
$$


The estimate $F(\theta)$ has a distribution because of the distribution of the model parameter $\theta$, which is the posterior:


$$
p(\theta\vert x) =\frac{p(x\vert\theta)p(\theta)}{p(x)}.
$$


The uncertainty of the estimate $F(\theta)$ results from the distribution of $\theta$Which is inferred from the data.



## Two Examples

The model of the data can be expressed as a mixture model:


$$
x \sim \sum_{i=1}^N \pi_i H(\theta_i),
$$


where $H$ is the basis distribution, $\theta_i$ are the model parameters, and $\pi_i$ are the weights. 



We discuss uncertainty estimates from data in two extreme cases.



### Case 1: A Single Normal Distribution

- Observed Data: $x_1, x_2, \ldots, x_N$
- Model: data are generated from a single normal distribution $x \sim \mathcal{N}(\mu, 1)$ with unknown $\mu$. 

- Prior of model: noninformative flat prior simplicity: $p(\mu)=\rm{constant}$.

- Posterior: 

  
  $$
  \begin{array}{cl}
  p(\mu\vert x) & \propto & \exp\left(-\frac{1}{2}\sum_{i=1}^{N} (x_i-\mu)^2\right)\\
         & = & \exp\left(-\frac{(\mu-\bar{x})^2}{2\frac{1}{N}}\right)\exp\left(-\frac{\overline{x^2}-\bar{x}^2}{2\frac{1}{N}}\right)\\
         & \propto & \exp\left(-\frac{(\mu-\bar{x})^2}{2\frac{1}{N}}\right),
  \end{array}
  $$
  

  where $\bar{x}=\sum_i x_i/N$ and $\overline{x^2}=\sum_i{x_i^2}/N$.

Therefore the posterior distribution of the model parameter $\mu$ is normally distributed, with a standard deviation $\sqrt{1/N}$.


$$
p(\mu\vert x) = \frac{1}{\sqrt{2\pi\frac{1}{N}}}\exp\left(-\frac{(\mu-\bar{x})^2}{2\frac{1}{N}}\right)
$$


The estimate of a function of the data $f(x)$ is 


$$
\begin{array}{cl}F(\mu)&=&\int_{-\infty}^\infty f(x) p(x\vert \mu)  dx\\ & = & \int_{-\infty}^\infty f(x) \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2}\right) dx \\\end{array}
$$


As an example, to estimate the mean of the data,  $f(x) = x$. From Equation (6) and Equation (7), we have


$$
F(\mu) = \mu \sim \mathcal{N}\left(\bar{x}, \sqrt{\frac{1}{N}}\right).
$$


The uncertainty of the estimate of the mean can be obtained from the distribution in Equation (8).

In general, the posteriors can be obtained numerically, e.,g., using MCMC, the distribution of $F(\theta)$ can be obtained numerically from the posteriors. 



### Cased 2: Direc Delta Basis Distribution

- Observed Data: $x_1, x_2, \ldots, x_N$
- Model: data are generated from a mixture of Dirac Delta distributions $x \sim \sum_{i=1}^N \pi_i \delta(x-\mu_i)$  

- Prior of model: noninformative flat prior simplicity: $p(\mu_i)=\rm{constant}$.



This is an extreme case where the basis distribution is the limiting case of a Normal distribution with its standard deviation $\sigma\rightarrow0$.
$$
H = \lim_{\sigma\rightarrow0}\mathcal{N}(\mu, \sigma)=\delta(x-\mu)
$$
To generate all the observed data, it is necessary to have $N$ components in the mixture model and $\mu_i=x_i$. In addition,  because of symmetry among the observed data, the weights $\pi_i$ are equal: $\pi_i = 1/N$.  Therefore the data generation model is


$$
x \sim \frac{1}{N} \sum_{i=1}^{N}  \delta(x- x_i).
$$


This model generates a new data point by selecting randomly from the observed data with equal probability. The randomness in the selection results in the uncertainty of estimates of any function of the data.

To generate $N$ data points from this model, there are $N^N$ Possibilities. Each possibility is equivalent to the sampling of the observed data points, $x_1, x_2, \ldots, x_N$ with replacement. It is impractical to generate all $N^N$ possible combinations and only a subset of combinations are randomly generated in practice. This is precisely the bootstrap method.