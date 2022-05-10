---
Title: Uncertainty Estimation from Posterior Distribution 
---



## Uncertainty of Estimation from Data

For a set of observed data, the estimate of some functions of the data, for example, the mean, has uncertainty. The reason for the uncertainty can be understood from the Bayesian point of view.



In the estimate, one has a model of the data which is expressed as the likelihood function
$$
p(x \vert \theta)
$$
The function of the data $f(x)$ is estimated from the posterior distribution of the model parameters $\theta$:
$$
G(\theta) = \int f(x) p(x\vert\theta)dx.
$$


The estimate $G(\theta)$ has a distribution because the model parameters $\theta$ have distribution.



The distribution of $\theta$ is the posterior:
$$
p(\theta\vert x) =\frac{p(x\vert\theta)p(\theta)}{p(x)}.
$$

## An Example

- Model: data from the normal distribution $x \sim N(\mu, 1)$ with unknown $\mu$. 

- Observation: $x_1, x_2, \ldots, x_N$

- Prior: noninformative flat prior simplicity: $p(\theta)=\rm{constant}$.

- Posterior: 
  $$
  \begin{array}{cl}
  p(\mu\vert x) & \propto & \exp\left(-\frac{1}{2}\sum_{i=1}^{N} (x_i-\mu)^2\right)\\
         & = & \exp\left(-\frac{(\mu-\bar{x})^2}{2\frac{1}{N}}\right)\exp\left(-\frac{\bar{x^2}-\bar{x}^2}{2\frac{1}{N}}\right)\\
         & \propto & \exp\left(-\frac{(\mu-\bar{x})^2}{2\frac{1}{N}}\right),
  \end{array}
  $$
  where $\bar{x}=\sum_i x_i/N$ And $\bar{x^2}=\sum_i{x_i^2}/N$.

Therefore the posterior distribution of the model parameter $\mu$ is normally distributed, with a standard deviation $\sqrt{1/N}$.
$$
p(\mu\vert x) = \frac{1}{\sqrt{2\pi\frac{1}{N}}}\exp\left(-\frac{(\mu-\bar{x})^2}{2\frac{1}{N}}\right)
$$


Let the function of data be $f(x) = x^4$.
$$
p(x\vert\mu) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2}\right)
$$
The expected value of the function $f(x)$ is a now a function of $\mu$:
$$
\begin{array}{cl}G(\mu)&=&\int_{-\infty}^\infty f(x) p(x\vert \mu)  dx\\ & = & \int_{-\infty}^\infty x^4 \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2}\right) dx \\ & = & \mu^4 +6 \mu^2 + 3\end{array}
$$
Because the distribution of $\mu$ is the posterior $p(\mu\vert x)$, $G(\mu)$ has its own distribution from which the uncertainty of estimate can be obtained. In general, the posteriors are obtained numerically, e.,g., using MCMC, the distribution of $G(\theta)$ can be obtained numerically from the posteriors. 



Uncertainty estimate: mixture model, Dirichlet process, and bootstrap as a special case



*Dirichlet process for a fixed K, the distribution is more uniform as $\alpha$ Increases. Use a simulation examples, for example, K=30, $\alpha=0.5, 1, 10, 100, 1000, 1000$ and generate N=30, 50, 100 samples.*

Uncertainty estimation is data model dependent. For a set of observed data, multiple reasonable models can be used.

For example, the data can be modeled by a mixture of basis distributions
$$
x \sim \sum_{i=1}^N \pi_i H(\theta_i).
$$
For example, a Gaussian mixture model, where $H$ is the Gaussian distribution. 

One extreme case is that there is only one component, i.e., the data are modeled by a single distribution. The uncertainty of the function of the data can be inferred from the uncertainty of the posterior distributions of parameters $\theta$. In this case, the uncertainty 



Another extreme case is that the basis distribution is the limiting case of a Gaussian distribution with its standard deviation $\sigma\rightarrow0$.
$$
H = \lim_{\sigma\rightarrow0}N(\mu, \sigma)=\delta(\mu)
$$
There are $N$ components in the mixture model and the weights $\pi_i$ are equal, $\pi_i = 1/N$. And $\mu_i=x_i$.

In this model, the uncertainty of the data function comes from the randomness of generating the data with each data point $x_i$ has the probability of $1/N$. The uncertainty is from the weights whereas the basis function $\delta(\mu$ ) is specifically defined. 

To generate $N$ data points from this model, there are $N^N$ Possibilities. Each possibility is equivalent to the sampling of the observed data points, $x_1, x_2, \ldots, x_N$ with replacement. This is exactly the bootstrap method.



In between the two extremes, the uncertainty comes from both the weights and the distribution of the basis distribution parameters.



The Dirac delta case: 
$$
x = \sum_{i=1}^{N} \pi_i \delta(x- x_i)
$$
Since all the observed data $x_i$ are equivalent to each other, by this symmetry, $\pi_i$ are equal. Hence $\pi_i = 1/N$.

