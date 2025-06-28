---

title: "Bayesian Parameter Estimation: Laplace Approximation, MCMC, and Variational Inference"
date: 2025-06-19
typora-root-url: ./..
---



In this note, we estimate distribution parameters from observed data using the posterior distribution. Essentially, the posterior contains all the information about the distribution parameters. There are various methods to infer these parameters. The exact distribution can be numerically estimated with Markov Chain Monte Carlo (MCMC) sampling. However, MCMC can be computationally intensive for large problems. Instead, approximation methods are used. One such method, called Laplace's Approximation, approximates the distribution with a multivariate normal distribution. Another method, known as variance inference (VI), approximates the posterior with a simpler distribution that is optimized to be as close as possible to the true posterior.



We applied Laplace's Approximation and variance inference to data drawn from a normal distribution and compared the results to the MCMC samples. We then used both MCMC and VI on data from a Beta distribution to examine the assumptions of distribution parameter independence in VI. The NumPyro package was used for the MCMC and VI computations.



### Posterior Distribution

Suppose we have observed data, $x$,  drawn from a distribution characterized by parameters $\bf{\theta}$. What is the distribution of $\theta$,  given the observed data?  In other words, we want to find the posterior



$$
p(\theta|x) \propto p(x|\theta)p(\theta)\notag,
$$



where  $p(x\mid\theta)$ is the likelihood and $p(\theta)$ is the prior. The posterior $p(\theta\mid x)$ fully defines the distribution of $\theta$. 



### Laplace's Approximation



The posterior is approximated by a normal distribution.



$$
p(\theta|x)\approx \mathcal{N}(\hat{\theta},\Sigma ) \notag
$$


where $\hat{\theta}$  the maximum a posteriori estimate of $\theta$, and 



$$
\Sigma^{-1} = -\nabla_\theta^2 \log p(\theta|x)|_{\theta=\hat{\theta}}. \notag
$$



To obtain these results,  first Taylor expand the posterior around $\hat{\theta}$, where the posterior reaches its maximum.



$$
\nabla_\theta \log P(\theta|x)|_{\theta=\hat{\theta}}=0. \notag
$$



and 



$$
\log p(\theta|x) \approx \log p(\hat{\theta}|x)+\frac{1}{2}(\theta-\hat{\theta})^T\nabla_\theta^2\log p(\hat{\theta}|x)(\theta-\hat{\theta}). \label{eqn:laplace_approx}
$$




Based on Equation ($\ref{eqn:laplace_approx}$), the analytical form of $p(\theta\mid x)$ is the PDF of a normal distribution.



###  Bayesian Inference 

The Markov Chain Monte Carlo (MCMC) method can generate samples that approach the true posterior distribution $p(\theta\mid x)$. However, MCMC's main drawback is the long computation time.  A  faster but less accurate alternative is the Variational Inference (VI) method. VI uses a tractable and simpler distribution $q_\phi(\theta)$ to approximate the posterior $p(\theta\mid x)$, where $\phi$ is the parameter defining the distribution $q$.  $\phi$ is obtained by numerical optimization that minimizes the Kullback-Leibler divergence between the distributions $q_\phi(\theta)$ and $p(\theta\mid x)$:



$$
\begin{align} \notag
D_{KL}(q||p) &= \int_{\theta \sim q} q_\phi (\theta)\log\frac{q_\phi(\theta)}{p(\theta|x)} d\theta\\ \notag
&= \int_{\theta \sim q}q_\phi (\theta)\log\frac{q_\phi(\theta)p(x)}{p(\theta, x)} d\theta \\ \notag
&= \log p(x) -\left( E_q\left[\log p(\theta, x)\right]-E_q\left[q_\phi(\theta)\right]\right)
\end{align}
$$



where $E_q(f(\theta))=\int_{\theta\sim q} q_\phi(\theta)d\theta$ is the expectation of $f(\theta)$ with respect to distribution $q_\phi(\theta)$.  $\log p(x)$ is the evidence, and the term $E_q\left[\log p(\theta, x)\right]-E_q\left[q_\phi(\theta)\right]$ is the so-called evidence lower bound (ELBO). 



$$
\mathrm{ELBO} =E_q\left[\log p(\theta, x)\right]-E_q\left[q_\phi(\theta)\right] \label{eqn:elbo}
$$




To minimize the KL divergence, we instead maximize the ELBO instead to find the best distribution $q_\phi(\theta)$ to approximate the posterior $p(\theta \mid x)$.



The joint distribution $p(\theta, x)$ is used because the posterior $p(\theta\mid x)$ is unknown, whereas $p(\theta,x)=p(x\mid \theta)p(\theta)$ is known from the likelihood and prior. 



One special form of $q_\phi(\theta)$ is the mean-field approximation, where the components of $\theta$ are assumed to be independent:



$$
q_\phi(\theta) =\Pi_k q_\phi (\theta_k),
$$


where $\theta = \left[\theta_1, \ldots, \theta_k, \ldots\right]$.





### Two Examples



We  use  two examples: estimating the mean, $\mu$,  and the standard deviation, $\sigma$, of a normal distribution, and estimating $\alpha$ and $\beta$ parameters of a Beta distribution, both from data drawn from their respective distributions. For simplicity and without loss of generality, we assume a non-informative prior, as it simplifies the process, making the posterior distribution proportional to the likelihood of the data.



#### Example 1: Normal Distribution



For observed data, $x = x_1, x_2, \ldots, x_N$,  drawn from a normal distribution, the posterior (with the non-informative prior assumption) is



$$
\begin{align}\notag
p(\mu, \sigma|x) & = \Pi_{i=1}^N p(x_i|\mu, \sigma)\\ 
& = \frac{1}{(2\pi)^{\frac{N}{2}}\sigma^N}\exp\left(-\frac{\sum_i(x_i-\mu)^2}{2\sigma^2}\right) \label{eqn:norm_likelihood}
\end{align}
$$




$$
\log p(\mu, \sigma|x) = -\frac{N}{2}\log(2\pi)-N\log\sigma-\frac{\Sigma_i(x_i-\mu)^2}{2\sigma^2} \notag
$$


The code below generates 1,000 random numbers from a normal distribution $\mathcal{N}(-2, 10)$. Figure 1 shows the histogram of these numbers. 



```python
import numpyro
import numpyro.distributions as dist
from jax import random
rng_key = random.PRNGKey(1)

data_norm = dist.Normal(-2, 10).sample(rng_key, (1000,))
```





<figure>
  <center>
  <img src="/assets/images/BI_norm_hist.svg" width="450">
   </center>
  <center>
    <figcaption> Figure 1. Histogram of 1,000 random numbers from a normal distribution with &mu;=-2 and &sigma;=10.
    </figcaption>
  </center>
</figure>


Figure 2 is the contour plot of the joint distribution of $\mu$ and $\sigma$ on a logarithmic scale. It shows that the joint probability $p(\mu, \sigma)$ is not a multivariate normal distribution, as the contours are not symmetrical in $\sigma$. 



<figure>
  <center>
  <img src="/assets/images/BI_normal_posterior_contourplot.svg" width="450">
   </center>
  <center>
    <figcaption> Figure 2. Joint distribution of &mu; and &sigma; on a logarithmic scale.
    </figcaption>
  </center>
</figure>



Another way to see that the joint probability $p(\mu, \sigma)$ is not a multivariate normal distribution is to compute the marginal probabilities for $\mu$ and $\sigma$. Equation ($\ref{eqn:norm_likelihood}$) can be rewritten as



$$
p(\mu, \sigma|x) 
 = \frac{1}{(2\pi)^{\frac{N}{2}}\sigma^N}\exp\left(-\frac{(\mu-\bar{x})^2+(\overline{x^2}-\bar{x}^2)}{2\frac{\sigma^2}{N}}\right), \label{eqn:norm_posterior}
$$



where $\bar{x}=\sum_i x_i/N$ and $\overline{x^2}=\sum_i x_i^2/N$.  For example, 



$$
p(\mu)=\int p(\mu, \sigma) d\sigma \propto N^{\frac{1-N}{2}} \left((\mu-\bar{x})^2 +(\overline{x^2}-\bar{x}^2)\right)^{\frac{1-N}{2}},
$$



which is not a normal distribution. Therefore,  the joint distribution $p(\mu, \sigma)$ is not a multivariate normal distribution because the marginal probability of a multivariate normal distribution is normal. 



##### Laplace's Approximation



The posterior in Equation ($\ref{eqn:norm_posterior}$) can be approximated using Laplace's approximation ($\ref{eqn:laplace_approx}$), which yields a multivariate normal distribution centered at the mode of the posterior. Applying Laplace's approximation gives:



$$
\begin{align} \notag
\frac{\partial \log p}{\partial \sigma} & = -\frac{N}{\sigma} + \frac{N} {\sigma^3}\left[(\mu-\bar{x})^2+(\overline{x^2} - \bar{x}^2)\right]\\ \notag
\frac{\partial \log p}{\partial \mu}&=-\frac{N}{\sigma^2}(\mu -\bar{x})\\ \notag
\frac{\partial^2\log p}{\partial \sigma\partial\mu} & = \frac{\partial^2\log p}{\partial \mu\partial\sigma} = \frac{2N}{\sigma^3}(\mu - \bar{x})\\ \notag
\frac{\partial^2\log p}{\partial\mu^2} & = -\frac{N}{\sigma^2}\\ \notag
\frac{\partial^2\log p}{\partial\sigma^2} & = \frac{N}{\sigma^2}-\frac{3N}{\sigma^4}\left[(\mu-\bar{x})^2+(\overline{x^2} - \bar{x}^2)\right].

\end{align}
$$



At the  posterior,  denoted by $\hat{\mu}$ and $\hat{\sigma}$,   the gradient of the log-posterior vanishes: 



$$
\frac{\partial\log p}{\partial \sigma}|_{\mu=\hat{\mu}, \sigma=\hat{\sigma}}=\frac{\partial\log p}{\partial \mu}|_{\mu=\hat{\mu}. \notag \sigma=\hat{\sigma}}=0, 
$$



Those conditions define the maximum a posteriori (MAP) estimates. When a non-informative (e.g., constant) prior is assumed, the MAP estimates coincide with the maximum likelihood estimates (MLE):



$$
\begin{align} \notag
\hat{\mu} &= \bar{x}\\ \notag
\hat{\sigma} &= \sqrt{\overline{x^2}-\bar{x}^2}. \notag
\end{align}
$$



To construct the Laplace approximation, the precision matrix (inverse covariance) is calculated:  



$$
\Sigma^{-1} = 
\begin{pmatrix}
\frac{N}{\hat{\sigma}^2} & 0\\ \notag
0 & \frac{2N}{\hat{\sigma}^2}
\end{pmatrix},
$$



And thus, the covariance matrix is:



$$
\Sigma = 
\begin{pmatrix}
\frac{\hat{\sigma}^2}{N} & 0\\ \notag
0 & \frac{\hat{\sigma}^2}{2N}
\end{pmatrix}.
$$



Hence, under Laplace's approximation, the joint posterior of $\mu$ and $\sigma$ is approximated by a diagonal multivariate normal distribution. Notably, the variance of $\sigma$ is twice that of $\mu$, reflecting greater uncertainty in estimating the scale parameter compared to the mean.



##### Numerical Infernece: MCMC



The joint posterior distribution of $\mu$ and $\sigma$ is numerically inferred with MCMC sampling. The following code generates $4000$ samples using the NUTS sampler. 



```python
from numpyro.infer import MCMC, NUTS

def norm_model(x):
    mu = numpyro.sample("mu", dist.Uniform(-20, 20))
    sigma = numpyro.sample("sigma", dist.Uniform(0, 50))

    with numpyro.plate("data", len(x)):
        numpyro.sample("obs", dist.Normal(mu, sigma), obs=x)
# NUTS sampler
nuts_kernel = NUTS(norm_model)
norm_mcmc = MCMC(nuts_kernel, num_samples=4000, num_warmup=2000)
norm_mcmc.run(rng_key,data_norm)

# sampling from fitted model
samples_norm_mcmc = norm_mcmc.get_samples()
```



The trace plots of the MCMC samples are shown in Figure 3. 

<figure>
  <center>
  <img src="/assets/images/BI_norm_mcmc_traceplot.svg" width="850">
   </center>
  <center>
    <figcaption> Figure 3. Density and trace plots of the MCMC samples. Vertical lines in the density plots and horizontal lines in the trace plots represent sample means.
    </figcaption>
  </center>
</figure>




Summary statistics for the actual values, sample estimates, and MCMC results are shown below:

|               |  $\mu$  | $\sigma$ |
| :-----------: | :-----: | :------: |
|   True Mean   |  $-2$   |   $10$   |
|  Sample Mean  | $-2.49$ | $10.01$  |
|   MCMC Mean   | $-2.48$ | $10.03$  |
| MCMC Variance | $0.103$ | $0.051$  |



The covariance is of $\mu$ and $\sigma$ in the MCMC samples is $-0.0012$, indicating near independence.  This is further confirmed by the pair plot shown in Figure 4, which shows the marginal and joint posterior distributions.  





<figure>
  <center>
  <img src="/assets/images/BI_norm_mcmc_pairplot.svg" width="450">
   </center>
  <center>
    <figcaption> Figure 4. Marginal and joint density plots of &mu; and &sigma;.
    </figcaption>
  </center>
</figure>







##### Numerical Inference: Variational Inference (VI)



Variational inference offers a faster alternative to MCMC by approximating the  posterior $p(\theta\mid x)$ with a simpler, parameterized distribution $q_\phi(\theta)$.  Assuming independence between $\mu$ and $\sigma$, we use a mean-field approximation:



$$
q_\phi(\mu, \sigma) = q_\phi(\mu) q_\phi(\sigma). \notag
$$



We model $\mu$ using a normal distribution:



$$
\mu \sim \mathcal{N} (\phi_1, \exp(\phi_2)). \notag
$$



and $\sigma >0$ using a log-normal distribution:



$$
\sigma \sim \mathrm{Lognormal}(\phi_3, \exp(\phi_4)).
$$



The relationship between the variational paraemters $\phi$  and moments of each distribution is summarized below: 



| Parameter          | $\mu$          | $\sigma$                                                     |
| ------------------ | -------------- | ------------------------------------------------------------ |
| Distribution       | Normal         | LogNormal                                                    |
| Mean               | $\phi_1$       | $\exp\left(\phi_2 +\frac{\exp(2 \phi_4)}{2}\right)$          |
| Standard Deviation | $\exp(\phi_3)$ | $\sqrt{\left[\exp(\exp(2\phi_4))-1\right]\exp(2\phi_2+\exp(2\phi_4))}$ |



The variational parameters $\phi$ are optimized by maximizing the Evidence Lower Bound (EBO), defined in Equation ($\ref{eqn:elbo}$). Since EBLO involves expectations over $q_\phi(\theta)$, we estimate it using Monte Carlo sampling and $\phi$ are optimized numerically with the Powell algorithm. 



```python
from scipy.stats import optimize


def logp(theta, data):
    mu, sigma = theta
    n = len(data) # number of data points
    return -n*np.log(sigma) - ((mu - data)**2).sum()/sigma**2/2 # const term -n/2*log(2*pi) omitted

def negELBO(phi, data, num_samples):
    """ 
    		phi: an array [phi1, phi2, phi3, phi4], parameters that define q(theta)
    		data: observed data
        num_smaples: number of samples from the q(theta) to numerically calculate ELBO
    """
    key = random.PRNGKey(stats.randint(0, 100_000).rvs())
    mu1, mu2, s1, s2 = phi # parameters of q_phi(theta)

    s1 = np.exp(s1)
    s2 = np.exp(s2)
    # samples of latent variables
    mu_dist = dist.Normal(mu1, s1)
    mu_samples = mu_dist.sample(key, (num_samples,))
    sigma_dist = dist.LogNormal(mu2, s2)
    sigma_samples = sigma_dist.sample(key, (num_samples,))
    # log q(theta)
    logq_mean = np.mean(mu_dist.log_prob(mu_samples)+sigma_dist.log_prob(sigma_samples)) 
    # log p(theta,x)
    lp = [logp(_, data) for _ in zip(mu_samples, sigma_samples)]
    logp_mean = np.mean(lp)
    # elbo
    return -logp_mean+logq_mean
  
 
optimize.minimize(negELBO, [0, 0, 0, 0], args=(data_norm, 10000),  method='Powell')
```



The optimization converged successfully:

```
message: Optimization terminated successfully.
 success: True
  status: 0
     fun: 2805.025146484375
       x: [-2.462e+00  2.305e+00 -1.139e+00 -3.805e+00]
     nit: 3
   direc: [[ 1.000e+00  0.000e+00  0.000e+00  0.000e+00]
           [ 0.000e+00  0.000e+00  0.000e+00  1.000e+00]
           [ 0.000e+00  0.000e+00  1.000e+00  0.000e+00]
           [ 4.248e-01 -8.721e-01  1.741e-01  7.986e-01]]
    nfev: 178
```



| $\phi_1$ | $\phi_2$ | $\phi_3$ | $\phi_4$ |
| :------: | :------: | :------: | :------: |
|  -2.462  |  2.305   |  -1.139  |  -3.805  |



From these optimized parameters, we compute the means and standard deviations:



|     Parameter      |  $\mu$  | $\sigma$  |
| :----------------: | :-----: | :-------: |
|    Distribution    | Normal  | LogNormal |
|        Mean        | $-2.46$ |  $10.02$  |
| Standard Deviation | $0.32$  |  $0.22$   |

Figure 5 shows strong agreement between the MCMC and VI results. This is largely due to the near-independence of $\mu$ and $\sigma$, making the mean-field approximation effective in this case. 

<figure>
  <center>
  <img src="/assets/images/BI_norm_MCMC_vs_VI.svg" width="750">
   </center>
  <center>
    <figcaption> Figure 5. Comparison of MCMC and VI results. Histograms represent MCMC samples; solid lines are probability densities from VI.
    </figcaption>
  </center>
</figure>





#### Example 2: Beta Distribuiton



In this example, 1,000 samples are generated from a known Beta distribution.  The PDF of a Beta distribution $\mathrm{B}(\alpha, \beta)$ is 



$$
\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} x^{\alpha-1} (1-x)^{\beta-1}, \,\, x\in (0,1) \notag
$$



The normalization factor,  $\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}$, introduces a correlation between $\alpha$ and $\beta$ in the likelihood and posterior distributions.



Figure 6 shows the histogram of 1,000 samples from a  Beta distribution with $\alpha=2$ and $\beta=5$.

<figure>
  <center>
  <img src="/assets/images/BI_beta_hist.svg" width="550">
   </center>
  <center>
    <figcaption> Figure 6. Histogram of 1,000 samples from a Beta distribtuion with &alpha;=2 and &beta;=5.
    </figcaption>
  </center>
</figure>



We use three Bayesian inference techniques to recover the parameters:

1. MCMC with NUTS
2. Variational Inference (VI) with a diagonal normal approximation (assumes $\alpha$ and $\beta$ are independent)
3. VI with a full multivariate normal approximation (captures correlation between $\alpha$ and $\beta$ )

The code below generates the data and performs inference:

```python
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS,SVI, Trace_ELBO, autoguide
from jax import random

# generate observed data
rng_key = random.PRNGKey(1)
data_beta = dist.Beta(2, 5).sample(rng_key, (1000,))

# model of observed data
def beta_model(x):
    alpha = numpyro.sample("alpha", dist.Uniform(0, 20))
    beta = numpyro.sample("beta", dist.Uniform(0, 20))

    with numpyro.plate("data", len(x)):
        numpyro.sample("obs", dist.Beta(alpha, beta), obs=x)
        
#####################
# inference with MCMC
#####################
nuts_kernel = NUTS(beta_model)
beta_mcmc = MCMC(nuts_kernel, num_samples=4000, num_warmup=2000)
beta_mcmc.run(rng_key,data_beta)
# sampling from fitted model
samples_beta_mcmc = beta_mcmc.get_samples()

#####################
# inference with VI 
#####################
# define function for VI inference and sampling
def vi_samples(random_key, data, model, guide, num_samples):
    """
    Inputs:
        random_key: random key
        data: observed data
        model: model of the observed data
        guide: guide that generates the approximate distribution q
        num_samples: number of samples drawn from the inferred posterior
    Outputs:
        samples from the inferred posterior distribtuion
    """
    niter = 50000; # number of iterations in optimization
    svi = SVI(model, guide,
          optim=numpyro.optim.Adam(step_size=0.005),
          loss = Trace_ELBO())
    svi_result = svi.run(random_key, niter ,data)
    samples = guide.sample_posterior(random_key, 
              svi.get_params(svi_result.state), sample_shape=(num_samples,))
    return samples
 
# VI with diagonal normal distribution (independent alpha and beta)
guide_diag = autoguide.AutoDiagonalNormal(beta_model)
samples_diag = vi_samples(rng_key, data_beta, beta_model, guide_diag, 2000)

# VI with multivariate normal distribution (correlated alpha and beta)
guide_multi = autoguide.AutoMultivariateNormal(beta_model)
samples_multi = vi_samples(rng_key, data_beta, beta_model, guide_multi, 2000)
```



Figure 7 displays the joint density estimates of $\alpha$ and $\beta$. Both MCMC and VI with a multivariate normal approximation capture the correlation between $\alpha$ and $\beta$. As expected, VI with a diagonal normal approximation fails to reflect this correlation due to the independence assumption. 

<figure>
  <center>
  <img src="/assets/images/BI_beta_lapha_beta_density.svg" width="600">
   </center>
  <center>
    <figcaption> Figure 7. Joint density plots of &alpha; and &beta;. Both MCMC and multivariate VI capture parameter correlation, while diagonal VI does not.
    </figcaption>
  </center>
</figure>


Figure 8 compares the marginal distributions of $\alpha$ and $\beta$  from the three methods. MCMC and VI with a multivariate approximation are in close agreement. However, VI with diagonal approximation underestimates the posterior variance, highlighting the limitation of assuming independence in the variational parameters.   



<figure>
  <center>
  <img src="/assets/images/BI_beta_MCMC_VI_diag_multi_density.svg" width="650">
   </center>
  <center>
    <figcaption> Figure 8. Marginal probabilities of &alpha; and &beta;. VI with a diagonal approximation underestimates the uncertainty compared to MCMC and multivariate VI
    </figcaption>
  </center>
</figure>




### Conclusion



Bayesian inference provides a principled framework for estimating uncertain parameters.

- MCMC offers accurate sampling from the true posterior but is computationally intensive.
- Laplace's approximation provides a fast, local Gaussian approximation near the posterior mode.
- Variational Inference (VI) offers a flexible and efficient alternative, approximating the posterior with a simpler distribution.

However, the choice of variational family is critical. Assuming independence between parameters (as in a diagonal normal approximation) can lead to underestimated uncertainty and poor approximation of posterior structure, especially when the true posterior exhibits correlation.

This case study demonstrates that when parameters are correlated, using a full-covariance variational approximation is crucial for accurately capturing the posterior geometry.
