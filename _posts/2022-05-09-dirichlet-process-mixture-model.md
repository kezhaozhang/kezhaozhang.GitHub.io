---
title: Dirichlet Process in Mixture Model
typora-root-url: ../../kezhaozhang.GitHub.io
---



We use the Dirichlet process to generate the weights in the mixture model to determine the optimal number of components automatically. 

## Dirichlet Process

We usually do not know the number of components in the mixture model before inference.  We can find the optimal number of components that maximizes the information criterion like WAIC. And this requires the evaluation of multiple models with different numbers of components.

Using the Dirichlet process to define the priors for component weights, there is no need to specify the number of components in the model. The posterior distribution of the MCMC sampling will find the components with dominant weights.

The weights of the components, $\pi_n$, in the mixture model are generated using the stick-breaking process


$$
\pi_n = \beta_n \Pi_{k=1}^{n-1}(1-\beta_k),
$$


where $\beta_n \sim \rm{Beta}(1, \alpha)$. 

$\alpha$ is the parameter that controls the distribution of $\pi_n$. As shown in Figure 1, when $\alpha$ is small, the weight distribution is very skewed: with a few close to 1 and the rest near 0. As $\alpha$ increases the weight distribution becomes more uniform. 

<figure>
  <center>
  <img src="/assets/images/dp_stick_breaking_hist.svg" width="700">
   </center>
  <center>
  <figurecaption>
    Figure 1. &beta; and &pi; are generated with the stick-breaking method for various &alpha;. As &alpha; increases, the weight distribution becomes more spread out and uniform. 
  </figurecaption>
  </center>
</figure>



## Mixture Model 

The generative model of the data $x$ is a combination of $K$ Student's t distributions:


$$
x \sim \sum_{i=1}^K \pi_i f(x\mid\mu_i, \sigma_i, \nu_i),
$$


where the weights $\sum_{i=1}^K\pi_i=1$, and


$$
f(x\mid \mu, \sigma, \nu)=\frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\Gamma\left(\frac{\nu}{2}\right)}\frac{1}{\sqrt{\pi\nu}\sigma}\left[1+\frac{1}{\nu}\left(\frac{x-\mu}{\sigma}\right)^2\right]^{-\frac{\nu+1}{2}}.
\label{eqn:student}
$$


In the model inference, the weights $\pi_i$ are generated with the Dirichlet process using the stick-breaking scheme.



## An Example

In a previous [post](https://kezhaozhang.github.io/2022/01/08/student.html), we used a Student's t mixture model to separate the groups in the 2D data, as shown in Figure 2. We compared the models with one, two, and three components in that analysis and selected the best model using WAIC.  In this note, we use the Dirichlet process to determine the optimal number of components automatically.

<figure>
  <center>
  <img src="/assets/images/dp_orig_scatterplot.svg" width="500">
   </center>
  <center>
  <figurecaption>
    Figure 2. The 2D scatter plot of the data. There appears to be more than one group in the data.  
  </figurecaption>
  </center>
</figure>

### Data Transformation

The two-dimensional problem is transformed into a one-dimensional one by calculating the residual of a spline fit of $y$ on $x$.  A mixture model of Student's t distributions is applied to the residual to determine the number of components and separate the components.

<figure>
  <center>
  <img src="/assets/images/dp_residual_hist.svg" width="500">
   </center>
  <center>
  <figurecaption>
    Figure 3. Distribution of the residual of the spline fit of the original two-dimensional data.  
  </figurecaption>
  </center>
</figure>

### PyMC3 Model

The mixture model and its inference are implemented using the PyMC3 package. The code and the model diagram are displayed below. In the MCMC sampling, four chains with 1000 samples are generated.  30 components ($K=30$) are used. 

```python
import pymc3 as pm
import theano.tensor as tt

# stick breaking
def stick_breaking(beta):
    return beta * tt.concatenate([[1], tt.cumprod(1 - beta)[:-1]])

K=30

with pm.Model() as model:
    # priors for weights, generated from Dirichlet process
    α = pm.Exponential('α',0.5)
    β = pm.Beta('β', 1, α, shape = K)
    w = pm.Deterministic('w', stick_breaking(β))

    # priors for Student's-t distribution
    μ = pm.Normal('μ', mu=0, sigma=10, shape=K)
    σ = pm.HalfNormal('σ', 1, shape=K)
    ν = pm.Exponential('ν', 1, shape=K)
    
    dists = pm.StudentT.dist(nu=ν, mu=μ, sigma=σ, shape=K)
    # data: the residual of the spline fit (Figure 3)
    obs = pm.Mixture('obs', w = w, comp_dists=dists, observed=data) 
    
    trace = pm.sample(return_inferencedata=True)
```

<figure>
  <center>
  <img src="/assets/images/dp_model.svg" height="600">
   </center>
  <center>
  <figurecaption>
    Figure 4. Student's-t mixture model with weights generated with Dirichlet Process.
  </figurecaption>
  </center>
</figure>

### Result

#### Number of Components

The posteriors obtained from the MCMC show two dominant components (Figure 5 and Figure 6). The few dominant components are consistent with the small value of the posterior of $\alpha$ (Figure 7). 

<figure>
  <center>
  <img src="/assets/images/dp_w_barchart.svg" height="350">
   </center>
  <center>
  <figurecaption>
    Figure 5. Weight posterior averaged over chains and samples. There are two dominant components.
  </figurecaption>
  </center>
</figure>

<figure>
  <center>
  <img src="/assets/images/dp_w_posterior.svg" height="620">
   </center>
  <center>
  <figurecaption>
    Figure 6. Posterior distributions of the weights of the first six components.
  </figurecaption>
  </center>
</figure>



<figure>
  <center>
  <img src="/assets/images/dp_alpha_posterior.svg" height="300">
   </center>
  <center>
  <figurecaption>
    Figure 7. Posterior distribution of &alpha;. Because &alpha; is ~ 0.36, there are very few components with non-zero weights.
  </figurecaption>
  </center>
</figure>


#### Component Assignment

We calculate the likelihood ratio for each data point $x$ as follows:


$$
\mathrm{log\_ratio} = \log\frac{\pi_1 f(x\vert \mu_1, \nu_1, \sigma_1)}{\pi_2 f(x\vert \mu_2, \nu_2, \sigma_2)}.
$$


Then the average of $\textrm{log\_{}ratio}$ over posterior distributions is calculated. The component the data point belongs to is assigned according to:


$$
\mathrm{component} = \left\{ \begin{array}{lll}
															1 & : & \left<\mathrm{log\_ratio}\right> >0 \\
															2 & : & \left<\mathrm{log\_ratio}\right> \le 0
                            \end{array}
                     \right.
$$


where $\left<\textrm{log\_{}ratio}\right>$  is the average over the posterior distributions.

The original two-dimensional data are plotted with the components represented by color in Figure 8. The component assignment is very comparable to that in the previous [post](https://kezhaozhang.github.io/2022/01/08/student.html).

<figure>
  <center>
  <img src="/assets/images/dp_final_xy_scatterplot.svg" height="350">
   </center>
  <center>
  <figurecaption>
    Figure 8. Scatter plot of the original data with the two components in different colors. The component assignment is calculated from the posterior distributions using Equation (4) and Equation (5).
  </figurecaption>
  </center>
</figure>


## Conclusion

We applied the Dirichlet process to the mixture model and automatically obtained the optimal number of components from the resulting posterior distribution. The result is comparable to the previous analysis of model selection with WAIC. The Dirichlet process provides a flexible building block for priors in the mixture model for density estimation and clustering.