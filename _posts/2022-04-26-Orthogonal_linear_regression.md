---
title: "Multivariate Orthogonal Linear Regression Using PyMC"
date: 2022-04-26
typora-root-url: ../../kezhaozhang.GitHub.io
---

This note describes a multivariate orthogonal linear regression method using the PyMC probabilistic programming package. The formulation is based on an intuitive geometrical interpretation. 

Orthogonal regression considers the errors in both the predicted and the predictors.  A comparison of the ordinary linear regression (OLS) and orthogonal linear regression is illustrated in a one-dimensional case in Figure 1.

<figure>
  <center>
  <img src="/assets/images/odr_OLS_vs_ODR.svg" height="450">
   </center>
  <center>
  <figurecaption>
    Figure 1. Comparison of orthogonal and ordinary linear regressions in the one-dimensional case. Ordinary linear regression (left) assumes that only <i>y</i> has an error, and the regression coefficients are obtained by minimizing the square of the difference between actual and predicted <i>y</i>; Orthogonal regression assumes that both <i>y</i> and <i>x</i> contain errors and minimizes the square of the distance from the data point to the regression line.
  </figurecaption>
  </center>
</figure>



### Theory

A linear regression model is a surface in a high-dimensional space. An orthogonal regression model minimizes the perpendicular distance from the data points to the surface (see Figure 2).

<figure>
  <center>
  <img src="/assets/images/odr_ODR_geometry.svg" height="400">
   </center>
  <center>
  <figurecaption>
    Figure 2. When there are two or more predictors, the regression model is a surface in a high-dimensional space.
  </figurecaption>
  </center>
</figure>


The surface of the linear regression model is defined by


$$
y = X\cdot\beta + C,
$$


where $y$ is the predicted, $X$ the predictors, $\beta$ regression coefficients, and $C$ the intercept.



The perpendicular distance between a point $(X_i, y_i)$ and the surface is


$$
d = \frac{y_i-X_i\cdot\beta-C}{\sqrt{1+\beta^T\beta}}.
\label{eq:d}
$$


The value of $d$ is positive on one side of the model surface and negative on the other side and 0 when on the surface.

It is reasonable to assume the distribution of $d$ is a normal distribution with zero mean. the distribution of $\beta$  can be inferred using the Markov chain Monte Carlo (MCMC) method.



### PyMC3 Implementation

The key is to calculate the perpendicular distance to the regression surface, $d$. Then the likelihood of $d$ is added using `pm.Potential` function because there is no way to use a predefined distribution to achieve the same effect. 

```python
import pymc3 as pm
import theano.tensor as tt

 
with pm.Model() as model:
  	# predictor: x shape (N, ndim)
		# predicted: y shape (N,)
		# N: sample size
		# ndim: dimension of predictor
    s = 1
    sigma_intercept = 1
    # Priors
    beta = pm.Normal('beta', 0, sigma=s, shape=ndim)  
    intercept = pm.Normal('intercept', 0, sigma_intercept)
    sigma = pm.Exponential('sigma', 0.5)

    f = tt.sqrt(1+tt.sum(beta**2)) # normalization factor
    # perpendicular distance to regression surface
    d = (y - tt.dot(x, beta) - intercept)/f 
    #Likelihood
    pm.Potential('Likelihood', pm.Normal.dist(mu=0, sigma=sigma).logp(d))

 		# sampling
		trace = pm.sample(return_inferencedata=True)
```

As shown in the diagram in Figure 3, the model contains priors for regression coefficients $\beta$, intercept, and the standard deviation of $d$. The prior for $\beta$ regulates its magnitudes and affects the performance of the MCMC, which will be discussed in a later section.

<figure>
  <center>
  <img src="/assets/images/odr_model_diagram.svg" height="300">
   </center>
  <center>
  <figurecaption>
    Figure 3. Diagram of the orthogonal linear regression model.
  </figurecaption>
  </center>
</figure>



### An Example

The PyMC3 orthogonal linear regression is applied to simulated data with three predictors.

#### Data

The data are generated with the code below. Some correlation is added among the predictors to make the regression harder.

```python
x1 = np.random.randn(1000)
x2 = 0.6*x1 + np.random.randn(1000)
x3 = 0.25*x1 + np.random.randn(1000)

# the predicted
y = 2*x1 -5*x2 + x3 + np.random.randn(1000)

# add noise to predictors
x1 = x1 + np.random.randn(1000)
x2 = x2 + np.random.randn(1000)
x3 = x3 + np.random.randn(1000)

x = np.c_[x1,x2,x3]
```

<figure>
  <center>
  <img src="/assets/images/odr_data_pairplot.svg" height="600">
   </center>
  <center>
  <figurecaption>
    Figure 4. Pairwise plots of the simulated data.
  </figurecaption>
  </center>
</figure>

#### MCMC Samples

The posterior traces and densities are plotted in Figures 5 and  6.  The regression coefficients are close to the actual values used in the simulated data. 

<figure>
  <center>
  <img src="/assets/images/odr_mcmc_traces.svg" height="500">
   </center>
  <center>
  <figurecaption>
    Figure 5. Posterior traces. 
  </figurecaption>
  </center>
</figure>

<figure>
  <center>
  <img src="/assets/images/odr_mcmc_posterior_plots.svg" height="500">
   </center>
  <center>
  <figurecaption>
    Figure 6. Density plots of the posteriors. 
  </figurecaption>
  </center>
</figure>

#### Comparison to Other Models

The PyMC3 model result is compared with the scipy orthogonal distance regression package (https://docs.scipy.org/doc/scipy/reference/odr.html) and ordinary linear regression (OLS).  The PyMC3  and spicy.odr models have comparable results, whereas the OLS model is significantly worse.

|            |    $\beta_0$    |    $\beta_1$    |   $\beta_2$    |   Intercept    |
| :--------: | :-------------: | :-------------: | :------------: | :------------: |
| True Value |        2        |       -5        |       1        |       0        |
|   PyMC3    |  2.22$\pm$0.22  | -4.97$\pm$0.17  | 0.86$\pm$0.17  | 0.12$\pm$0.18  |
| scipy.odr  |  2.40$\pm$0.18  | -5.14$\pm$0.19  | 0.88$\pm$0.13  | 0.13$\pm$0.18  |
|    OLS     | 0.281$\pm$0.085 | -2.48$\pm$0.078 | 0.62$\pm$0.082 | 0.076$\pm$0.12 |

#### Distribution of $d$

In the model, normal distribution for the distance of data point to the model surface is assumed.  To check the normality of the distribution of $d$, 100 predictive posterior samples are drawn using `pm.sample_posterior_predictive`  from the MCMC samples. Then $d$ is calculated using Equation ($\ref{eq:d}$). 

For each posterior predictive sample, the corresponding distribution of $d$ is tested with `scipy.stats.normaltest`, which combines skewness and kurtosis for the test. The resulting *p-value* is much larger than the conventional $0.005$ threshold. Therefore the distribution of $d$ is comparable to normal distribution.

<figure>
  <center>
  <img src="/assets/images/odr_mcmc_ppc_d_distribution_normality_test_pvalue_s1.svg" height="450">
   </center>
  <center>
  <figurecaption>
    Figure 8. <i>p=value</i> of the normality test of distribution of <i>d</i> calculated from the posterior predictive samples.
    </figurecaption>
  </center>
</figure>



### Discussion

It is essential to scale the predictors if they have very different scales. A balanced contribution from components of the scaled data helps the robustness of the MCMC sampling.  Multiple scalers can be used; one example is a standard scaler that standardizes the features by removing the mean and scaling to unit variance.

Another critical factor is the prior for the regression coefficients $\beta$. In this model,  a normal distribution with a standard deviation $s$ is used: $\beta \sim N(0, s)$.  $s$ effectively regularizes the magnitudes of $\beta$. Large $s$ leads to inconsistent and divergent MCMC sampling results. As shown in Figure 7, where the distributions of predicted $y$ with $s=1$ and $s=5$ and actual values are compared. With $s=5$, the range of the predicted $y$ is extremely large and unrealistic. The MCMC sampling does not converge with $s=5$. Therefore, it is helpful to use prior predictive checks to choose the priors such that the predicted values are reasonable.

<figure>
  <center>
  <img src="/assets/images/odr_mcmc_prior_check_y_cdf.svg" height="450">
   </center>
  <center>
  <figurecaption>
    Figure 8. CDFs of predicted <i>y</i> sampled from the priors with <i>s=1</i> and <i>s=5</i>, compared with the actual <i>y</i>. 
  </figurecaption>
  </center>
</figure>


The PyMC model in this note is developed for linear regressions. The next step is to create a model for nonlinear regressions using the formulation of scipy.odr and ODRPACK ([ODRPACK User’s Guide](https://docs.scipy.org/doc/external/odrpack_guide.pdf)). The equation relates the predicted and predictors


$$
y_i-\delta_i = f(x_i - \epsilon_i\vert\beta),
$$


where $f(x\vert\beta)$ is a nonlinear function of $x$ with model parameters $\beta$. The errors of the predicted, $y$,  and the predictor, $x$, are $\delta_i$ and $\epsilon_i$, respectively. This formulation is more general than the geometrical approach for linear regression. A different approach is likely needed. 
