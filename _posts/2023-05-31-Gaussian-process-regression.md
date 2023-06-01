---
title: "Gaussian Process Regression"
typora-root-url: ./..
---

### Introduction

Gaussian Process regression is a non-parametric approach that models the relationship between a set of input variables, denoted as $\mathbf{x}$, and their corresponding outputs, denoted as $\mathbf{y}$. It leverages two key results to estimate the distribution of $\mathbf{y}_2$ at a given $\mathbf{x}_2$, based on observations $\mathbf{y}_1$ at $\mathbf{x}_1$.

1. **Conditional probability distribution of multivariate Gaussian Distribution**:

   The unknown values $\mathbf{y}_2$ and observations $\mathbf{y}_1$ are assumed to follow a multivariate Gaussian distribution:


$$
\mathbf{y} = \left[
\begin{array}{c}
\mathbf{y}_1 \\
\mathbf{y}_2
\end{array}
\right] \sim \mathcal{N}(0, \Sigma),
$$


​	where the covariance matrix $\Sigma$ can be expressed in block matrix form


$$
\Sigma = \left[
\begin{array}{cc}
\Sigma_{11} & \Sigma_{12}\\
\Sigma_{21} & \Sigma_{22}
\end{array}
\right],
$$


​	where $\Sigma_{ij}=\mathrm{cov}(\mathbf{y}_i, \mathbf{y}_j)$, $i=1, 2$.

​	The distribution of $\mathbf{y}_2$ conditioned on $\mathbf{y}_1$ follows a Gaussian distribution:


$$
\mathbf{y}_2 \mid \mathbf{y}_1 \sim \mathcal{N}(\mathbf{\mu}_1+\Sigma_{21}\Sigma_{11}^{-1} (\mathbf{y}_1-\mathbf{\mu}_1),  
\Sigma_{22}-\Sigma_{21}\Sigma_{11}^{-1}\Sigma_{12}).
\label{eqn:conditional_dist}
$$

​	where $\mathbf{\mu}_1$ and $\mathbf{\mu}_2 = \mathbf{y}_2$ represent the means of $\mathbf{y}_1$ and $\mathbf{y}_2$, respectively.

​	A derivation of Equation ($\ref{eqn:conditional_dist}$) can be found in an earlier post.

2. **Covariance of $\mathbf{y}$ from $\mathbf{x}$**:

   
   
   The covariance of $\mathbf{y}$ is determined solely by the predictor $\mathbf{x}$:


$$
\Sigma_{ij}=\mathrm{cov(\mathbf{y}_i, \mathbf{y}_j)} = K(\mathbf{x}_i, \mathbf{x}_j),
$$

​	where $K$ denotes the kernel function. 



#### Kernel

In this report, we adopt the squared exponential kernel:
$$
K(\mathbf{x}_i, \mathbf{x}_y) = \alpha \exp\left({-\frac{\mid \mathbf{x}_i-\mathbf{x}_j\mid^2}{2 l^2}}\right),
\label{eqn:kernel}
$$

where $\alpha$ represents the amplitude and $l$ corresponds to the length scale.



#### Kernel Hyperparameters Optimization

To optimize the kernel hyperparameters $\theta$ and maximize the likelihood $p(\mathbf{y} \mid \theta)$, we start with the training data $\mathbf{x}$ and $\mathbf{y}$. The goal is to find the best $\theta$ that fits the data.

The likelihood function is defined as:

$$
p(\mathbf{y}\mid\theta) = \frac{1}{\sqrt{(2 \pi)^d \mathrm{det}(\Sigma)}}\exp\left({- \frac{(\mathbf{y}-\mathbf{\mu})^T\Sigma^{-1} (\mathbf{y}-\mathbf{\mu})}{2}}\right),
$$



where $\mathbf{\mu}$ represents the mean of $\mathbf{y}$.

Alternatively, we can minimize the negative log-likelihood function:
$$
L = \log(\mathrm{det}(\Sigma)) + (\mathbf{y}-\mathbf{\mu})^T \Sigma^{-1} (\mathbf{y}-\mathbf{\mu}),
$$

where $\Sigma = K(\mathbf{x}, \mathbf{x} \mid \theta)$ denotes the covariance matrix.



By finding the optimal $\theta$ that minimizes the negative log-likelihood, we can determine the best-fit kernel hyperparameters that maximize the likelihood $p(\mathbf{y} \mid \theta)$.



In this report, we assume the mean for $\mathbf{y}$ is zero, i.e., $\mathbf{\mu}=0$.

### Implementation



Below is an implementation of Gaussian process regression in Python:

```python
import numpy as np
from scipy.spatial.distance import cdist

# Squared exponetial kernel
def squared_exp(x1, x2, l, α=1):
    """ Squared Exponential covariance kernel between x1 and x2
    
     K(x_1, x_2) = \alpha \exp(-\frac{\mid x_1-x_2\mid^2}{2 l^2})
    Inputs:
        x1: array of data, each row is one observation
        x2: array of data, each row is one observation
        l: length scale
        α: amplitude scale
    """
    distance = cdist(x1, x2, metric='sqeuclidean')
    return α*np.exp(-0.5*distance/l**2)

#Gaussian Process regression
def gpr(x1, y1, x2, kernel):
    """ Gaaussian Process regression
    Inputs:
    x1: training data coordinate
    y1: training data values
    x2: coordinates of predicted values
    kernel: kernel. e.g. lambda x1, x2: squared_exp(x1,x2, 1, 1)
    Outputs:
        σ: standard deviation of the predicted
        μ: predicted mean, assuming 0 prior mean
    """
    
    Σ11 = kernel(x1, x1)
    Σ12 = kernel(x1, x2)
    Σ21 = Σ12.T
    Σ22 = kernel(x2, x2)
    
    m = Σ21 @ np.linalg.inv(Σ11)
    
    #mean of prediction assuming 0 mean in prior
    μ = m @ y1
    # standard deviation of prediction
    v = np.diag(Σ22 - m @ Σ12)
    σ = np.sqrt(np.maximum(0, v)) #
    
    return σ, μ
```



### Example

We use the same generated data as in the Sklearn Gaussian Process Regression [User Guide](https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy_targets.html#sphx-glr-auto-examples-gaussian-process-plot-gpr-noisy-targets-py). 

#### Noiseless Case

In this scenario, the observations $\mathbf{y}*1$ are devoid of noise. The covariance matrix is given by $\Sigma_{ij} = K(\mathbf{x}_i, \mathbf{y}_j)$. We randomly select six data points from the function $y = x \sin x$ using the following code snippet. Figure 1 illustrates the data points and the corresponding function curve.

```python
import numpy as np

X = np.linspace(start=0, stop=10, num=1_000).reshape(-1, 1)
y = np.squeeze(X * np.sin(X))

rng = np.random.RandomState(1)
training_indices = rng.choice(np.arange(y.size), size=6, replace=False)
X_train, y_train = X[training_indices], y[training_indices]
```

<figure>
  <center>
  <img src="/assets/images/gpr_data.svg" width="600">
   </center>
  <center>
    <figcaption> Figure 1. The true function (blue line) and six data points (black) randomly selected from the curve for Gaussian Process regression.
    </figcaption>
  </center>
</figure>



##### Kernel Hyperparameter Optimization

We examine the negative log-likelihood's dependence on the kernel hyperparameters $\alpha$ and $l$ using the kernel function ($\ref{eqn:kernel}$) and the observation data `X_train` and `y_train`. Figure 2 illustrates the contour plot of the negative log-likelihood. 

<figure>
  <center>
  <img src="/assets/images/gpr_kernel_hyperparameter_contour.svg" width="600">
   </center>
  <center>
    <figcaption> Figure 2. Negative log-likelihood as a function of the kernel hyperparameters <i>&alpha;</i> and <i>l</i>. The contour values are on a logarithmic scale.
    </figcaption>
  </center>
</figure>



We determine that $l=1.43364382$ and $\alpha=25.22123667$ maximize the likelihood $p(\mathbf{y}\mid \theta)$ and use these values for the model.

```python
def negLogLikelihood(θ, kernel, x, y):
    """ Negative log likelihood with squared exponential kernel
    Inputs:
        θ: kernel hyperparameters, an array
        kernel: kernel function k(x,y,θ)
        x: coordinates of the data
        y: values of the data
    Output:
        negative log likelihood
    """
    
    Σ = kernel(x,x,θ)
    
    det = np.maximum(1e-10, np.linalg.det(Σ))
    L = np.log(np.linalg.det(Σ)) + y @ np.linalg.inv(Σ) @ y

    return L

kernel = lambda u, v, s: squared_exp(u, v, *s) #squared_exp kernel with l=s[0], α=s[1]
# objective function: negative log-likelihood
objective = lambda θ: negLogLikelihood(θ, kernel, X_train, y_train) 
# find hyperparameter values that maximize likelihood (minimizes neg log-likelihood)
minimize(objective, [1,1])
```

```
      fun: 18.872678814160338
 hess_inv: array([[2.00444679e-01, 6.61462890e+00],
       [6.61462890e+00, 2.87167294e+02]])
      jac: array([1.19209290e-06, 4.76837158e-07])
  message: 'Optimization terminated successfully.'
     nfev: 78
      nit: 23
     njev: 26
   status: 0
  success: True
        x: array([ 1.43364382, 25.22123667])
```



##### Prediction

The code provided below demonstrates how to predict values at other locations and plot the mean and variation of the predicted values. The resulting plot is shown in Figure 3.

```python
import matplotlib.pyplot as plt
#prediction
θ = [ 1.43943929, 25.46547279]
kernel = lambda u, v: squared_exp(u,v, *θ)
σ, μ = gpr(X_train, y_train, X, kernel = kernel)

#plot result
plt.figure(dpi=300)
plt.plot(X, μ, 'C1', label='Mean Prediction')
plt.plot(X, y, 'C0', label='$f(x) = x\sin x$');
plt.scatter(X_train, y_train, c='k', s=45, label='Observation');
plt.fill_between(X.ravel(), μ-2*σ, μ+2*σ, alpha=0.25, color='C1', label='95% Confidence Interval'); 
plt.legend(loc='upper left');
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
```

<figure>
  <center>
  <img src="/assets/images/gpr_prediction_zero_mean_0.svg" width="600">
   </center>
  <center>
    <figcaption> Figure 3. Predicted mean value and confidence interval. The uncertainty is close to zero at the observation data points.
    </figcaption>
  </center>
</figure>



#### Noisy Observation

When noise is present in the observations, we have noisy values defined as $y_i^{\mathrm{noise}} = y_i + \epsilon_i$, and the covariance between the noise terms is given by $\mathrm{cov}(\epsilon_i, \epsilon_j) = \sigma^2\delta_{ij}$. The covariance matrix of the observations $\mathbf{y}_1$ is calculated as:


$$
\Sigma_{11} = K(\mathbf{x}_1, \mathbf{x}_1) + \sigma^2 \mathbf{I},
$$


where $\mathbf{I}$ is the identity matrix, and $\sigma$ represents the standard deviation of the noise. To perform Gaussian Process regression and likelihood calculation with noisy observations, we need to use this modified covariance matrix $\Sigma_{11}$. The code below shows how to calculate the negative log-likelihood with a squared exponential kernel and the modified covariance matrix.

```python
def negLogLikelihood_noisy(θ, noise_std, kernel, x, y):
    """ Negative log-likelihood with squared exponential kernel
    Inputs:
        θ: kernel hyperparameters, an array
        noise_std: noise standard deviation
        kernel: kernel function k(x,y,θ)
        x: coordinates of the data
        y: values of the data
    Output:
        negative log likelihood
    """
    
    Σ = kernel(x,x,θ) + np.eye(len(x))*noise_std**2
    L = np.log(np.linalg.det(Σ)) + y @ np.linalg.inv(Σ) @ y
   
    return L


def gpr_noisy(x1, y1, x2, kernel, noise_std):
    """ Gaussian Process regression
    Inputs:
    x1: training data coordinate
    y1: training data values
    x2: coordinates of predicted values
    kernel: kernel. e.g. lambda x1, x2: squared_exp(x1,x2, 1, 1)
    Outputs:
        σ: standard deviation of the predicted
        μ: predicted mean, assuming 0 prior mean
    """
    
    Σ11 = kernel(x1, x1) + noise_std**2*np.eye(len(x1))
    Σ12 = kernel(x1, x2)
    Σ21 = Σ12.T
    Σ22 = kernel(x2, x2)
    
    #m = Σ21 @ np.linalg.inv(Σ11 + np.identity(len(Σ11))*1e-10)
    m = Σ21 @ np.linalg.inv(Σ11)
    
    #m = scipy.linalg.solve(Σ11, Σ12, assume_a='pos').T 
    
    #mean of prediction assuming 0 mean in prior
    μ = m @ y1
    # standard deviation of prediction
    v = np.diag(Σ22 - m @ Σ12)
    σ = np.sqrt(np.maximum(0, v))
    
    return σ, μ
```



We can generate noisy observed data, optimize the kernel hyperparameters, and predict values using the following code:

```python
# Generate noisy observed data
noise_std = 0.75
y_train_noisy = y_train + rng.normal(loc=0.0, scale=noise_std, size=y_train.shape)
```



```python
# Optimize kernel hyperparameters
kernel = lambda u, v, s: squared_exp(u, v, *s) #squared_exp kernel with l=s[0], α=s[1]
objective_noisy = lambda θ: negLogLikelihood_noisy(θ, noise_std, kernel, 
                                    X_train, y_train_noisy) # objective function: negative log likelihood
minimize(objective_noisy, [1,1])
```

```
     fun: 19.915965193360737
 hess_inv: array([[  0.16438608,   3.20226368],
       [  3.20226368, 127.14156175]])
      jac: array([-3.57627869e-06,  0.00000000e+00])
  message: 'Optimization terminated successfully.'
     nfev: 66
      nit: 20
     njev: 22
   status: 0
  success: True
        x: array([ 1.10435408, 18.30415574])
```



```python
# Predict and plot
θ = [ 1.10435408, 18.30415574]
kernel = lambda u, v: squared_exp(u,v, *θ)
σ, μ = gpr_noisy(X_train, y_train_noisy, X, kernel = kernel, noise_std=noise_std)

plt.figure(dpi=300)
plt.plot(X, μ, 'C1', label='Mean Prediction')
plt.plot(X, y, 'C0', label='y = x\sin x$');
plt.errorbar(X_train, y_train_noisy, noise_std, linestyle='None', marker='.', markersize=15, c='k', label='Observation');
plt.fill_between(X.ravel(), μ-2*σ, μ+2*σ, alpha=0.25, color='C1', label='95% Confidence Interval'); 
plt.legend(loc='upper left');
plt.xlabel('$x$');
plt.ylabel('$y$');
```



<figure>
  <center>
  <img src="/assets/images/gpr_prediction_noisy_mean_0.svg" width="600">
   </center>
  <center>
    <figcaption> Figure 4. Predicted mean and its confidence level for observation with noise. The uncertainty at the observation data points is significant due to the noise.
    </figcaption>
  </center>
</figure>



### Summary

We presented a Gaussian Process regression implementation, focusing on the conditional distribution of a multivariate Gaussian and covariance matrix computation from predictors. In the upcoming report, we will explore the zero mean assumption for the data and the impact of large observation sample sizes. These investigations aim to enhance our understanding of Gaussian Process regression and identify potential areas for improvement.
