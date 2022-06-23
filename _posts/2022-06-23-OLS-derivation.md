---
title: "Derivation of Linear Regression Coefficients and Their Variation with Minimal Matrix Algebra"
---



This is a simple calculation of linear regression coefficients and their variances using covariance and variance with minimal need for matrix algebra.



## Single Predictor

Suppose a linear relationship between  $y$ and $x$:


$$
y = \beta_0 + \beta_1x +\epsilon,
\label{eqn:linear}
$$


where $\epsilon$ is the noise.



### Estimate of $\beta_0$ and $\beta_1$

Calculate the covariance of $y$ and $x$, we have


$$
\begin{array}{cc}
\mathrm{Cov}(y,x) & =\mathrm{Cov(x, \beta_1 x)} + \mathrm{Cov}(\epsilon, x) \\
 & = \beta_1 \mathrm{Var}(x) + \mathrm{Cov}(\epsilon, x)
\end{array} \notag
$$


If with the **homoscedasticity** assumption, i.e.,  $x$ and noise $\epsilon$ are independent, 



$$
\mathrm{Cov}(\epsilon, x) =0.
$$


Then

$$
\mathrm{Cov}(y,x)  =\beta_1 \mathrm{Var}(x). \notag
$$



Hence the estimated $\beta_1$ for the observed data is



$$
\boxed{
\hat{\beta_1} = \frac{\mathrm{Cov}(y,x)}{\mathrm{Var}(x)}
},
\label{eqn:beta1}
$$


where the hat symbol indicates an estimate from observed data.

In Equation ($\ref{eqn:linear}$), take the expected value or the mean of both sides of the equation, and we have



$$
\langle{y}\rangle = \beta_0 +\beta_1 \langle{x}\rangle, \notag
$$



where $\langle y\rangle$ and $\langle{x}\rangle$ are the expected values or mean of $y$ and $x$, respectively; and $\langle{\epsilon}\rangle=0$.




$$
\boxed{
\hat{\beta_0} = \langle{y}\rangle-\hat{\beta_1} \langle{x}\rangle \label{eqn:beta0}
}.
$$


### Variance of $\hat{\beta_0}$ and $\hat{\beta_1}$

For given observed data $x$ and $y$, the variation of the estimated coefficients is due to the noise $\epsilon$.

From Equation ($\ref{eqn:beta1}$):



$$
\mathrm{Var}(\hat{\beta_1}) = \mathrm{Var}\left(\frac{\mathrm{Cov}(y, x)}{\mathrm{Var}(x)}\right) = \frac{1}{\mathrm{Var}(x)^2}\mathrm{Var}\left(\mathrm{Cov}(y,x)\right),  \label{eqn:beta1_var_1}
$$



where



$$
\begin{align} \notag
\mathrm{Var}\left(\mathrm{Cov}(y,x)\right) &= \mathrm{Var}(\mathrm{Cov}\left(\beta_0+\beta_1 x+\epsilon, x)\right) \\ \notag
& =  \mathrm{Var}\left( \beta_1 \mathrm{Var}(x)+\mathrm{Cov}(\epsilon, x)\right)\\\notag
&= \mathrm{Var}\left(\mathrm{Cov}(\epsilon, x)\right)
\end{align}
$$



In the derivation above, we have used the following:

- $\mathrm{Cov}(\beta_0, x)=0$: covariance of a constant and $x$ is zero,
- $\mathrm{Var}\left(\beta_1\mathrm{Var}(x)\right)=0$: variance of a constant is zero.



Suppose there are $n$ values of $x$: $x_1, x_2, \ldots, x_n$ and corresponding noise values $\epsilon_1, \epsilon_2, \ldots, \epsilon_n$, then



$$
\mathrm{Cov}(\epsilon, x)=\frac{1}{n}\sum_{i=1}^n \epsilon_i (x_i-\bar{x}). \notag
$$



Further, assume that $\epsilon_i$ are **i.i.d. random variables**  and $\mathrm{Var}(\epsilon_i)=\sigma^2$, then



$$
\begin{array}{cl}
\mathrm{Var}\left(\mathrm{Cov}(\epsilon, x)\right) &=& 
\frac{1}{n^2}\sum_{i=1}^n \mathrm{Var}\left(\epsilon_i (x_i -\bar{x}\right))\\
&= & \frac{1}{n^2}\sum_{i=1}^n (x_i-\bar{x})^2\mathrm{Var}(\epsilon_i) \\
&=& \frac{\sigma^2}{n^2}\sum_{i=1}^n (x_i-\bar{x})^2\\
&=& \frac{\sigma^2}{n}\mathrm{Var}(x)
\end{array}
\label{eqn:beta1_var_2}
$$


Plug Equation($\ref{eqn:beta1_var_2}$) into Equation($\ref{eqn:beta1_var_1}$)



$$
\boxed{
\begin{align*}\notag
\mathrm{Var}(\hat{\beta_1}) &= \mathrm{Var}\left(\frac{\mathrm{Cov}(y, x)}{\mathrm{Var}(x)}\right) = \frac{1}{\mathrm{Var}(x)^2}\mathrm{Var}\left(\mathrm{Cov}(y,x)\right) \\ 
&= \frac{\sigma^2}{n \mathrm{Var}(x)} 
\end{align*}
}.
\label{eqn:beta1_var}
$$



And



$$
\begin{align*}
\mathrm{Var}(\hat{\beta_0}) &= \mathrm{Var}\left(\langle{y}\rangle-\hat{\beta_1} \langle{x}\rangle \right) \\
& = \mathrm{Var}(\langle{y}\rangle) + \langle{x}\rangle^2 \mathrm{Var}(\hat{\beta_1}).
\end{align*}
$$



Because $\mathrm{Var}(\langle{y}\rangle)= \frac{1}{n}\sigma^2$, 



$$
\boxed{
\mathrm{Var}(\hat{\beta_0})=\frac{\sigma^2}{n} + \frac{\sigma^2 \langle{x}\rangle^2}{n \mathrm{Var}(x)} = \frac{\sigma^2 \langle{x^2}\rangle}{n}
},\label{eqn:beta0_var}
$$



where $\mathrm{Var}(x) = \langle{x^2}\rangle- \langle{x}\rangle^2$ is used in the final step of the equation. 



## Multivariate Predictor

The linear relationship between the predicted $y$ and the predictor $\vec{x}$ is



$$
y = \beta_0 + \vec{x}\beta +\epsilon,
$$



where $\vec{x}$ is a $p$-dimensional multivariate variable: $\vec{x} = \left[ x_1\ldots,x_i\ldots,x_p\right]^T$ where each element $x_i$ is a variable that has $n$ values in measurement. $\beta$ is also a $p$-dimensional vector and $\beta_0$ is a scalar.

### Estimate

Calculating the covariance of $y$ and $\vec{x}$, we have


$$
\mathrm{Cov}(y,\vec{x}) = \mathrm{Cov}(\vec{x}, \vec{x})\beta+\mathrm{Cov}(\epsilon, \vec{x}) = \mathrm{Cov}(\vec{x}, \vec{x})\beta,
$$



where in the last step, homoscedasticity is assumed: $\mathrm{Cov}(\epsilon, \vec{x}=0)$.

Then 



$$
\boxed{
\hat{\beta} = A^{-1} \mathrm{Cov}(y, \vec{x})
},
$$



where $A$ is the covariance matrix of $\vec{x}$:



$$
A  = \mathrm{Cov}(\vec{x}, \vec{x}).
$$



The estimate of $\beta_0$ is



$$
\boxed{
\hat{\beta_0} = \langle y\rangle - \langle \vec{x} \rangle \hat{\beta}
}.
$$



### Variance



$$
\begin{align*}
\mathrm{Var}(\hat{\beta}) &=\mathrm{Var}(A^{-1}\mathrm{Cov}\left(y, \vec{x})\right)\\
& = \mathrm{Var}(A^{-1}\mathrm{Cov}\left(\beta_0+\vec{x}\beta+\epsilon, \vec{x})\right)\\
& =  \mathrm{Var}\left(A^{-1}\mathrm{Cov}(\epsilon, \vec{x})\right)\\
& = (A^{-1})^{\circ 2}\mathrm{Var}\left(\left\langle \epsilon (\vec{x}-\langle\vec{x}\rangle)\right\rangle\right),
\end{align*}
$$



where superscript $\circ 2$ denotes element-wise squared (see [Hadamard Product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices))).

For a component of $\vec{x}$,  $x_i$, let $n$ be the number of measurements, then 



$$
\begin{align*}
\mathrm{Var}\left(\left\langle \epsilon (x_i -\langle x_i\rangle)\right\rangle\right) &= \mathrm{Var}\left(\frac{1}{n}\sum_{j=1}^n \epsilon_j \left(x_{i,j}-\langle x_i \rangle\right)\right) \\
& = \frac{1}{n^2} \sum_{j=1}^n (x_{i,j}-\langle x_i\rangle)^2\mathrm{Var}(\epsilon_j)\\
& = \frac{\sigma^2}{n}\mathrm{Var}(x_i).
\end{align*}
$$



Therefore 


$$
\boxed{
\mathrm{Var}(\hat{\beta})=A^{-1}\mathrm{Var}\left(\mathrm{Cov}(\epsilon, \vec{x})\right) = \frac{\sigma^2}{n} (A^{-1})^{\circ 2}\mathrm{Var}(\vec{x})
}.
$$



And



$$
\boxed{
\begin{align*}
\mathrm{Var}(\hat{\beta_0})&=\mathrm{Var}(\langle y\rangle) +\mathrm{Var}(\langle\vec{x}\rangle \hat{\beta})\\
& = \frac{\sigma^2}{n} + \frac{\sigma^2}{n} (A^{-1}\langle \vec{x}\rangle)^{\circ 2}\mathrm{Var}(\vec{x})
\end{align*}
}
$$
