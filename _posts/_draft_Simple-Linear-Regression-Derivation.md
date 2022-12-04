---
title: Simple Linear Regression Derivation
---



This note derives the linear regression coefficient, residual mean square error, and R square as functions of variance and covariance of the predictor and the predicted.



## Univariate Case

We have the linear model
$$
y = a + b x,
$$
where $x$ is the predictor, $y$ is the predicted;  $a$ and $b$ are the coefficients to be determined.



In an ordinary linear regression, the 
$$
Y = a + b X +\epsilon. \label{eqn:ols}
$$
The residual $\epsilon$ is orthogonal to $X$, i.e., their covariance is zero.
$$
\mathrm{cov}(X,\epsilon)=0
$$
To derive $a$ and $b$, first, take the mean of the Equation ($\ref{eqn:ols}$):
$$
E[Y] = a + b E[X], \label{eqn:mean}
$$
where $E$ stands for the mean.

Next, calculate the covariance of $X$ and $Y$:
$$
\mathrm{cov}(Y, X)  =\mathrm{cov}(a + b X +\epsilon, X)= b \mathrm{var}(X) 
$$
Therefore
$$
b = \frac{\mathrm{cov(Y,X)}}{\mathrm{var}(X)}=\frac{\sigma_{XY}}{\sigma^2_X}.
$$
From Equation ($\ref{eqn:mean}$), we have
$$
a = E[Y] - \frac{\sigma_{XY}}{\sigma^2_X} E[X].
$$


To estimate the variance of the residual $\epsilon$, we calculate the variance of $Y$ using Equation ($\ref{eqn:ols}$):
$$
\mathrm{var}(Y)=\mathrm{var}(a + b X + \epsilon)=b^2\mathrm{var}(X) +\sigma^2, \label{eqn:varY}
$$
where $\sigma^2 = \mathrm{var}(\epsilon)$.

Hence
$$
\sigma^2_Y = b^2\sigma^2_X + \sigma^2=\frac{\sigma^2_{XY}}{\sigma^2_X}+\sigma^2.
$$

$$
\sigma^2 = \sigma^2_Y - \frac{\sigma^2_{XY}}{\sigma^2_X}.
$$

By definition, the R square of the regression is
$$
R^2 = 1-\frac{\sigma^2}{\sigma^2_Y}=\frac{\sigma^2_{XY}}{\sigma^2_X\sigma^2_Y}.
$$
The error term $\epsilon$ in Equation ($\ref{eqn:ols}$) plays an important role in the analysis above. It presents the assumption that the predictor $X$ has no error whereas the predicted variable $Y$ has error. This breaks the asymmetry between $X$ and $Y$ in the regression coefficient calculation. Because without the $\epsilon$ term, 
$$
\sigma^2_Y= \mathrm{var}(Y)=\mathrm{cov}(Y, a + b X)=b \sigma^2_{XY}.
$$
 And  hence
$$
b=\frac{\sigma^2_Y}{\sigma^2_{XY}}\neq \frac{\sigma^2_{XY}}{\sigma^2_X}.
$$
$b=\frac{\sigma^2_Y}{\sigma^2_{XY}}$ is the inverse of the coefficient in the model $X = a + b Y + \epsilon$, where $Y$ is assumed to be without error and $X$ with error.



## Multivariate Case



~~~mermaid
```mermaid
  graph TD;
      A-->B;
      A-->C;
      B-->D;
      C-->D;
```
~~~