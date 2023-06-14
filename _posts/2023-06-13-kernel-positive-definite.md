---
title: "Positive Definiteness of Kernels"
date: 2023-06-13
---

This post summarizes the proof demonstrating the positive definiteness of the multivariate squared exponential kernel (radial basis function) and exponential kernel. The proofs primarily rely on sources such as (Wendland 2004) and stackexchange.com. Additionally, Python `numpy` commands are included for numerically testing the positive definiteness of a matrix.



### Definition of Positive Definiteness

According to (Wendland 2004), a continuous function $\Phi$: $\mathbb{R}^d \rightarrow \mathbb{C}$ is considered positive definite if  the quadratic form 


$$
\sum_{i=1}^N\sum_{j=1}^N\alpha_i \overline{\alpha_j}\Phi(\mathbf{x}_i - \mathbf{x}_k) \label{eqn:quadratic}
$$
is positive for all $N\in \mathbb{N}$, all sets of pairwise distinct centers $X = \{\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_N\} \subseteq \mathbb{R}^d$, and all $\alpha \in \mathbb{C}^N$,



The function $\Phi(\mathbf{x})$ can be expressed in terms of its Fourier transform:


$$
\Phi(\mathbf{x}) = (2\pi)^{-d/2}\int_{\mathbb{R}^d}\widehat{\Phi}(\boldsymbol{\omega})e^{i\mathbf{x}^T \mathbf{\omega}}d\boldsymbol{\omega}.\notag
$$
The Fourier  transform is defined as 


$$
\widehat{\Phi}(\boldsymbol{\omega})= (2\pi)^{-d/2}\int_{\mathbb{R}^d}{\Phi}(\mathbf{x})e^{-i\mathbf{x}^T\boldsymbol{\omega}}d\mathbf{x}. \notag
$$




The quadratic form in Equation ($\ref{eqn:quadratic}$) becomes


$$
\begin{align*}
& (2\pi)^{-d/2}\sum_{i,j=1}^N \alpha_i\overline{\alpha_j}\int_{\mathbb{R}^d}\widehat{\Phi}(\boldsymbol{\omega})e^{i\boldsymbol{\omega}^T(\mathbf{x}_i-\mathbf{x}_j)}d\boldsymbol{\omega}\\
= & (2\pi)^{-d/2}\int_{\mathbb{R}^d}\widehat{\Phi}(\boldsymbol{\omega})\left\vert \sum_{i=1}^N \alpha_i e^{i\mathbf{x}^T\boldsymbol{\omega}}\right\vert^2 d\boldsymbol{\omega}
\end{align*}.
$$



Thus, if $\widehat{\Phi} > 0$, the quadratic form is positive, and $\Phi$ is positive definite.

Next, we proceed to calculate the Fourier transform of the squared exponential kernel and the exponential kernel to establish their positive definiteness.



### Squared Exponential Kernel $e^{-\alpha \mid \mathbf{x}\mid^2}$

The Fourier transform of the kernel is given by



$$
\begin{align*}
\widehat{\Phi}(\boldsymbol{\omega})&= (2\pi)^{-d/2}\int_{\mathbb{R}^d} e^{-\alpha \mathbf{x}^T\mathbf{x}}e^{-i\mathbf{x}^T\boldsymbol{\omega}}d\boldsymbol{\omega}\\
& = (2\pi)^{-d/2} \prod_{i=1}^d \int_{\mathbb{R}}e^{-\alpha x_i^2}e^{-i x_i \omega_i}d\omega_i\\
& = (2 \alpha)^{-d/2} \exp\left({-\frac{\vert \boldsymbol{\omega}\vert^2}{4\alpha}}\right).
\end{align*}
$$



Clearly $\widehat{\Phi}(\boldsymbol{\omega})>0$. Therefore, the squared exponential kernel is positive definite.




### Exponential Kernel $e^{-\mid \mathbf{x}\mid}$

To calculate the Fourier transform of the multivariate kernel $e^{-\mid \mathbf{x}\mid}$, we can use the following equation (source: [Math StackExchange](https://math.stackexchange.com/questions/2569910/fourier-transform-in-mathbbrn-of-e-x)):



$$
e^{-a} = \frac{1}{\sqrt{\pi}}\int_{-\infty}^\infty \exp\left(-t^2-\frac{a^2}{4 t^2}\right) dt.
$$



Let $a = \vert \mathbf{x}\vert$, and the Fourier transform of the kernel is

$$
\begin{align*}
\widehat{\Phi}(\boldsymbol{\omega})&= (2\pi)^{-d/2}\int_{\mathbb{R}^d} e^{-\vert \mathbf{x}\vert}e^{-i\mathbf{x}^T\boldsymbol{\omega}}d\boldsymbol{\omega}\\
& = (2\pi)^{-d/2}  \int_{\mathbb{R}^d}\frac{1}{\sqrt{\pi}}\int_{-\infty}^{\infty}\exp\left(-t^2-\frac{\mathbf{x}^T\mathbf{x}}{4 t^2}\right)e^{-i \mathbf{x}^T \boldsymbol{\omega}}d\boldsymbol{\omega}\\
& = \frac{1}{\sqrt{\pi}}\int_{-\infty}^{\infty}e^{-t^2}\left[(2\pi)^{-d/2}\int_{\mathbb{R}^d}\exp\left(-\frac{\mathbf{x}^T\mathbf{x}}{4 t^2}\right)e^{-i \mathbf{x}^T \boldsymbol{\omega}}d\boldsymbol{\omega}\right]d t\\
& = \frac{1}{\sqrt{\pi}}\int_{-\infty}^\infty (2 t^2)^{d/2} e^{-t^2 -\vert \boldsymbol{\omega}\vert^2 t^2} dt\\
& = \frac{2^{d/2}}{\sqrt{\pi}}\frac{\Gamma(\frac{d+1}{2})}{\left(1+\vert\boldsymbol{\omega}\vert^2\right)^{\frac{d+1}{2}}}.
\end{align*}
$$


Clearly $\widehat{\Phi}(\boldsymbol{\omega})$ is positive for all $\boldsymbol{\omega}$.  Therefore the exponenpositivernel is postive definite.



### Numerical Test

Due to floating-point operations, the kernel may not be strictly positive definite because some of the eigenvalues of the covariance matrix produced by the kernel may be too close to zero. Here are some `numpy` commands to check the positive definiteness of the matrix:

```python
import numpy as np
from numpy.linalg import matrix_rank
from scipy.linalg import issymmetric, eigvals
from sklearn.gaussian_process.kernels import RBF
# generate some random number
rng = np.random.default_rng(42)
x = rng.random(size=20)
# covariance matrix using the squared exponential kernel
S = RBF(0.25)(x.reshape(-1, 1))
```

First, check if the matrix is symmetric

```python
issymmetric(S)
```

```
True
```

The check the eigenvalues:

```python
eigvals(S)
```

```
array([ 1.13985838e+01+0.00000000e+00j,  5.40354181e+00+0.00000000e+00j,
        2.37962291e+00+0.00000000e+00j,  6.64898522e-01+0.00000000e+00j,
        1.23933967e-01+0.00000000e+00j,  2.64244831e-02+0.00000000e+00j,
        2.55016837e-03+0.00000000e+00j,  3.97544299e-04+0.00000000e+00j,
        4.32255682e-05+0.00000000e+00j,  3.39255202e-06+0.00000000e+00j,
        1.97259255e-07+0.00000000e+00j,  1.72130981e-08+0.00000000e+00j,
        6.45856879e-10+0.00000000e+00j,  6.26311180e-12+0.00000000e+00j,
        8.44860574e-14+0.00000000e+00j,  5.50899273e-16+0.00000000e+00j,
        1.61835986e-16+0.00000000e+00j, -2.91921847e-17+1.42920245e-16j,
       -2.91921847e-17-1.42920245e-16j, -5.98295830e-17+0.00000000e+00j])
```

The imaginary part of the eigenvalues are effectively $0$, so we can discard it.

```python
e = np.real(eigvals(S)) 
```

```
array([6.57394925e+00, 4.17360323e+00, 2.72701226e+00, 3.08394175e+00,
       1.53185732e+00, 8.94197525e-01, 5.23211869e-01, 2.61797136e-01,
       1.22453532e-01, 7.20876241e-02, 2.89583165e-02, 5.21265849e-03,
       1.32822957e-03, 2.96374850e-04, 8.87602732e-05, 3.96057297e-06,
       2.04120206e-07, 3.23772029e-09, 1.24510109e-12, 2.42094613e-10])
```

Some eigenvalues are very small. The closeness to $0$ can be checked for a given tolerance, and $8$ out of $20$ of the eigenvalues are close to $0$ for the given tolerance.

```python
np.isclose(e, 0, rtol=1e-5, atol=1e-8)
```

```
array([False, False, False, False, False, False, False, False, False,
       False, False, False,  True,  True,  True,  True,  True,  True,
        True,  True])
```

Similarly, matrix rank gives the same result.

```python
matrix_rank(S,tol=1e-8)
```

```
12
```

The covariance matrix is not positive definite because of the floating point computations.



### References

Wendland, H. (2004). Scattered Data Approximation. Cambridge University Press

Fourier transform in $\mathbb{R }^n$ of $e^{-\mid \mid x\mid\mid}$. https://math.stackexchange.com/questions/2569910/fourier-transform-in-mathbbrn-of-e-x