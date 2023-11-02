---
title: "Simple Derivation and Intuitive Understanding of Independence Test Using HSIC"
date: 2023-10-31
typora-root-url: ./..
---

In this article, we will derive the HSIC formula in a clear and straightforward manner. We will also explore how to estimate the statistical significance using bootstrap sampling and gain an intuitive understanding of why mapping data into a feature space is crucial for independence testing.



### Introduction

When dealing with random variables that are dependent on each other, it's not uncommon for them to exhibit no linear correlation. As illustrated in Figure 1, the linear correlation metric, such as the Pearson correlation coefficient, might fail to detect this dependence.



Random variables that are dependent on each other may have no linear correlation; an example is shown in Figure 1. Therefore, the linear correlation metric, like the Pearson correlation coefficient, cannot detect the dependence between the variables. 

<figure>
  <center>
  <img src="/assets/images/hsic_parabola.svg" width="600">
   </center>
  <center>
    <figcaption> Figure 1. Random variables <i>x</i> are <i>y</i> exhibit no linear correlation (Pearson correlation coefficient <i>r</i> = -0.019) despite being dependent.
    </figcaption>
  </center>
</figure>


To address this issue, Gretton et al. introduced the Hilbert-Schmidt Independence Criterion (HSIC) as a means to assess statistical dependence. HSIC involves mapping the data to a feature space using kernels, constructing a cross-covariance matrix, and ultimately quantifying the degree of dependence based on the Hilbert-Schmidt norm of this matrix.



### Hilbert-Schmidt Independence Criterion (HSIC) 



#### Overview Of How HSIC Works

The mathematical definition of independence is that two random variables, $\mathbf{x}$ and $\mathbf{y}$, are independent if and only if the covariance of any bounded functions of these variables, $f(\mathbf{x})$ and $g(\mathbf{y})$, is zero. This covariance is termed the cross-covariance and is given by:



$$
C_{xy} = \mathbf{E}_{x,y}[(f(\mathbf{x})-\overline{f(\mathbf{x}}))\otimes(g(\mathbf{y})-\overline{g(\mathbf{y}}))],
\label{eqn:cross-covariance}
$$



where 

- $\mathbf{E}_{x,y}$ denotes the expectation over the joint distribution of $\mathbf{x}$ and $\mathbf{y}$
-  $\overline{f(\mathbf{x})}=\mathbf{E}_x[f(\mathbf{x})]$
-   $\overline{g(\mathbf{y})}=\mathbf{E}_y[g(\mathbf{y})]$.



To conduct the independence test, we need to transform the data using functions $f$ and $g$. Kernel functions are suitable choices for $f$ and $g$ because the cross-covariance can be expressed through dot products of the transformed variables, simplifying computations by allowing us to apply the kernels to the original data directly.



The Hilbert-Schmidt norm (or Frobenius norm) of the cross-covariance is defined as as $\mathrm{HSIC}$:



$$
\begin{array}{cl}
\mathrm{HSIC} &=& \vert C_{xy} \vert_{\mathrm{HS}}^2\\
& = & \sum_{ij} C_{ij}^2 \\
& = & \mathrm{tr}(C_{xy}^T C_{xy})
\end{array}.
\label{eqn:hsic}
$$



Here, $C_{ij}$ is the element at the $i$-th row and $j$-th column of $C_{xy}$.  $x$ and $y$ are considered independent when $\mathrm{HSIC} = \vert C_{xy}\vert_{\mathrm{HS}}^2 = 0$.




The further $\mathrm{HSIC}$ deviates from zero, the more likely $x$ and $y$ are dependent. To quantify this likelihood, one can perform random permutation of the data $x$ and $y$ and compare the $\mathrm{HSIC}$ of the non-permuted data with the distribution of $\mathrm{HSIC}$ values from the permuted data. This comparison can be expressed as a $p-$value, where a smaller $p-$value indicates a higher likelihood of dependence.



#### HSIC Estimate Formula From Observations

Let's assume two random variables, $x$ and $y$, are mapped to a feature space using functions $\psi$ and $\phi$:



$$
\begin{array}{c}
\mathbf{x} &\rightarrow& \psi(\mathbf{x})\\
\mathbf{y} &\rightarrow& \phi(\mathbf{y})
\end{array}. \notag
$$



The observations consist of $N$ independent data pairs $\{ (x_1, y_1), \ldots, (x_N, y_N)\}$.  We can define matrices $\Psi$ and $\Phi$ as follows:



$$
\Psi = \left(
\begin{array}{c}
\psi(x_1) \\
\vdots \\
\psi(x_k)\\
 \vdots  \\
\psi(x_N) 
\end{array}
\right),  \notag
$$


and



$$
\Phi = \left(
\begin{array}{c}
\phi(y_1) \\
\vdots \\
\phi(y_k)\\
 \vdots  \\
\phi(y_N) 
\end{array}
\right),  \notag
$$



Each row in $\Psi$ and $\Phi$ corresponds to a mapped vector in the feature space. 

The means of $\Psi$ and $\Phi$ are represented as $\overline{\Psi} = A \Psi$ and $\overline{\Phi} = A \Phi$, where $A$ is an $N\times N$ square matrix with all its elements set to $1/N$.

The cross-covariance matrix is then given by: 



$$
\begin{array}{cl}
C_{xy} &=& \frac{1}{N}(\Psi - \overline{\Psi})^T (\Phi - \overline{\Phi}) \\
& = & \frac{1}{N} \Psi^T(I-A)(I-A)\Phi \\
& = & \frac{1}{N} \Psi^T H H \Phi
\end{array} 
\label{eqn:c_matrix}
$$



where $I$ is the identity matrix of size $N$ and $H = I - A$. 




Plug Equation ($\ref{eqn:c_matrix}$) into the $\mathrm{HSIC}$ formula ($\ref{eqn:hsic}$), we get



$$
\begin{array}{cl}
\mathrm{HSIC}=\vert C_{xy} \vert_{\mathrm{HS}}^2  &=& \frac{1}{N^2}\mathrm{tr}(\Phi^T H H \Psi \Psi^THH\Phi)\\
&=& \frac{1}{N^2}\mathrm{tr}(H\Phi\Phi^T H H \Psi \Psi^TH)\\
&=& \frac{1}{N^2}\mathrm{tr}(H K_y HHK_xH) \\
& = & \frac{1}{N^2}\mathrm{tr}(H K_x HHK_yH)
\end{array},
\label{eqn:hsic_estimate}
$$



where $K_x = \Psi \Psi^T$ and $K_y =\Phi \Phi^T$. $K_x$ and $K_y$ are the Gram matrices of the mapped features for $\mathbf{x}$ and $\mathbf{y}$, respectively. In addition, in the second step in the equation above, the following property of matrix trace is used: $\mathrm{tr}(P Q)=\mathrm{tr}(Q P)$, where $P = \Phi^T H H \Psi \Psi^TH$ and $Q=H\Phi$. Again, in the last step, the same trace property is used. 

Note that $HK_xH$ is the centered kernel $K_x$ and $HK_yH$ is the centered kernel $K_y$.



#### Statistical Significance

To determine whether the $\mathrm{HSIC}$ estimated using the formula above is statistically different from 0, we can employ bootstrap sampling. This involves randomly sampling the observations $\left\{x_1, ..., x_N\right\}$ and $\left\{y_1, ..., y_N\right\}$ with replacement for each pair of sampled data and calculating the $\mathrm{HSIC}$ for each permuted sample. As the randomly sampled $x$ and $y$ are independent, the distribution of the bootstrapped $\mathrm{HSIC}$ represents the distribution when the random variables are independent. The $\mathrm{HSIC}$ value of the original observation is then compared to this distribution. If it is significantly larger than the distribution, it suggests that $x$ and $y$ are dependent. A $p-$value can be defined as the proportion of the permuted samples whose $\mathrm{HSIC}$ is greater than that of the original data.



### A Numerical Example

Let's illustrate $\mathrm{HSIC}$ with a numerical example. First, we generate two variables with an intrinsic quadratic relationship:

```python
import numpy as np
n = 1000
x =1- 2* np.random.rand(n,1)
y = xx**2 + 0.1*np.random.randn(n,1)
```

The scatter plot of $x$ vs. $y$ is shown in Figure 1, where the Pearson correlation coefficient between $x$ and $y$ is very small ($r = -0.019$), indicating that linear correlation cannot detect the dependence between them.

We can calculate the $\mathrm{HSIC}$ as follows:


```python
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel

def hsic(Kx, Ky):
""" Calculate HSIC of two kernels
    Inputs:
        Kx: kernel of variable x
        Ky: kernel of variable y
    Output:
        HSIC
"""
    N = Kx.shape[0]
    H = np.identity(N) - 1/N*np.ones((N,N))
    Kxp = H @ Kx @ H
    Kyp = H @ Ky @ H
    result = np.sum( np.diag( Kxp @ Kyp ) )
    return result


```

The $\mathrm{HSIC}$ with radial basis function (RBF) kernel is

```python
hsic(rbf_kernel(x), rbf_kernel(y))
10750.84655613255
```

Next, we perform bootstrap sampling to estimate the distribution of $\mathrm{HSIC}$: 

```python
# bootstrap for HSIC distribution
n_boot = 1000
result_rbf = []
N = len(x)
for i in range(n_boot):
    idx_x = np.random.choice(np.arange(N), size=N)
    idx_y = np.random.choice(np.arange(N), size=N)
    Kx = rbf_kernel(x[idx_x])
    Ky = rbf_kernel(y[idx_y])
    result_rbf.append(hsic(Kx, Ky))  
    
p_value = np.mean(
  np.array(result_rbf) - hsic(rbf_kernel(x), rbf_kernel(y))
)
```

The $p-$value is calculated as the proportion of bootstrap samples whose $\mathrm{HSIC}$ value is greater than the $\mathrm{HSIC}$ of the original data. A smaller $p-$value indicates a higher likelihood of dependence.

Figure 2 displays the distribution of $\mathrm{HSIC}$ for the randomly permuted samples (blue bars) and the $\mathrm{HSIC}$ of the original data (red vertical line). The larger the difference from zero, the smaller the $p-$value, indicating a greater likelihood of dependence.

<figure>
  <center>
  <img src="/assets/images/hsic_rbf_bootstrap_dist.svg" width="800">
   </center>
  <center>
    <figcaption> Figure 2.  The HSIC of the original data is significantly larger than the HSIC of the randomly permutated data, indicating dependence between <i>x</i> and <i>y</i>. 
    </figcaption>
  </center>
</figure>



Similarly, we can apply the same analysis to a pair of variables that are independent of each other. The $\mathrm{HSIC}$ and bootstrap calculations indeed show that $x$ and $y$ are independent, as demonstrated in Figure 3.

```python
x = np.random.randn(500,1)
y = np.random.randn(500,1)
```

<figure>
  <center>
  <img src="/assets/images/hsic_nocorr_bootstrap_dist.svg" width="800">
   </center>
  <center>
    <figcaption> Figure 3. HSIC and its values for the randomly permutated samples show that <i>x</i> and <i>y</i> are independent.
    </figcaption>
  </center>
</figure>



### Why HSIC Works

The key to the HSIC method's success lies in the transformation of data into a higher-dimensional feature space using the kernel operator. In this feature space, data is represented by vectors of higher dimensions, which can even be infinite, as in the case of the RBF kernel. These feature space components are polynomial functions of the original data of various degrees (see this [previous blog post](<https://kezhaozhang.github.io/2023/04/26/kernel-transformation.html>)).

As Figure 4 illustrates using dependent but not linearly correlated data, the correlation between the transformed data with polynomial functions of various degrees becomes apparent. In this specific case, the mapped data exhibits strong correlations with even degrees. Therefore, by mapping with polynomial kernels of even degrees, the dependence between $x$ and $y$ can be detected.

<figure>
  <center>
  <img src="/assets/images/hsic_x^d_vs_y^d.svg" width="850">
   </center>
  <center>
    <figcaption> Figure 4. The correlation between polynomial mapping of <i>x</i> and <i>y</i> for various degrees.
    </figcaption>
  </center>
</figure>



Kernels can be conceptually likened to the Taylor expansion of the original data, effectively breaking down the nonlinear relationship between $x$ and $y$ into correlations among these polynomial components. Consequently, the cross-covariance matrix will contain non-zero elements, resulting in an $\mathrm{HSIC}$ distinct from zero, a clear indicator of statistical dependence.





### References

Gretton, A., Herbrich, R., Smola A., Bousquet, O. and Schölkopf, B. (2005). Kernel Methods for Measuring Independence. Journal of Machine Learning Research, 2005, vol 6 no 70, 2075--2129. http://jmlr.org/papers/v6/gretton05a.html



Gretton, A., Bousquet, O., Smola, A., Schölkopf, B. (2005). Measuring Statistical Dependence with Hilbert-Schmidt Norms. In: Jain, S., Simon, H.U., Tomita, E. (eds) Algorithmic Learning Theory. ALT 2005. Lecture Notes in Computer Science, vol 3734. Springer, Berlin, Heidelberg. https://doi.org/10.1007/11564089_7
