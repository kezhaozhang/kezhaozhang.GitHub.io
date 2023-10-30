---
title: "Understanding Kernel Principal Component Analysis (Kernel PCA)"
date: 2023-10-26
typora-root-url: ./..
---

Kernel Principal Component Analysis (Kernel PCA) is a powerful technique used in machine learning for dimensionality reduction. It allows us to perform principal component analysis on data that has been nonlinearly mapped to a higher-dimensional feature space. This article will provide a step-by-step derivation of the Kernel PCA formula, followed by an illustrative example to showcase its practical application. We will also compare our results with explicit mapping in feature space and the Kernel PCA implementation in Scikit-Learn.



### Derivation of Kernel PCA

Kernel PCA is a variant of traditional Principal Component Analysis (PCA) that operates in a feature space created by a nonlinear mapping of the original data. This mapping transforms data vectors, denoted as $\mathbf{x}_i$, into vectors in the feature space $\mathbf{\phi}(\mathbf{x}_i)$, where $i=1,2,\ldots, N$, and $N$ is the number of data points.  



The crucial relationship between the mapped vectors in the feature space is represented by the dot product, which is equal to the kernel function, $f$,  applied to the original vectors:


$$
\langle \phi(\mathbf{x}_i), \phi(\mathbf{x}_j)\rangle = f(\mathbf{x}_i, \mathbf{x}_j)
\label{eqn_dot_prod}.
$$


We can assemble these mapped vectors into a matrix $\psi$, where each row corresponds to a mapped vector in the feature space.


$$
\psi = \left(
\begin{array}{c}
\phi(\mathbf{x}_1) \\
\vdots \\
\phi(\mathbf{x}_k)\\
 \vdots  \\
\phi(\mathbf{x}_N) 
\end{array}
\right). \notag
$$




The mean of this matrix $\psi$ is calculated as:
$$
\bar{\psi} = A \psi, \notag
$$

where $A$ is a square matrix of size $N$, with all its elements set to $1/N$. The covariance matrix in the feature space is derived as follows:



$$
\begin{array}{ccl}
C &=& \frac{1}{N}(\psi -\bar{\psi})^T (\psi-\bar{\psi})\\
 &=& \frac{1}{N} \psi^T (I - A) (I - A)\psi
 \end{array} \notag
$$



where $I$ is an identity matrix of size $N$.



Principal Component Analysis in the feature space seeks the eigenvector $V$ of the covariance matrix $C$:



$$
C V = \lambda V. \label{eqn_eig_eqn}
$$



By expressing $V$ as a linear combination of the centered mapped data, we obtain:



$$
V = (\psi - \bar{\psi})^T \alpha=\psi^T(I-A)\alpha, \label{eqn_eigv}
$$


where $\alpha$ is a vector of size $N$.



Now, if we plug Equation ($\ref{eqn_eigv}$) into the eigenvector equation ($\ref{eqn_eig_eqn}$), we have



$$
\frac{1}{N} \psi^T(I-A)(I-A)\psi\psi^T (I-A)\alpha = \lambda \psi^T(I-A)\alpha. 
\notag
$$



By multiplying both sides by $(I-A)\psi$, we arrive at:



$$
\widetilde{K}\widetilde{K} \alpha =  \widetilde{K} \lambda N \alpha,
\label{eqn_k_tilde_eig}
$$


where



$$
\begin{array}{ccl}
\widetilde{K} &=& (\psi - \bar{\psi})(\psi -\bar{\psi})^T \\
& = & (I-A) \psi \psi^T (I-A) \\
&=& (I-A) K (I-A) \\
\end{array}.
\notag
$$

And



$$
K = \psi \psi^T. \notag
$$


Here $K$ is the the dot product of the mapped vectors in the feature space, as defined by $K_{ij} = f(\phi(\mathbf{x})_i, \phi(\mathbf{x}_j)$. We can calculate $K$ in the original space using the kernel function. 



From Equation ($\ref{eqn_k_tilde_eig}$), $\alpha$ becomes the solution for 


$$
\widetilde{K} \alpha = \lambda' \alpha
\label{eqn_eig}
$$


where $\lambda' = N\lambda$. To normalize $V$  for the $k$-th eigenvector $V_k$, we require:



$$
V_k^T V_k =\alpha_k^T (I-A)\psi \psi^T (I-A)\alpha_k=\alpha_k^T \widetilde{K}\alpha_k = 1.\notag
$$



As a result of Equation ($\ref{eqn_eig}$), we scale $\alpha$ as $\alpha' = \alpha/\sqrt{\lambda'}$. The $k$-th principal component corresponding to $\alpha_k$ for centered mapped data $\psi - \bar{\psi}$  is given by:



$$
\text{PC}_k = (\psi -\bar{\psi}) V_k =(I-A)\psi \psi^T (I -A)\alpha'_k = \widetilde{K}\alpha'_k=\sqrt{\lambda'_k}\alpha_k. \notag
$$


### An Example

In this example, we will demonstrate the use of Kernel PCA with a polynomial kernel. We'll also compare our results with explicit mapping in the feature space and the Kernel PCA implementation in Scikit-Learn.

#### Data

First, 100 data points on two concentric circles:

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
X, group = make_circles(n_samples=100, factor=0.5, noise=0.05, random_state=123)

plt.scatter(X[:,0],X[:,1], c=[f"C{i}" for i in y], s=50);
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.axis('equal')
```



<figure>
  <center>
  <img src="/assets/images/kpca_circle.svg" width="600">
   </center>
  <center>
    <figcaption> Figure 1. Randomly generated data points on two concentric circles.
    </figcaption>
  </center>
</figure>

#### Kernel 

We use a polynomial kernel of degree 4:
$$
f(\mathbf{x}, \mathbf{y}) = (\mathbf{x}\cdot\mathbf{y})^4. \notag
$$


#### Kernel PCA without Explicit Mapping

We calculate the first four principal components using the polynomial kernel. Figure 2 shows the relationship among these components. Notably, the third principal component, $\text{PC}3$, effectively separates the data in the two concentric circles.

```python
import numpy as np
from scipy.linalg import eigh
from sklearn.metrics.pairwise import polynomial_kernel

N = X.shape[0]
K = polynomial_kernel(X, degree=4, gamma=1, coef0=0) 
A = np.ones_like(K)/N # \bar{phi} = A phi
I = np.identity(N) # idenity matrix
K_tilde = (I - A) @ K @ (I - A)
lam, alpha = eigh(K_tilde, subset_by_index=[N-4, N-1]) # 4 largest eigenvalues 
                                                       # and corresponding eigenvectors
kpc = alpha*np.sqrt(lam) #top 4 principal components of kernel PCA
kpc = kpc[:, ::-1] #sort in descending order of eigenvalue
```



<figure>
  <center>
  <img src="/assets/images/kpca_pc_pairwise.svg" width="800">
   </center>
  <center>
    <figcaption> Figure 2. Relationship among the 4 top principal components.
    </figcaption>
  </center>
</figure>



#### Explicit Mapping in Feature Space

Alternatively, we can perform PCA on the mapped vectors in the feature space. The mapping to the feature space by a polynomial kernel is explicitly constructed, as described in a [previous blog post](<https://kezhaozhang.github.io/2023/04/26/kernel-transformation.html>). In this example, the data in the original space is two-dimensional and the polynomial kernel has a degree of 4. For a data point $\mathbf{x}=(x_1, x_2)$,  the mapping  is


$$
\phi(\mathbf{x}) =\left(
\begin{array}{c}
x_1^4\\
2 x_1^3 x_2\\
\sqrt{6}x_1^2 x_2^2\\
2 x_1 x_2^3\\
x_2^4
\end{array}
\right). \notag
$$
The following code converts data into feature space and calculate the PCA.

```python
from scipy.special import binom
def poly(x, n):
    """convert 2d data using polynomial kernel of degree n"""
    x1 = x[:, 0]
    x2 = x[:, 1]
    xc = np.column_stack(
      tuple(x1**(n-i)*x2**i*np.sqrt(binom(n,i)) 
            for i in range(n+1)))

    return xc
  
psi = poly(X, 4)  # mapping to feature space
psi = psi - psi.mean(axis=0) # center the mapped vector
cov = np.cov(psi.T) # covariance matrix
e,v = np.linalg.eig(cov) # eigenvalues and eigenvectors
index = np.argsort(e)[::-1] # select the largest 4 eigenvalues
pc_explicit = psi @ v[:, index[:4]] # top 4 principal components
```



To compare the results of the Kernel PCA method and explicit mapping, we examine the correlations between their principal components. The results are shown in Figure 3. Notably, the two methods exhibit a perfect correlation, differing only in the sign of the slope.



<figure>
  <center>
  <img src="/assets/images/kpca_kernel_vs_explicit.svg" width="800">
   </center>
  <center>
    <figcaption> Figure 3. The results of PCA using the kernel method and the explicit mapping are in strong agreement.
    </figcaption>
  </center>
</figure>



#### Kernel PCA with Scikit Learn

Lastly, we demonstrate Kernel PCA using Scikit-Learn's implementation. This provides another point of reference for our results.

```python
from sklearn.decomposition import KernelPCA
pca = KernelPCA(n_components=4, kernel='poly', degree=4, coef0=0, gamma=1)
pc_sklearn = pca.fit_transform(X)
```



The results using Scikit-Learn's implementation of Kernel PCA are also visualized in Figure 4.

<figure>
  <center>
  <img src="/assets/images/kpca_kernel_vs_sklearn.svg" width="800">
   </center>
  <center>
    <figcaption> Figure 4. The results of PCA using the kernel method and the Scikit-Learn Kernel PCA implementation are in strong agreement.
    </figcaption>
  </center>
</figure>



### Conclusion

In conclusion, Kernel PCA is a powerful technique for nonlinear data analysis. This article has explained the mathematical foundations of Kernel PCA and demonstrated its application on a real dataset. The results obtained through Kernel PCA, explicit mapping in the feature space, and Scikit-Learn's implementation are consistent, confirming the validity and utility of this method.

### References

Schölkopf, B., Smola, A., Müller, KR. (1997). Kernel principal component analysis. In: Gerstner, W., Germond, A., Hasler, M., Nicoud, JD. (eds) Artificial Neural Networks — ICANN'97. ICANN 1997. Lecture Notes in Computer Science, vol 1327. Springer, Berlin, Heidelberg. https://doi.org/10.1007/BFb0020217
