---


title:	Multivariate Gaussian Distribution As Linear Transformation of Independent Normally Distributed Random Variables
---



This note explores the relationship between multivariate Gaussian variables and the linear transformation of independent, normally distributed random variables. The main results include the derivation of the probability density function (PDF) for multivariate Gaussian distribution and the recognition that there are infinite linear transformations capable of transforming independent, normally distributed random variables into multivariate Gaussian variables. The report also demonstrates specific methods of constructing these transformations using decomposition techniques such as singular value decomposition and Cholesky decomposition. 

Without loss of generality, we assume zero mean for the random variables.

### Independent Random Variables with Normal Distribution

We begin by considering a set of independent random variables, denoted as $\mathbf{u}$, each following a normal distribution with zero mean and an $n\times n$ identity covariance matrix $\bf{I}$:


$$
\mathbf{u} \sim \mathcal{N}(0, \mathbf{I}).\notag
$$



### Linear Transformation

Next, we introduce a linear transformation, where $\mathbf{y}$ is obtained by multiplying $\mathbf{u}$ with a transformation matrix $A$ of size $n \times n$, i.e., $\mathbf{y} = A \mathbf{u}$. The covariance matrix of $\mathbf{y}$ is given by $\Sigma = A A^T$.



### Probability Distribution Function (PDF)

To derive the PDF of $\mathbf{y}$, we start by considering the PDFs of $\mathbf{u}$ and $\mathbf{y}$, denoted as $p(\mathbf{u})$ and $q(\mathbf{y})$, respectively. We have the relationship:



$$
p(\mathbf{u})d\mathbf{u} = q(\mathbf{y})d\mathbf{y},\notag
$$


where $d\mathbf{u}=du_1du_2\cdots du_n$ and $d\mathbf{y} = dy_1 d_y2\cdots dy_n$.

Since the components of $\mathbf{u}$ are independent, $p(\mathbf{u})$ can be expressed as the product of $n$ one-dimensional normal distributions:


$$
p(\mathbf{u}) = \frac{1}{(2 \pi)^{\frac{n}{2}}}\exp\left({- \frac{1}{2}\mathbf{u}^T \mathbf{u}}\right).\notag
$$



We can express $\mathbf{u}$ in terms of $\mathbf{y}$ as $\mathbf{u} = A^{-1}\mathbf{y}$. Therefore, we have:


$$
d \mathbf{u} = \mid \mathrm{det}(A^{-1})\mid d \mathbf{y}. \notag
$$


Furthermore, by substituting $\mathbf{u}$ into the expression for $p(\mathbf{u})$, we obtain:


$$
\begin{align}\notag
 p(\mathbf{u}) &=\frac{1}{(2 \pi)^{\frac{n}{2}}}\exp\left({- \frac{1}{2}\mathbf{u}^T \mathbf{u}}\right)\\ \notag
 & = p(\mathbf{u}(\mathbf{y}))\\ \notag
 & = \frac{1}{(2 \pi)^{\frac{n}{2}}}\exp\left({- \frac{1}{2}(A^{-1} \mathbf{y})^T (A^{-1}\mathbf{y})}\right)\\
 & = \frac{1}{(2 \pi)^{\frac{n}{2}}}\exp\left({- \frac{1}{2}\mathbf{y}^T (A^{-1})^T A^{-1}\mathbf{y}}\right)
\end{align}.
\label{eqn:pu_y}
$$



Using  the properties of invertible matrices, specifically $(AB)^{-1} = B^{-1}A^{-1}$ and $(A^T)^{-1} = (A^{-1})^T$ , we have


$$
\Sigma^{-1} = (A A^T)^{-1} = (A^{-1})^T A^{-1}.\notag
$$


Equation ($1$) becomes
$$
p(\mathbf{u}) = \frac{1}{(2 \pi)^{\frac{n}{2}}}\exp\left({-\frac{1}{2}\mathbf{y}^T\Sigma^{-1}\mathbf{y}}\right). \notag
$$



Additionally, we have the properties:


$$
\mathrm{det}(A) = \mathrm{det}(A^T) \notag
$$

and


$$
\mathrm{det}(\Sigma) =\mathrm{det}(A A^T)=\mathrm{det}(A)^2. \notag
$$


Using these properties, we can deduce that:


$$
\mid \mathrm{det}(A^{-1})\mid = \frac{1}{\mid \mathrm{det}(A)\mid}=\frac{1}{\sqrt{\mathrm{det}(\Sigma)}}. \notag
$$


Finally, we can express the probability density function (PDF) for $\mathbf{u}$ in terms of $\mathbf{y}$ as follows:


$$
p(\mathbf{u}) d\mathbf{u} = p(\mathbf{u}(\mathbf{y}))\mid \mathrm{det}(A^{-1})\mid d \mathbf{y}
 = \frac{1}{(2 \pi)^{\frac{n}{2}}\sqrt{\mathrm{det}(\Sigma)}}\exp\left({-\frac{1}{2}\mathbf{y}^T\Sigma^{-1}\mathbf{y}}\right) d\mathbf{y}.\notag
$$


Therefore, the probability density function (PDF) for $\mathbf{y}$ is given by:


$$
q(\mathbf{y}) = \frac{1}{(2 \pi)^{\frac{n}{2}}\sqrt{\mathrm{det}(\Sigma)}}\exp\left({-\frac{1}{2}\mathbf{y}^T\Sigma^{-1}\mathbf{y}}\right),
$$



where $n$ represents the length of the vector $\mathbf{y}$.



### Infinite Transformations From Independent Normal Distributions



The transformation of independent, normally distributed random variables can generate multivariate Gaussian distributed variables. Notably, there are infinitely many such transformations available.

Some of these transformations can be constructed using the decomposition of the covariance matrix $\Sigma$. For instance, by employing singular value decomposition, we can express $\Sigma$ as:


$$
\Sigma = u\Lambda u^T,\notag
$$
 

where $u$ is a matrix of eigenvectors and $\Lambda$ is a diagonal matrix of eigenvalues. The linear transformation $A$ can then be constructed as:


$$
A = u \sqrt{\Lambda}
$$


Another construction method involves Cholesky decomposition, where we can express $\Sigma$ as:


$$
\Sigma = L L^T,
$$
with $L$ representing the lower triangular matrix resulting from the decomposition and the linear transformation is $L$.



There exist infinitely many such transformations. Suppose we have a transformation matrix $A$ with its rows representing vectors in the linear space. The angle between any pair of these vectors is determined by the covariance of the corresponding multivariate Gaussian variables. If we rigidly rotate all the row vectors, the resulting matrix would lead to the same multivariate Gaussian distribution since the mean and covariance matrix remain unchanged. Since there are infinite possible rotations, there are consequently infinitely many transformations that can convert independent, normally distributed random variables into the multivariate Gaussian distribution. Furthermore, all these transformation matrices maintain the same angles between pairs of row vectors.



Let's consider the two-dimensional case, where the covariance matrix can be written as:

 
$$
\Sigma = \left (
\begin{array}{cc}
\sigma_1^2 & \rho \sigma_1 \sigma_2\\
\rho \sigma_1 \sigma_2 & \sigma_2^2
\end{array}
\right)
$$


where $-1\leq \rho \leq 1$, and we can express $\rho = \cos(\Delta)$.



We can construct the transformation matrix $A$ as:


$$
A = \left(
\begin{array}{cc}
\sigma_1 \cos\theta & \sigma_1\sin\theta\\
\sigma_2 \cos(\theta+\Delta) & \sigma_2\sin(\theta+\Delta)
\end{array}
\right),
$$


where $\theta$ can take on infinitely many values. It can be confirmed that:

 
$$
A A^T =\left(
\begin{array}{cc}
\sigma_1^2 & \sigma_1\sigma_2\cos\Delta\\
\sigma_1\sigma_2\cos\Delta & \sigma_2^2
\end{array}
\right) = \Sigma
$$



Since there are infinite values of $\theta$ available to construct the transformation matrix $A$, there are infinitely many transformation matrices that lead to the same covariance matrix $\Sigma$.