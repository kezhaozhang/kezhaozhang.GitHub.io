---
title: Conditional Distribution of Multivariate Gaussian Variables: A Simple Derivation
date: 2023-05-30
---



We present a straightforward derivation for calculating the conditional probability distribution of multivariate Gaussian variables.

Let's consider a multivariate Gaussian random variable $\mathbf{y}$ with a mean of zero, denoted as:



$$
\mathbf{y} \sim \mathcal{N}(0, \Sigma).\notag
$$


where $\Sigma$ is the covariance matrix of $\mathbf{y}$.

We partition $\mathbf{y}$ into two parts, $\mathbf{y}_1$ and $\mathbf{y}_2$:


$$
\mathbf{y}=\left[
\begin{array}{c}
\mathbf{y}_1\\
\mathbf{y}_2
\end{array}
\right].
\notag
$$


Our goal is to find out the probability distribution of $\mathbf{y}_2$ conditioned on $\mathbf{y}_1$.



Since $\Sigma$ is symmetric and positive definite, we can perform a Cholesky decomposition to express it as the product of a lower triangular matrix $L$ and its transpose:



$$
\Sigma =L L^T, \notag
$$



We can further express $L$ in block form:



$$
L = \left[
\begin{array}{cc}
A & 0 \\
C & D
\end{array}
\right],
\notag
$$



where $A$ and $D$ are square matrices that are lower triangular. The dimensions of $A$, $D$, and $C$ are $n\times n$, $m\times n$, and $m\times m$ respectively, where $n$ and $m$ represent the lengths of $\mathbf{y}_1$ and $\mathbf{y}_2$ respectively.



By utilizing this decomposition, we can represent $\mathbf{y}$ as a linear transformation of independent, normally distributed variables:



$$
\mathbf{y} = L \mathbf{u},
\label{eqn:transform}
$$


where  $\mathbf{u} \sim  \mathcal{N}(0, \mathbf{I})$ and $\mathbf{I}$ denotes the identity matrix.



Expanding this equation, we have:


$$
\left[
\begin{array}{c}
\mathbf{y}_1\\
\mathbf{y}_2
\end{array}

\right]= \left[
\begin{array}{cc}
A & 0 \\
C & D
\end{array}
\right] 
\left[
\begin{array}{c}
\mathbf{u}_1\\
\mathbf{u}_2
\end{array}
\right],
$$


where the lengths of $\mathbf{u}_1$ and $\mathbf{u}_2$ are $n$ and $m$, respectively.

Isolating $\mathbf{y}_2$, we find:
$$
\mathbf{y}_2 = C \mathbf{u}_1 + D\mathbf{u}_2.
$$




Given that $\mathbf{y}_1$ and $\mathbf{u}_1$ are fixed, we can express $\mathbf{u}_1$ as:


$$
\mathbf{y}_1 = A \mathbf{u}_1
$$


$$
\mathbf{u}_1=A^{-1}\mathbf{y}_1.
$$


This implies that $\mathbf{y}_2$ follows a multivariate Gaussian distribution:



$$
\mathbf{x}_2 \mid \mathbf{x}_1 \sim \mathcal{N}(C\mathbf{u}_1, DD^T).
$$



To proceed, let's calculate $C\mathbf{u}_1$ and $D D^T$. 



First we express the covariance matrix  in block form:



$$
\Sigma =\left[
\begin{array}{cc}
\Sigma_{11} & \Sigma_{12}\\
\Sigma_{21} & \Sigma_{22}
\end{array}
\right],
\notag
$$



where $\Sigma_{ij}=\mathrm{cov}(\mathbf{y}_i, \mathbf{y}_j)$ and $i, j=1,2$. Then we can rewrite $\Sigma$ as:



$$
\Sigma = 

L L^T 
=\left[
\begin{array}{cc}
A A^T & A C^T\\
C A^T & C C^T + D D^T
\end{array}
\right],
\label{eqn:cov_matrix}
$$



where $L$ is the matrix given earlier. From the equation above, we can observe that:



$$
\Sigma_{21} = C A^T \Rightarrow C = \Sigma_{21} (A^T)^{-1}
$$
Now, let's calculate $C\mathbf{u}_1$:


$$
C\mathbf{u}_1 = \Sigma_{21} (A^T)^{-1} A^{-1}\mathbf{y}_1 = \Sigma_{21}(A A^T)^{-1} \mathbf{y}_1 = \Sigma_{21}\Sigma_{11}^{-1} \mathbf{y}_1,
$$


where we have used the fact that $(A A^T)^{-1} = (A^T)^{-1} A^{-1}$.

Next, we calculate $C C^T$:



$$
C C^T = \Sigma_{21}(A^T)^{-1} A^{-1}\Sigma_{12}=\Sigma_{21}(A A^T)^{-1}\Sigma_{12}=\Sigma_{21}\Sigma_{11}^{-1}\Sigma_{12}.
$$


Using this result, we can find $D D^T$:


$$
D D^T = \Sigma_{22} - C C^T = \Sigma_{22}-\Sigma_{21}\Sigma_{11}^{-1}\Sigma_{12}.
$$


Finally, when the mean $\mathbf{\mu}$ is non-zero:


$$
\mathbf{\mu}=\left[
\begin{array}{c}
\mathbf{\mu}_1\\
\mathbf{\mu}_2
\end{array}
\right],\notag
$$


we adjust the mean and obtain:
$$
\mathbf{y}_2 \mid \mathbf{y}_1 \sim \mathcal{N}(\mathbf{\mu}_2+\Sigma_{21}\Sigma_{11}^{-1} (\mathbf{y}_1-\mathbf{\mu}_1), \Sigma_{22}-\Sigma_{21}\Sigma_{11}^{-1}\Sigma_{12}).
$$

