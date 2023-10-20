---
title: "Transformations Corresponding to Kernels"
---


Mercer's Theorem is a fundamental result in kernel theory. It states that if we have a positive semi-definite kernel that is symmetric, we can find a mapping function $\phi$ that maps the input vector $\mathbf{x}$ to a higher dimensional space such that the dot product of the transformed vectors equals the kernel function $K(\mathbf{x},\mathbf{y})$. This is what is referred to as the kernel trick in support vector machines (SVM).

We will use Mercer's Theorem to calculate the transformed vectors in the high-dimensional space for the polynomial kernel and linear combination of kernels.



### Polynomial Kernel

The polynomial kernel is defined as:


$$
K(\mathbf{x}, \mathbf{y}) =  (\mathbf{x}\cdot\mathbf{y})^d = (x_1 y_1 + x_2 y_2 + \ldots + x_n y_n)^d,
\label{eqn:polynomial}
$$


where $\mathbf{x}=[x_1, x_2, \ldots, x_n]^T$,  and $\mathbf{y}=[y_1, y_2, \ldots, y_n]^T$.



We can expand and rearrange the RHS of the Equation ($\ref{eqn:polynomial}$) as follows:


$$
(x_1 y_1 + x_2 y_2 + \ldots + x_n y_n)^d = \sum_{\{i_m\}} \frac{d!}{\Pi_{m=1}^n i_m !}\Pi_{m=1}^n (x_m y_m)^{i_m},
$$


where $\{i_m\}$ are all combinations of $n$ non-zero integers such that $\sum_{m=1}^n i_m=d$. For example, if $n=2$, and $d=3$,   {$i_m$}= {3,0}, {2,1}, {1,2}, {0,3}.



If we define a transformation function $\phi$ as follows:


$$
\phi(\mathbf{x})=\left( 
	\begin{array}{c} 
	\vdots \\
	\sqrt{\sum_{\{i_m\}} \frac{d!}{\Pi_{m=1}^n i_m !}\Pi_{m=1}^n} x_m ^{i_m}\\
	\vdots
	\end{array}
\right)
$$



where each row is one combination of $\{i_m\}$ such that $\sum_{i_m=1}^n=d$,   then $\phi$ transforms $\mathbf{x}$ of $n$-dimensional original space into a vector in a generally higher-dimensional space. 

For instance, when $n=2$ and $d=3$, {$i_m$}= {3,0}, {2,1}, {1,2}, {0,3}.



$$
\phi(\mathbf{x})=\left(
\begin{array}{c}
x_1^3\\
\sqrt{3}x_1^2 x_2\\
\sqrt{3} x_1 x_2^2\\
x_2^3
\end{array}
\right).
$$


We can verify


$$
\begin{align*}
\left<\phi(\mathbf{x}), \phi(\mathbf{y})\right> = & x_1^3 y_1^3 + 3x_1^2 x_2 y_1^2 y_2+3 x_1 x_2^2 y_1 y_2^2+ x_2^3 y_2^3\\
= & (x_1 y_1 + x_2 y_2)^3 \\
= & (\mathbf{x}\cdot\mathbf{y})^3
\end{align*}.
$$



### Linear Combination of Kernels

Suppose a kernel $K$ is the sum of two kernels $K_1$ and $K_2$:


$$
K =\alpha^2 K_1 + \beta^2 K_2
$$


Let $\phi_1$ and $\phi_2$ be the corresponding transformations from the original space to the higher-dimensional spaces for $K_1$ and $K_2$, respectively. Then, the transformation for $K$ can be obtained by vertically concatenating $\phi_1$ and $\phi_2$ as follows:


$$
\phi(\mathbf{x})=\left(\begin{array}{c}
\alpha\phi_1(\mathbf{x})\\
\beta\phi_2(\mathbf{x})
\end{array}
\right).
$$


In other words, $\phi(\mathbf{x})$ is a vertical concatenation of $\phi_1(\mathbf{x})$ and $\phi_2(\mathbf{x})$. Then, we have


$$
\begin{align*}
K(\mathbf{x}, \mathbf{y})  =&\left<\phi(\mathbf{x}),\phi(\mathbf{y})\right>=\left<\left(\begin{array}{c}
\alpha\phi_1(\mathbf{x})\\
\beta\phi_2(\mathbf{x})
\end{array}\right), \left(\begin{array}{c}
\alpha\phi_1(\mathbf{y})\\
\beta\phi_2(\mathbf{y})
\end{array}\right)\right>\\
= & \alpha^2\left<\phi_1(\mathbf{x}), \phi_1(\mathbf{y})\right> + \beta^2\left<\phi_2(\mathbf{x}), \phi_2(\mathbf{y})\right>\\ 
= & \alpha^2 K_1(\mathbf{x}, \mathbf{y}) + \beta^2 K_2(\mathbf{x}, \mathbf{y})
\end{align*}
$$



### Radial Basis Function Kernel

The Radial Basis Function (RBF) kernel is defined as follows:


$$
\begin{align*}
K(\mathbf{x}, \mathbf{y}) &= e^{-\gamma \mid \mathbf{x}-\mathbf{y}\mid^2}\\
 & = e^{-\gamma (\mathbf{x}\cdot\mathbf{x}+\mathbf{y}\cdot\mathbf{y})}e^{2\gamma\mathbf{x}\cdot\mathbf{y}}
\end{align*}
$$


where $\gamma$ is a positive constant. 

We can express $e^{2\gamma \mathbf{x}\cdot\mathbf{y}}$ as a sum of an infinite number of polynomials:


$$
e^{2\gamma \mathbf{x}\cdot\mathbf{y}}=\sum_{k=0}^\infty\frac{(2\gamma \mathbf{x}\cdot\mathbf{y})^k}{k!},
$$


The transformation corresponding to the RBF kernel is obtained by concatenating an infinite number of transformations corresponding to polynomial kernels:



$$
\phi_{\mathrm{RBF}}(\mathbf{x}) =e^{-\gamma \mathbf{x}\cdot \mathbf{x}}
\left( 
	\begin{array}{c} 
\vdots \\
\sqrt{\frac{(2 \gamma)^{k-1}}{(k-1)!}} \phi_{k-1}(\mathbf{x})\\
\sqrt{\frac{(2 \gamma)^k}{k!}} \phi_k(\mathbf{x})\\
\sqrt{\frac{(2 \gamma)^{k+1}}{(k+1)!}} \phi_{k+1}(\mathbf{x})\\
\vdots
\end{array}
\right),
$$


where $\phi_k$ is the transformation corresponding to a polynomial kernel of degree $k$:


$$
K(\mathbf{x}, \mathbf{y})=(\mathbf{x}\cdot\mathbf{y})^k = \left<\phi_k(\mathbf{x}), \phi_k(\mathbf{y})\right>.
$$
