





https://towardsdatascience.com/understanding-support-vector-machine-part-2-kernel-trick-mercers-theorem-e1e6848c6c4d



Mercer's Theorem requires Kernel to be

- symmetric: $K(x_i, x_j) = K(x_j, x_i)$
- positive semi-definite: $\forall c,$ $c^T K c\geq 0$.

There exists a mapping $\phi$: $x: \mapsto \phi(x)$, such that $\left < \phi(x), \phi(y)\right> = K(x,y)$.



An example: Gaussian Radial Basis kernel
$$
K(x_i, x_j)=\exp(-\gamma (x_i-x_j)^2).
$$
Using Taylor expansion
$$
\begin{aligned}
K(x_i, x_j)&=\exp(-\gamma (x_i-x_j)^2)\\
  &= \exp(-\gamma(x_i^2+x_j^2))\exp(2\gamma x_i x_j)\\
  & = \exp(-\gamma(x_i^2+x_j^2))\sum_{k=0}^\infty \frac{(2\gamma x_i x_j)^k}{k!}
\end{aligned}
$$
If 
$$
\phi(x) = \exp(-\gamma x^2)\left[
	\begin{array}{c} 
	1 \\
	\sqrt{2\gamma} x\\
	\vdots \\
	\sqrt{\frac{(2\gamma)^k}{k!}}x^k\\
	\vdots
	\end{array}\right]
$$
Then $ K(x_i, x_j) = \left<\phi(x_i), \phi(x_j)\right>$.



This construction is limited to one-dimensional.

- It's limited to one dimension. When $x$ is more than one-dimensional, there is no appropriate definition of $\phi$, because there is no way to construct a dot product in the feature space where the $k-$th component of the Taylor expansion is $\left<\vec{x_i}, \vec{x_j}\right>^k$.

#### Positve Definite Kernel 

https://en.wikipedia.org/wiki/Positive-definite_kernel

Let $\mathcal {X}$ be a nonempty set, sometimes referred to as the index set. A [symmetric function](https://en.wikipedia.org/wiki/Symmetric_function) $K: \mathcal {X}\times \mathcal {X}\to \mathbb{R} $ is called a positive-definite (p.d.) kernel on $\mathcal {X}$ if
$$
\sum _{i=1}^{n}\sum _{j=1}^{n}c_{i}c_{j}K(x_{i},x_{j})\geq 0
$$
holds for any $ x_{1},\dots ,x_{n}\in \mathcal {X}$, given $ n\in \mathbb {N} ,c_{1},\dots ,c_{n}\in \mathbb {R} $

**A linear combination of Mercer kernels is a Mercer kernel (symmetry and positive semi-definite).**
$$
\exp(\mathbf{x_i}\cdot\mathbf{x_j}) = \sum_{k=0}^\infty \frac{(\mathbf{x_i}\cdot\mathbf{x_j})^k}{k!}
$$
A polynomial kernel 
$$
K(\mathbf{x}, \mathbf{y}) = (\mathbf{x}\cdot\mathbf{y})^d = (x_1 y_1 + x_2 y_2 + \ldots + x_n y_n)^d,
$$
where $\mathbf{x}=[x_1, x_2, \ldots, x_n]^T$,  and $\mathbf{y}=[y_1, y_2, \ldots, y_n]^T$.



