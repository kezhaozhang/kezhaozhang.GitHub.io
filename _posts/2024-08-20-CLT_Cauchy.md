---

title: "Central Limit Theorem and Cauchy Distribution"
date: 2024-08-20
typora-root-url: ./..
---



This note demonstrates the Central Limit Theorem (CLT) using the Fourier transform of the probability density function (PDF) and emphasizes the requirement that the mean and variance of the random variable must exist. It then contrasts this with the Cauchy distribution as a counter-example, where neither the mean nor the variance is defined, illustrating why the CLT does not hold in this case.



### Central Limit Theorem

The Central Limit Theorem states that the average of nn independent and identically distributed (i.i.d.) random variables approaches a normal distribution as nn becomes large, provided that the mean and variance of the random variables exist and are finite. The following derivation uses the Fourier transform to illustrate this concept.



 Let $x$ be a random variable. The Fourier transform of its PDF $p(x)$ is given by



$$
\langle e^{ikx}\rangle=\int p(x) e^{ikx}dx =\hat{p}(k),\notag
$$



where $\langle\cdot\rangle$ denotes the expected value,  and $\hat{p}(x)$ represents the Fourier transform of $p(x)$. If we know the Fourier transform of the PDF,  we can obtain the PDF from its inverse Fourier transform.



Without loss of generality, assuming the mean of the random variable $x$ is $0$:   


$$
\langle x \rangle=0 \notag.
$$


and its variance is 


$$
\sigma^2 = \langle x^2\rangle.\notag
$$



 Consider the normalized average of $n$ i.i.d. random variables:



$$
y = \frac{x_1 + x_2 + \cdots + x_n}{\sqrt{n}} \notag
$$



and the Fourier transform of $y$ is



$$
\begin{align} \notag
\langle e^{iky}\rangle &= \langle e^{i\frac{x_1+x_2+\cdots+x_n}{\sqrt{n}}}\rangle \\ \notag
&= \langle \Pi_{i=1}^n e^{i\frac{x_i}{\sqrt{n}}}\rangle\\ \notag
&=\Pi_{i=1}^n\langle e^{i\frac{x_i}{\sqrt{n}}}\rangle \\
& = \langle e^{i\frac{x}{\sqrt{n}}}\rangle^n  \label{eqn:ft}
\end{align}
$$



The penultimate line follows from the independence of  $x_1, x_2, \ldots, x_n$ are i.i.d. To demonstrate:



Let $u$ and $v$ be two i.i.d. random variables,  because of independence


$$
p(u, v) = p(u) p(v).\notag
$$


Then for any functions $f$ and $g$,  


$$
\begin{align}\notag
\langle f(u) g(v)\rangle &= \iint p(u, v) f(u) g(v)du dv \\ \notag
& = \iint p(u)f(u)du p(v)g(v)dv \\ \notag
& = \int p(u)f(u)du \int p(v)g(v)dv\\ \notag
& = \langle f(u)\rangle \langle g(v)\rangle.
\end{align}
$$



Apply this to Equation ($\ref{eqn:ft}$): 


$$
\begin{align}\notag
\langle e^{iky} \rangle = \langle e^{i\frac{x}{\sqrt{n}}}\rangle^n & \approx\langle1+i\frac{kx}{\sqrt{n}}-\frac{k^2x^2}{2n}\rangle^n \\
& = (1+i\frac{k\langle x\rangle}{\sqrt{n}}-\frac{k^2 \langle x^2 \rangle}{2 n})^n \\
\end{align}
$$


As $n\to\infty$, 



$$
\lim_{n\to\infty}\langle e^{iky}\rangle = \lim_{n\to\infty}\left(1-\frac{k^2\sigma^2}{2n}\right)^n = e^{-\frac{k^2\sigma^2}{2}} =\hat{p}(k) \label{eqn:limit}
$$


Thus, the PDF of $y$ is the inverse Fourier transform of $\hat{p}(k)$:



$$
p(y) = \frac{1}{2\pi}\int_{-\infty}^\infty \hat{p}(k) e^{-i k y} dk = \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{y^2}{2\sigma^2}}.\notag
$$



Therefore, the distribution of $y$ is normal, $y\sim \cal{N}(0, \sigma^2)$. The derivation above relies on the existence of a finite mean and variance for $x$.



### Cauchy Distribution



The Cauchy distribution is a well-known counterexample where the CLT does not apply because its mean and variance are undefined. The PDF of the Cauchy distribution is given by


$$
p(x) = \frac{1}{\pi \gamma \left(1+\frac{(x-\mu)^2}{\gamma^2}\right)}
$$



where $\mu$ is the location parameter and $\gamma$ is the scaling factor. Without loss of generality, we assume $\mu=0$. 



For the Cauchy distribution, 


$$
\langle e^{ik x} \rangle = \int p(x) e^{ikx} dx = e^{-\gamma \mid k\mid}.\notag
$$


Consider the average of nn i.i.d. random variables from the Cauchy distribution:


$$
y = \frac{x_1 + x_2 +\cdots + x_n}{n}.\notag
$$


The Fourier transform of $y$ is


$$
\langle e^{ik y} \rangle = \Pi_i \langle e^{ik\frac{x_i}{n}}\rangle = \left(e^{-\frac{\gamma \mid k\mid}{n}}\right)^n = e^{-\gamma\mid k \mid}.
$$



$$
\begin{align} \notag
 \langle e^{ik\frac{x}{n}}\rangle &= \int_{-\infty}^\infty p(x) e^{ik\frac{x}{n}}dx \\ \notag
 & = \int_{-\infty}^\infty \frac{1}{\gamma\pi\left(1+\frac{x^2}{\gamma^2}\right)}e^{ik\frac{x}{n}}dx \\ \notag
 & = \int_{-\infty}^\infty \frac{1}{\gamma\pi\left(1+\frac{x^2}{\gamma^2}\right)}\cos( k\frac{x}{n})dx \\ \notag
 & = e^{-\frac{\gamma \mid k\mid}{n}}
 \end{align}
$$



This result matches the Fourier transform of the Cauchy distribution. Thus, the average of i.i.d. Cauchy random variables is also Cauchy-distributed, not normally distributed. The failure of the CLT here is due to the undefined mean and variance of the Cauchy distribution.



### Numerical Experiments



We perform numerical experiments using the Wolfram Language to analyze the means of i.i.d. random variables from the Exponential and Cauchy distributions. Unlike the Cauchy distribution, the Exponential distribution has a well-defined mean and variance.



```mathematica
expdist = ExponentialDistribution[1];
cauchydist = CauchyDistribution[0,1];
(* mean of iid random variables *)
Xexp = Mean/@RandomVariate[expdist, {200, 10000}];
Xcauchy = Mean/@RandomVariate[cauchydist, {200, 1000}];
```





<figure>
  <center>
  <img src="/assets/images/clt_pdf_exp_cauchy.svg" width="600">
   </center>
  <center>
    <figcaption> Figure 1. PDFs of the exponential distribution and the Cauchy distribution.
    </figcaption>
  </center>
</figure>





<figure>
  <center>
  <img src="/assets/images/clt_histogram_mean_exp_cauchy.svg" width="700">
   </center>
  <center>
    <figcaption> Figure 2. Distribution of the mean of 10000 random variables.
    </figcaption>
  </center>
</figure>



<figure>
  <center>
  <img src="/assets/images/clt_qq_plot_exp_cauchy.svg" width="600">
   </center>
  <center>
    <figcaption> Figure 3. QQ-Plot shows that the mean of the Exponetial distribution is normally distributed but the mean of Cauchy distirbution is not. x-axis:  quantile of the mean of the random variable; y-axis: quantile of normal distribution.
    </figcaption>
  </center>
</figure>



We perform numerical experiments using the Wolfram Language to analyze the means of i.i.d. random variables from the Exponential and Cauchy distributions. Unlike the Cauchy distribution, the Exponential distribution has a well-defined mean and variance.



| Rank | Mean of Exponential Distribution         | Mean of Cauchy Distribution               |
| ---- | ---------------------------------------- | ----------------------------------------- |
| 1    | **NormalDistribution[1.00029, 0.00948]** | **CauchyDistribution[0.113, 0.791]**      |
| 2    | LogisticDistribution[1.00007, 0.00523]   | StudentTDistribution[0.112, 0.751, 0.911] |
| 3    | GammaDistribution[11069.5, 0.0000903]    | LaplaceDistribution[4.25, 10.23]          |



### Conclusion

The Central Limit Theorem reveals that the normal distribution emerges naturally from the sum of a large number of independent and identically distributed random variables, provided their mean and variance are finite. However, the applicability of the CLT is contingent upon the existence of finite mean and variance for these random variables. Without these finite parameters, the theorem does not hold, emphasizing the importance of these conditions for the CLT's validity.



### References

Körner, T. W. (1988). Fourier Analysis. Cambridge University Press.
