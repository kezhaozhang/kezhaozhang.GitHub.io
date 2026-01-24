---

title: "Determining Sample Sizes for Reliability Testing: A Bayesian Approach"
date: 2026-01-24
typora-root-url: ./..
---



When designing a new device, we often start with a reliability specification. For example, a common requirement might be that the failure rate should not exceed 1% during a specific testing period. This raises a practical question: how many devices do we need to test to be 80% confident that the true failure rate is actually below that 1% threshold? This note explores how to calculate those requirements.



### Failure Rate Distribution

Let $N$ be the total sample size,  $m$ be the number of failed devices observed during the test, and $\theta$ be the true (but unknown) failure rate ($1\le \theta \le 1$).



The likelihood of observing exactly $m$ failures in $N$ trials follows a binomial likelihood:



$$
p(N, m\mid \theta) =  \theta^m (1-\theta)^{N-m}. \notag
$$



Using Bayes' Theorem, we can determine the posterior distribution of $\theta$:



$$
p(\theta\mid N, m) =\frac{p(N, m\mid \theta) p(\theta)}{p(N, m)}, \label{eqn:posterior}
$$



where $p(\theta)$ is the prior and $P(N,m)$ is the normalization constant:



$$
p(N, m) = \int_0^1 p(N,m\mid \theta)p(\theta) d\theta.\notag
$$



If we assume a noninformative prior for $\theta$, we can use a uniform distribution where  $p(\theta) = 1$. By plugging this into Equation ($\ref{eqn:posterior}$), the posterior distribution of $\theta$  Becomes a Beta distribution, $\text{Beta}(\alpha, \beta)$, with $\alpha=m+1$ and $\beta=N-m + 1$:



$$
p(\theta\mid N, m) =\frac{\theta^{\alpha -1} (1-\theta)^{\beta-1}}{B(\alpha, \beta)},\notag
$$



where $B(\alpha, \beta)$ is the Beta function, defined using Gamma functions as  $B(\alpha, \beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha + \beta)}$. 



### Comparing Observation to Specification Limit

Once we have the posterior distribution for $\theta$, we can calculate the confidence level that our device meets the specification.

As shown in Figure 1, the area under the curve to the left of the specification limit ($\theta_0$) represents the probability that the true failure rate is below that limit. This area is our confidence level. The area to the right represents the risk that the failure rate exceeds our limit.

Mathematically, the confidence $C$ that $\theta<\theta_0$  is the Cumulative Distribution Function (CDF) of the Beta distribution:


$$
C = \int_0^{\theta_0} p(\theta\mid N, m) d \theta = I_{\theta_0}(m+1, N-m +1), \label{eqn:cdf}
$$


where $I_{\theta_0}$ is the regularized incomplete beta function.



<figure>
  <center>
  <img src="/assets/images/lifetime_confidence.svg" width="500">
   </center>
  <center>
    <figcaption> 
      Figure 1. The posterior distribtuion of the failure rate &theta;. The shaed area represents the confidence level (C) that the true failure rate i below the specification limit &theta;<sub>0</sub>.
    </figcaption>
  </center>
</figure>




### Results

In our original question the specification limit $\theta_0=0.01 \,(1\%)$. Becuase condifence level $C$ in Equation ($\ref{eqn:cdf}$) depends on both the sample size ($N$) and the number of failures ($m$), we can visualize the relationship using a contour plot.

The contour plot in Figure 2 reveals that the required sample size increases significantly as more failures are observed. The "best-case" scenario, where zero failures occur, provides the lower bound for the sample size.



<figure>
  <center>
  <img src="/assets/images/lifetime_contour.svg" width="450">
   </center>
  <center>
    <figcaption> 
      Figure 2. Contour plot of confidence levels (C) for a failure rate specification of &theta;<sub>0</sub>=0.01. The curves represent the required sample size (N) versus observed failures (m) to achieve 20%, 50%, and 80% confidence.
    </figcaption>
  </center>
</figure>



To see this more clearly, we can compare the confidence levels for $m=0$ and $m=2$ failures across various sample sizes. As shown in Figure 3, to achieve $80\%$ confidence that the failure rate is below 0.01:

- If 0 failures occur, we need at least 159 samples.
- If 2 failures occur, we need at least 427 samples.





<figure>
  <center>
  <img src="/assets/images/lifetime_m=2_p0.01_vs_N.svg" width="500">
   </center>
  <center>
    <figcaption> 
      Figure 3. Confidence level as a function of sample size (N) for cases with zero failures (m=0) and two failures (m=2). The plot illustrates the increased sample size required to maintain a 0.01 failure rate specification when failures are observed.
    </figcaption>
  </center>
</figure>






### Conclusion

This analysis demonstrates that the posterior distribution of a failure rate is a powerful and practical application of Bayes' theorem. By treating the failure rate as a distribution rather than a single fixed value, we can quantify our uncertainty and make data-driven decisions about test requirements. Using the Beta distribution as a conjugate prior allows for an elegant, closed-form solution to what would otherwise be a complex statistical problem.



### References

Beta distribution. In Wikipedia.  https://en.wikipedia.org/wiki/Beta_distribution

