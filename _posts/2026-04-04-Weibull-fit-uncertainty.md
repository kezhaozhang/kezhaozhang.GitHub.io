---

title: "Navigating Uncertainty: A Simulation Study on Censored Weibull Distributions"
date: 2026-04-04
typora-root-url: ./..
---





This note investigates the effect of censoring on the uncertainty of parametric estimation in survival data. We use simulations with random values drawn from a Weibull distribution to evaluate uncertainty in estimates of the shape and scale parameters, as well as the projected lifetime at specific failure rates across various censoring rates. We also check how censoring affects the correlation between the shape and scale parameters of the Weibull distribution.





### Introduction

In survival analysis, we often need to estimate the distribution of survival data and its derived parameters. For example, if the data is assumed to follow a Weibull distribution, we want to estimate the shape ($\beta$) and scale ($\alpha$) parameters. Additionally, we often estimate the projected lifetime at a specific failure rate so that experiment results can be compared against the design specification.



When survival data are censored, information about the intrinsic distribution is obscured. As a result, uncertainty increases in the parametric estimation. Without loss of generality, we assume the survival data follow a Weibull distribution and use simulations to quantify the effect of censoring on:

- Shape and scale parameters of the Weibull distribution.
- Projected lifetime at a given failure rate.
- Correlation between the shape and scale parameters.





### Weibull Distribution



The distribution is characterized by the shape parameter $\beta$ and the scale parameter $\alpha$.  The probability density function (PDF) is defined as:



$$
f (t; \alpha, \beta)= \frac{\beta}{\alpha}\left(\frac{t}{\alpha} \right)^{\beta-1} e^{-\left(t/\alpha\right)^\beta}. \label{eqn_pdf}
$$



The survival probability at time $t$ is



$$
S(t; \alpha, \beta) = e^{-\left(t/\alpha\right)^\beta}. \notag
$$



For a given failure rate $p$, $S(t_p; \alpha, \beta)= 1-p$, the projected lifetime $t_p$ is:



$$
t_p = \alpha \left[-\ln(1-p)\right]^{\frac{1}{\beta}} \label{eqn_lt}.
$$





### Censoring



In this study, we use a simple right-censoring scheme: samples with lifetime longer than a given cenoring limit are censored, and their censored lifetime is set to that limit. 



Figure 1 shows 50 samples of lifetime values from a Weibull distribution ($\alpha=100$, $\beta=1.5$) without and with censoring at a limit of 30.



<figure>
  <center>
  <img src="/assets/images/life_plot_censored_uncensored.svg" width="700">
   </center>
  <center>
    <figcaption> 
      Figure 1. Lifeline plots of 50 lifetime data samples. The left plot displays complete data, while the right plot incorporates a right-censoring limit of 30. The endpoints represent uncensored failures (black dots) and censored observations (red dots), respectively.
    </figcaption>
  </center>
</figure>






### Maximum Likelihood Estiamte of Weibull Distribution Parameters



The parameters $\alpha$ and  $\beta$ can be estimated from censored data with Maximum Likilihood Estimate (MLE). After censoring, the log likelohood consists of two parts: the sum of log PDFs of the uncensored samples and the sum of log survival probabilites of the censored samples:



$$
\log L = \sum_{i\in \mathrm{uncensored}} \log f(t_i; \alpha, \beta) + \sum_{j\in \mathrm{censored}} \log S(t_j; \alpha, \beta). \notag
$$



Figure 2 illustrates the contour plots of $\log L$ for uncensored and censored data. The contour shapes differ significantly, leading to changes in the correlation between estimated $\alpha$ and $\beta$.



<figure>
  <center>
  <img src="/assets/images/alpha_beta_contours_combined.svg" width="750">
   </center>
  <center>
    <figcaption> 
      Figure 2. Log likelihood contour plots as functions of &alpha; and &beta; for uncensored and censored data. Samples are randomly drawn from a Weibull distribtuion with true parameters &alpha;=100 and &beta;=1.5. For the censored case, a right-censoring limit of 30 is applied. The red dots indicate the MLE estimates.  
    </figcaption>
  </center>
</figure>






### Results



#### Estimate of Distribution Parameters



We simulated censored survival data from a Weibull distribution ($\alpha=100$, $\beta=1.5$). $50$ random values were generated; values greater than $30$ were censored and replaced by $30$. The data was fit to the Weibull distribution and then to estimate lifetimes at various failure rates. This procedure was repeated $200$ times to observe the variation.  Figure 3 shows the distribution of the esitmated $\alpha$ and $\beta$. 

 



<figure>
  <center>
  <img src="/assets/images/fit_alpha_beta_hist.svg" width="650">
   </center>
  <center>
    <figcaption> 
      Figure 3. Distribution of estimated scale paraemter &alpha; and shape paraemter &beta; from right-censored data. The vertical black lines indicate the true values used for data generation.
    </figcaption>
  </center>
</figure>






#### Porjected Lifetime at a Given Failure Rate




The projected lifetime at a given failure rate $p$ is calculated using Equation ($\ref{eqn_lt}$).  Figure 4 shows the distribution of  lifetime at various failure rates $p$. In these plots, the lifetime is scaled by its mean for each $p$.  The spread of the projected lifetime varies non-monotonically with $p$. 



<figure>
  <center>
  <img src="/assets/images/lifetime_at_fr_hist.svg" width="550">
   </center>
  <center>
    <figcaption> 
      Figure 4. Distributions of projected lifetimes at various failure rates <i>p</i>. Each lifetimes is scaled by its mean for each value of <it>p</it> to facilitate comparison across failure rates <it>p</it>.
    </figcaption>
  </center>
</figure>





The normal probability plots  or probability-probability (P-P) plots of the log lifetime in Figure 5 suggest that  theprojected lifetime can be approximated by a lognoraml distribution.

<figure>
  <center>
  <img src="/assets/images/lifetime_pp_plot.svg" width="550">
   </center>
  <center>
    <figcaption> 
      Figure 5. Normal probability plots of the log lifetime for variou failure rates <i>p</i>.
    </figcaption>
  </center>
</figure>

#### 

#### Uncertainty and the Coefficent of Variation (COV)




We use the Coefficient of Variation (COV), the ratio of the standard devation to the mean,  to represent the uncertianty of the lifetime estimate. Figure 6 shows the dependence of COV on $p$ across three fitting methods: 

1. Simultaneous estimation of $\alpha$ and $\beta$ : Uncertainty is the lowest around $p = 0.1$ and increases rapidly at samller values of $p$. This suggests that projecting lifetime for very low failure rate is inadvisable due to high uncertainty. 

2. Estimating $\alpha$ with $\beta$ fixed to its true value: Uncertianty is signficantly reduced and becomes independent of $p$. While this is a recommednde approach to handle low failire counts, it introduces bias if $\beta$ is incorrect. 

3. Estimating $\beta$ with $\alpha$ fixed to its true value: The minimum of COV occurvs at $p=1-1/e$, derived from the relationship COV $\propto  \frac{-1}{(1-p)\log (1-p)}$, which is a result of Equation ($\ref{eqn_lt}$).



<figure>
  <center>
  <img src="/assets/images/cov_vs_failure_rate_p.svg" width="550">
   </center>
  <center>
    <figcaption> 
      Figure 6. Uncertianty of the projected lifetime as a function of failure <it>p</it> 
    </figcaption>
  </center>
</figure>




Figure 7 demonstrates the cost of fixing $\beta$: while it reduces variance, it introduces a pronounced shift (bias) in the estimated $\alpha$ if the fixed $\beta$ is inaccurate. 



<figure>
  <center>
  <img src="/assets/images/fixed_beta_alpha_bias_cdf.svg" width="550">
   </center>
  <center>
    <figcaption> 
      Figure 7. Distributions of &alpha; estimates for different fixed levels of &beta;. This comparison illustrates the sensitivity of the scale parameter estimation to the assumed shape parameter.
    </figcaption>
  </center>
</figure>






#### Effect of Censoring Rate



As censoring masks information and increase uncertianty, the estimated distribution parameters have more variation with higher censoring rates, as shown in Figure 8. Censoring also shifts the correlation between esitmated $\alpha$ and $\beta$ (Figure 9). The correlation is positive for uncensored data but decreases and eventually becomes negative as the censoring rate increases. This trend is consistent with the shifting ridges in the log-likelihood plots show in Figure 2. 



<figure>
  <center>
  <img src="/assets/images/alpha_beta_cdf_vs_censoring.svg" width="750">
   </center>
  <center>
    <figcaption> 
      Figure 8. Distributions of estimated &alpha; and &beta; for various censoring rates.
    </figcaption>
  </center>
</figure>






<figure>
  <center>
  <img src="/assets/images/alpha_beta_correlation_vs_censoring_rate.svg" width="750">
   </center>
  <center>
    <figcaption> 
      Figure 9. Correlation between estimated &alpha; and &beta; depends on censoring rate.
    </figcaption>
  </center>
</figure>



The exact correlation coefficient  with uncensored data is shown in Equation ($\ref{eqn_cor}$), which is derived in the Appendix.  The distribution of the correlation coefficient from simulation is shown in Figure 10 and it agrees with the theoretical value indicated by the vertical black line.



<figure>
  <center>
  <img src="/assets/images/correlation_coefficient_bootstrap.svg" width="500">
   </center>
  <center>
    <figcaption> 
      Figure 10. Distribution of correlation coefficent between estimated &alpha; and &beta; from fitting uncensored simulated data to the Weibull distribution. The vertical black line indicates the theoretical correlation value.
    </figcaption>
  </center>
</figure>




### Conclusion



Censoring increases the uncertainty of distribution parameters and projected lifetimes. Due to this uncertainty, projecting lifetimes at very low failure rates is risky. While fixing the shape parameter $\beta$ can reduce uncertainty, it significantly increases the risk of estimation bias.






### Appendix: Correlation Coefficient Between $\alpha$ and $\beta$ Without Censoring



We calculate the correlation coefficient for uncersored data using the Fisher Information Matrix $\mathcal{I}$. For a single observation the log-likelihood $\ell$ is:



$$
\ell =\ln(\beta)−\beta\ln(\alpha)+(\beta−1)\ln(t)−\left(\frac{t}{\alpha}\right)^\beta. \notag
$$



The Fisher information matrix is



$$
\mathcal{I} = \begin{bmatrix} \notag
\mathcal{I}_{\alpha\alpha} & \mathcal{I}_{\alpha\beta} \\
\mathcal{I}_{\beta\alpha} & \mathcal{I}_{\beta\beta}

\end{bmatrix},
$$



where 

$$
\begin{align} \notag
\mathcal{I}_{\alpha\alpha} & = -E\left[\frac{\partial^2\ell}{\partial \alpha^2}\right]\\ \notag
& = -\int\frac{\partial^2 \ell}{\partial\alpha^2}f(t;\beta, \alpha) dt\\ \notag
& = \frac{\beta^2}{\alpha^2} \notag
\end{align},
$$



And


$$
\mathcal{I}_{\alpha\beta} = \mathcal{I}_{\beta\alpha}= \frac{\gamma -1}{\alpha}, \notag
$$




$$
\mathcal{I}_{\beta\beta} =\frac{1}{\beta^2}\left[\frac{\pi^2}{6} + (1-\gamma)^2\right], \notag
$$



where $\gamma \approx 0.577$ is Euler's constant.



The covariance matrix is



$$
\Sigma = \mathcal{I}^{-1} = \frac{6 \alpha^2}{\pi^2}\begin{bmatrix}
\mathcal{I}_{\beta\beta} & -\mathcal{I}_{\alpha_\beta}\\
-\mathcal{I}_{\alpha\beta} & \mathcal{I}_{\alpha\alpha}
\end{bmatrix}. \notag
$$



The correlation coefficient between estiamted $\alpha$ and $\beta$ is



$$
\rho = \frac{-\mathcal{I}_{\alpha\beta}}{\sqrt{\mathcal{I}_{\alpha\alpha}\mathcal{I}_{\beta\beta}}} = \frac{1-\gamma}{\sqrt{\frac{\pi^2}{6}+(1-\gamma)^2}} = 0.31. \label{eqn_cor}
$$
