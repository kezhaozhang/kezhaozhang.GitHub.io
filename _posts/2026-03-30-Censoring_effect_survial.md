---

title: "High Censoring Rates Mask Lifetime Differences"
date: 2026-03-20
typora-root-url: ./..
---



In this study, we use simulations to assess how the censoring of survival data affects the ability to detect lifetime differences between groups. The results demonstrate that when the censoring rate is high (e.g., over 98%), the likelihood of detecting even substantial differences becomes extremely low.



### Introduction

In reliability experiments comparing two designs, failure rates are sometimes very low. Consequently, statistical tests may show no significant difference in lifetime between designs, even if one is objectively superior. This raises a critical question: if two groups are intrinsically different, what is the likelihood that this difference can be detected at a given observed failure rate?



The failure rate in a survival experiment is the complement of the censoring rate. When an experiment is stopped early, samples with a lifetime longer than the test duration are considered censored. The censoring rate—the proportion of samples still alive when the experiment ends—is the complement of the failure rate.



To answer our question, we generated random samples from Weibull distributions for two groups with different lifetimes. We then censored the resulting data at various rates and assessed the statistical differences using the log-rank test. By repeating this process multiple times, we collected the p-value distributions to determine the detection rate for each scenario.



### Results



First, we assume the lifetimes of two designs follow Weibull distributions. The probability density function (PDF) is characterized by the shape parameter $\beta$ and scale parameter $\alpha$:


$$
f (t; \alpha, \beta)= \frac{\beta}{\alpha}\left(\frac{t}{\alpha} \right)^{\beta-1} e^{-\left(t/\alpha\right)^\beta}. \notag
$$


In our first simulation, the lifetime of Group 2 is 5 times of that of Group 1 due to a 5-fold difference in their scale parameters ($\alpha$).

| Group | Shape Parameter $\beta$ | Scale Parameter $\alpha$ |
| :---: | :---------------------: | :----------------------: |
|   1   |           1.5           |           100            |
|   2   |           1.5           |           500            |



The Weibull plot in Figure 1 shows the intrinsic lifetime distributions. As intended, Group 2 demonstrates a significantly longer lifetime than Group 1.





<figure>
  <center>
  <img src="/assets/images/weibplot_5x_full_data_censor_lines.svg" width="550">
   </center>
  <center>
    <figcaption> 
      Figure 1. Weibull probability plot comparing the intrinsic lifetime distributions of Group 1 and Group 2. The groups share a common shape parameter (&beta;=1.5), with Group 2 exhibiting a 5-fold increase in the scale parameter (&alpha;) over Group 1. 
    </figcaption>
  </center>
</figure>



We then censored the data at two different limits, indicated by the vertical lines A and B in Figure 1.



Figure 2 shows the Kaplan-Meier (KM) survival plots for the two cases. In Scenario A, the censoring limit is low (resulting in a 0.98 censoring rate for the shorter-lived group) The log-rank test fails to find a significant difference  ($p=0.65$). In Scenario B, the censor limit is higher (0.94 censoring rate) and the log-rank test successfully identifies a statistically significnat difference ($p=0.012$).  This illusttrates how heavy censoring strip away information and increases uncertainty. 

<figure>
  <center>
  <img src="/assets/images/survival_plots_censoring_A_and_b.svg" width="650">
   </center>
  <center>
    <figcaption> 
      Figure 2. Kaplan-Meier survival plots with different censoring: (A) 0.98 censoring rate; (B) 0.94 censoring rate. <it>p</it> represents the log-rank test p-value.
    </figcaption>
  </center>
</figure>



The result above represents a single simulation run. To find a trend, we repeated the process multiple times across various censoring rates. We defined the detection rate as the proportion of runs where the log-rank test yielded a $p<0.05$.



Figure 3 shows that the detection rate drops precipitously once the censoring rate exceeds 0.9. At a 0.99 censoring rate, the probability of detecting a 5x lifetime difference falls below 15%.



<figure>
  <center>
  <img src="/assets/images/weib_5x_detect_vs_censor.svg" width="500">
   </center>
  <center>
    <figcaption> 
      Figure 3. Detection rate vs. Censoring rate for a 5x lifetime difference.
    </figcaption>
  </center>
</figure>



We also calculated the detection rate for various intrinsic lifetime factors (the ratio between group lifetimes), as shown in Figure 4. The uncertainty effect of censoring is even more pronounced for smaller intrinsic differences. Even with a 20x intrinsic difference, the probability of detection is extremely low at a 0.99 censoring rate. We included a baseline (Lifetime Factor = 1) which correctly shows a 5% false-positive rate across all censoring levels, consistent with result of [this log post]( https://kezhaozhang.github.io/2026/01/23/pairwise-survival-comparison.html) where the sample size effect was evaluated.



<figure>
  <center>
  <img src="/assets/images/weib_5x_detect_vs_censor_various_factors.svg" width="500">
   </center>
  <center>
    <figcaption> 
      Figure 4. Detection rate versus censoring rate for various intrinsic lifetime differences
    </figcaption>
  </center>
</figure>



### Conclusion

Censoring survival data reduces available information and increases the uncertainty involved in detecting differences between groups. The detection rate decreases rapidly as the censoring rate increases; at extreme levels, even massive intrinsic differences become invisible.

These results suggest that for reliability experiments to be effective, tests must run long enough to achieve a failure rate of at least 3–6% in the baseline group, depending on the expected difference between the designs. Stopping an experiment too early doesn't just save time—it effectively guarantees a no significant difference result.



