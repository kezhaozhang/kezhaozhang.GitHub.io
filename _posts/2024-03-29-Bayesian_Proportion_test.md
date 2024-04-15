---

title: "Comparison of Proportion Tests"
date: 2024-03-29
typora-root-url: ./..
---



This note compares several statistical methods for detecting differences in failure rates between two groups. We explore Fisher's exact test, the Chi-squared test, and a Bayesian Monte Carlo approach, focusing on their conceptual simplicity, visual interpretability, and insights into uncertainty.



Imagine having two sets of samples, each undergoing a pass-fail test. The observation is summarized in the contingency table below. Our goal is to determine whether these groups have significantly different failure rates.

|      | Group 1 | Group 2 |
| :--: | :-----: | :-----: |
| Pass |  $29$   |  $14$   |
| Fail | $6851$  | $4030$  |





### Statistical Tests



#### Fisher's Exact Test

This method computes the exact probability of observing our sample counts under the assumption of equal failure rates.



|      | Group 1 | Group 2 |
| :--: | :-----: | :-----: |
| Pass |   $a$   |   $c$   |
| Fail |   $b$   |   $d$   |



The probability for the observed contingency table is



$$
p = \frac{\left(\begin{array}{c} a+b\\b\end{array}\right)\left(\begin{array}{c} c+d\\c\end{array}\right)}{\left(\begin{array}{c} a+b+c+d\\a+c\end{array}\right)}=\frac{(a+b)!(a+c)!(b+d)!(c+d)!}{a! b! c! d! (a+b+c+d)!}.
$$



where $\left(\begin{array}{c} n\\m\end{array}\right)=\frac{n!}{m!(n-m)!}$ is the binomial coefficient, which is the number of ways to choose an unordered subset of $m$ elements from  $n$ elements.



In order to obtain the $p$-value, we need to add up the probabilities of the contingency tables that have counts that are more extreme. The following example shows how to calculate it using `scipy`. The $p$-value obtained is $0.64$, which implies that the failure rate is not statistically different between the two groups.

```python
from scipy.stats import fisher_exact
count1 = 29
nobs1 = 6880
count2 = 14
nobs2 = 4044

fisher_exact([[count1, nobs1-count1], [count2, nobs2-count2]], alternative='two-sided')
```

```python
SignificanceResult(statistic=1.218487394957983, pvalue=0.6360414756991858)
```



#### Chi-squared Test

Here, we derive the Chi-square test.  The contingency table is rewritten in terms of failure rates $p_1$, $p_2$, and the number of observations $n_1$, $n_2$,  for Group 1 and Group 2.



|      |    Group 1    |    Group 2    |
| :--: | :-----------: | :-----------: |
| Pass |   $n_1 p_1$   |   $n_2 p_2$   |
| Fail | $n_1 (1-p_1)$ | $n_2 (1-p_2)$ |



The null hypothesis assumes that both groups have the same proportion, which is the pooled proportion.



$$
\hat{p} = \frac{n_1 p_1 + n_2 p_2}{n_1 + n_2} \label{eqn:pooled_p}
$$



The contingency table showing the expected counts becomes

|      |      Group 1      |      Group 2      |
| ---- | :---------------: | :---------------: |
| Pass |   $n_1 \hat{p}$   |   $n_2 \hat{p}$   |
| Fail | $n_1 (1-\hat{p})$ | $n_2 (1-\hat{p})$ |



When the counts are sufficiently large, the sum of the squared difference between the observation and expectation, normalized by the expectation, is approximately from the Chi-squared distribution:



$$
\begin{align}\notag
\chi^2 & = \frac{(n_1 p_1 - n_1\hat{p})^2}{n_1\hat{p}} +\frac{(n_2 p_2 - n_2\hat{p})^2}{n_2 \hat{p}}\\ \notag
& +\frac{\left[n_1(1-p_1)-n_1(1-\hat{p})\right]^2}{n_1(1-\hat{p})} +\frac{\left[n_2 (1-p_2)-n_2(1-\hat{p})\right]^2}{n_2\hat{p}}\\ \label{eqn:chi-squared}.
\end{align}
$$



Combine Equations ($\ref{eqn:pooled_p}$) and ($\ref{eqn:chi-squared}$​), we have



$$
\chi^2 = (p_1-p_2)^2/\sqrt{\frac{2\hat{\sigma}^2}{\hat{n}}},\label{eqn:chi2}
$$



where



$$
\hat{\sigma}^2 = \hat{p} (1-\hat{p}),\notag
$$



and



$$
\hat{n} = \frac{2}{1/n_1 + 1/n_2}. \notag
$$



$\hat{n}$ is the harmonic mean of $n_1$ and $n_2$, and  is more heavily affected by the smaller value of $n_1$ and $n_2$. 



Assuming equal variance of both populations and estimating variance using pooled proportion $\hat{p}$​,    the variance of each subpopulation is 



$$
\sqrt{\frac{\hat{p} (1-\hat{p})}{\hat{n}}}. \notag
$$



The difference in proportion between the two subpopulations has a standard deviation, denoted as ${\sigma}'$. It's calculated as:



$$
{\sigma}' = \sqrt{\frac{2}{\hat{n}}
}\hat{\sigma}=\sqrt{2\frac{\hat{p} (1-\hat{p})}{\hat{n}}}=\sqrt{\hat{p} (1-\hat{p})\left(\frac{1}{n_1}+\frac{1}{n_2}\right)} \notag
$$



The $z$-score, denoted as $z$, is then calculated as:



$$
z = \frac{p_1 - p_2}{\sigma}'. \label{eqn:z-score}
$$



The $\chi^2$ statistic in Equation ($\ref{eqn:chi2}$) is the square of the $z$-score. We can determine the $p$-value by looking at the tail area of the normal distribution using this $z$-score.



To illustrate, using the the $z$-score in Equation ($\ref{eqn:z-score}$) and `chi2_contigency` function from `scipy`, we calculate a two-sided $p$-value. In this case, both methods yield a $p$-value of $0.544$, suggesting there's no significant difference in failure rate between the two groups.



```python
from scipy.stats import norm
p1 = count1/nobs1
p2 = count2/nobs2
# pooled mean
p = (count1 + count2)/(nobs1 + nobs2)
#pooled sigma
sigma = np.sqrt( p*(1-p)*(1/nobs1+1/nobs2))
z_score = (p1-p2)/sigma
pvalue = 2*(1-norm.cdf(z_score))
pvalue
```

```
0.5438118435936659
```



```python
from scipy.stats import chi2_contingency
chi2_contingency([[count1, count2],[nobs1-count1, nobs2-count2]], correction=False)
```

```python
Chi2ContingencyResult(statistic=0.36852047306165053, pvalue=0.543811843593666, dof=1, expected_freq=array([[  27.08165507,   15.91834493],[6852.91834493, 4028.08165507]]))
```



#### Bayesian Monte Carlo Markov Chain Method



The Bayesian approach estimates the posterior distribution of the failure rate from the observed data and a prior using Markov chain Monte Carlo (MCMC) sampling. The following code uses the `PyMC` package for Python to calculate the posterior distributions of the failure rate. Uninformative priors are used.



```python
import pymc as pm
model = pm.Model()

with model:
    p1 = pm.Uniform('p1', lower=0, upper=1)
    p2 = pm.Uniform('p2', lower=0, upper=1)
    obs1 = pm.Binomial('obs1', n=nobs1, p=p1, observed=count1)
    obs2 = pm.Binomial('obs2', n=nobs2, p=p2, observed=count2)

with model:
    samples = pm.sample()
```



<figure>
  <center>
  <img src="/assets/images/proptest_traces.svg" width="1200">
   </center>
  <center>
    <figcaption> Figure 1. Markov Chain Monte Carlo samples and the resulting failure rate distributions.
    </figcaption>
  </center>
</figure>




Figure 1 shows the MCMC samples and the posterior distributions of the failure rates.  In Figure 2, we compare the failure rates of two groups, $p_1$ and $p_2$. They have different average rates but overlap a lot, showing they might not be that different.

<figure>
  <center>
  <img src="/assets/images/protest_posterior_similar_sample_size.svg" width="600">
   </center>
  <center>
    <figcaption> Figure 2. Posterior distributions of failure rates <i>p<sub>1</sub></i> and <i>p<sub>2</sub></i>.
    </figcaption>
  </center>
</figure>



The distribution of the difference between the failure rates, $p_1 - p_2$, is shown in Figure 3. The difference is around $0$, it means there is no statistically significant difference between $p_1$ and $p_2$. 

<figure>
  <center>
  <img src="/assets/images/proptest_posterior_dist.svg" width="600">
   </center>
  <center>
    <figcaption> Figure 3. Distribution of the difference between the posteriors, <i>p<sub>1</sub></i>-<i>p<sub>2</sub></i>.
    </figcaption>
  </center>
</figure>



The MCMC method gives us more insights compared to Fisher's exact test or the Chi-squared test. It helps us understand the data better, especially when we change things like sample size.

For instance, if we reduce Group 2's sample size drastically, like by 7 times, we see in Figure 4 that the uncertainty in Group 2's failure rate (denoted as $p_2$) shoots up because of the smaller sample size.

|      | Group 1 | Group 2 |
| :--: | :-----: | :-----: |
| Pass |  $29$   |   $2$   |
| Fail | $6851$  |  $578$  |





<figure>
  <center>
  <img src="/assets/images/protest_posterior_small_sample_size.svg" width="600">
   </center>
  <center>
    <figcaption> Figure 4. Posterior distributions of the failure rates,  showing a wider distribution for <i>p<sub>2</sub></i> due to the reduced sample size.
    </figcaption>
  </center>
</figure>





### Conclusion

Using the Bayesian MCMC method for proportion tests is straightforward and simple to put into practice. It gives results that are easy to understand. Unlike the Chi-squared test and Fisher's exact test, this method also gives us insight into the uncertainty of all groups in the analysis.



### References

[Fisher's Exact Test](https://en.wikipedia.org/wiki/Fisher%27s_exact_test), Wikipedia.
