---

title: "Derivation and Comparison of Log-rank, Score, and Likelihood Ratio Tests in Survival Analysis"
date: 2026-01-18
typora-root-url: ./..
---



When ranking the reliability of two groups based on survival data, the statistical significance of any observed difference can be assessed using several methods:

* **The Log-rank test:** A non-parametric test based on the difference between observed and expected failures.
* **The Cox Proportional Hazards Model:** Methods utilizing the raw likelihood ratio test or the variation of the score function (the first derivative of the partial log-likelihood).

This note derives the test statistics for these three methods and compares theoretical results with statistical software output using numerical data.



### Theoretical Derivations



#### Survival Data and Contigency Table

We compare Group 0 and Group 1 at various times $t_j$ using the following contingency table:

|         |        Failures         | Survivors         |     Number at Risk      |
| :-----: | :---------------------: | ----------------- | :---------------------: |
| Group 0 |        $d_{0j}$         | $n_{0j} - d_{0j}$ |        $n_{0j}$         |
| Group 1 |        $d_{1j}$         | $n_{1j}-d_{1j}$   |        $n_{1j}$         |
|  Total  | $d_j = d_{01} + d_{1j}$ | $n_j - d_j$       | $n_j = n_{0j} + n_{1j}$ |



#### Null Hypothesis

The null hypothesis ($H_0$) is that the two groups are statistically identical. 




#### Deriving the Log-rank Test

Under the null hypothesis ($H_0$), the observed failures $O_{01}=d_{0j}$ and $O_{1j}=d_{1j}$ follow a hypergeometric distribution. 

##### Expectation and Variance

The expected number of failures,  $d_{0j}$ and $d_{1j}$ are:


$$
E_{0j} = n_{0j}\frac{d_j}{n_j}, \quad E_{1j} = n_{1j}\frac{d_j}{n_j}. \notag
$$


The variance of $d_{0j}$ and $d_{1j}$ is:



$$
V_{j} = n_{1j}\frac{d_j}{n_j}\frac{n_j - d_j}{n_j}\frac{n_j-n_{1j}}{n_j -1} = \frac{d_j}{n_j}\frac{n_j - d_j}{n_j}\frac{n_{0j}n_{1j}}{n_j -1} \notag
$$

##### Test Statistic

Let $X_{ij} = O_{ij} - E_{ij}$ (where $O_{ij} = d_{ij}$) represent the difference between observed and expected failures. The sum over all times is $X_i = \sum_j X_{ij}$. Because failures at different time steps are independent, the variance of the total sum is $\text{Var}(X_i) = \sum_j V_j$.

The resulting test statistic $\lambda_{\text{Log-rank}}$ follows a chi-square distribution with 1 degree of freedom ($\chi_1^2$):


$$
\lambda_\text{Log-rank} = \frac{X_i^2}{\text{Var}(X_i)}= \frac{\left(\sum_j (O_{ij} - E_{ij})\right)^2}{\sum_j V_j} \sim \chi_1^2. \notag
$$




### Connection to Cox Proportional Hazards Model 

In the Cox the null hypothesis is equivalent to $\beta = 0$, where $\beta$ is the parameter of the model. 



#### Partial Likelihood



Partial likelihood  is defined as



$$
L(\beta) = \prod_{j=1}^{K}\frac{\exp\left(\beta \sum\limits_{i \in D_j}^{} x_i\right)}{\left[ \sum\limits_{l \in R(t_j)}\exp(\beta x_l)\right]^{d_j}}. \notag
$$


where

- $x_i$: the group the failure belongs to, 0 or 1.
- $d_j$: total number of failures at time $t_j$.
- $D_j$:  the set of individuals who fail at time $t_j$.
- $R(t_j)$: risk set at time $t_j$ (all individuals alive and not censored just before $t_j$).



The partial log-likelihood is 



$$
\ell(\beta) = \sum_{j=1}^K\left[ \beta\sum\limits_{i\in D_j}x_i -d_j \log\left(\sum\limits_{l \in R(t_j)}\exp(\beta x_l)\right)\right]. \notag
$$



#### Score and Information Functions

The score function is defined as the first derivative of $\ell(\beta)$:


$$
U(\beta) =\frac{\partial \ell(\beta)}{\partial \beta}.\notag
$$


Information $I(\beta)$ is the negative second derivative of $\ell(\beta)$:


$$
I(\beta) = -\frac{\partial^2 \ell(\beta)}{\partial \beta^2} = -\frac{\partial U(\beta)}{\partial \beta}. \notag
$$


#### Deriving the Likelihood Ratio Test

The Likelihood Ratio Test compares the goodness-of-fit between two models: the null model where $\beta=0$ and the alternative model where $\beta \neq 0$.

The log likelihood ratio is defined as 



$$
\lambda_{\text{Likelihood-ratio}} = -2 \log\left[\frac{\sup_{\beta=0} L(\beta)}{\sup_{\beta\neq 0} L(\beta)} \right] \notag
$$



By Wilk's Theorem,   $\lambda_\text{Likelihood-ratio}$ follows a chi-square distribution with one degree of freedom ($\chi^2_1$).

And by definition, 



$$
\log \left[\sup_{\beta=0} L(\beta)\right] = \log L(0)= \ell(0) \notag
$$



and 


$$
\log \left[\sup_{\beta\neq 0} L(\beta)\right] = \ell(\hat{\beta}), \notag
$$




where $\hat{\beta}$ is the maximum likelihood estimate of $\beta$.  

Using the properties of logarithms and the second-order Taylor expansion to approximate $\ell(\beta)$ around the null hypothesis value ($\beta=0$), the statistic becomes



$$
\begin{align} \notag
\lambda_\text{Likelihood-ratio} & = 2\left[\ell(\hat{\beta})-\ell(0)\right] \\ \notag
& \approx 2 \left[ \ell(0) + \frac{\partial \ell(\beta)}{\partial \beta}\bigg\rvert_{\beta=0} \hat{\beta} +\frac{1}{2}\frac{\partial^2 \ell(\beta)}{\partial \beta^2}\bigg\rvert_{\beta=0} -\ell(0)\right] \\  \label{llr_expand}
& = 2\left[U(0) \hat{\beta} -\frac{1}{2} I(0) \hat{\beta}^2\right] 
\end{align}
$$



At  $\hat{\beta}$,   the first derivative of the log likelihood is zeor: $U(\hat{\beta})=\frac{\partial \ell(\beta)}{\partial \beta}=0$. Expand $U(\hat{\beta})$ around $\beta=0$: 



$$
U(\hat{\beta}) \approx U(0) + \frac{\partial U(\beta)}{\partial \beta}\bigg\rvert_{\beta=0}\hat{\beta} = U(0) - I(0) \hat{\beta} = 0 \notag
$$



We have



$$
\hat{\beta} = \frac{U(0)}{I(0)}.\label{beta_hat}
$$


Plug ($\ref{beta_hat}$) into ($\ref{llr_expand}$),  we have the statistic $\lambda_\text{Likelihood-ratio}$:



$$
\lambda_\text{Likelihood-ratio} = \frac{U^2(0)}{I(0)} \sim \chi_1^2.\notag
$$



#### The Score Function and Its Variance

Another way to test the null hypothesis that $\beta = 0$ is to evaluate the score function and its variance. 

This method relies on two properties of the score function, $U(\beta)= \frac{\partial \ell(\beta)}{\partial \beta}$:

1. The expected value of the score function is zero at the true parameter value.
2. The variance of the score function is equal to the Expected Fisher Information.

We will show in the following that if the true value of $\beta = \beta_0 = 0$, mean of the score function at $\beta = \beta_0$ is zero; the observed $U(\beta)$ approximately follows a normal distribution; the variance of $U(0)$ and the p-value can be calculated.

To show that the mean of the score function is zero when $\beta$ is at its true value ($\beta_0$), we calculate the expectation over the data distribution:


$$
\begin{align} \notag
E_{\beta_0}(U(\beta_0)) &= \int U(\beta_0) L(x\mid \beta_0)dx \\ \notag
& = \int \frac{\partial \ell }{\partial \beta} L(x\mid \beta_0) dx \\ \notag
& = \int \frac{\partial L}{\partial \beta}\frac{L}{L} d x \\ \notag
& = \int \frac{\partial L}{\partial \beta} dx \\ \notag
& = \frac{\partial}{\partial \beta}\int L dx \\ \notag
& = \frac{\partial 1}{\partial \beta} \\\notag
& = 0, \notag
\end{align} \notag
$$




where $E_{\beta_0}$ is the average over the data given the parameter $\beta=\beta_0$, and $x$ represents the data.



The variance of $U(\beta)$ is defined as 


$$
\text{Var}(U(\beta_0)) = E_{\beta_0}\left[\left(U(\beta_0)-E_{\beta_0}(U(\beta_0)\right)^2)\right] = E_{\beta_0}\left[U(\beta_0)^2\right] \notag
$$


Since the mean is zero, this simplifies to:


$$
\text{Var}(U(\beta_0)) =E_{\beta_0}\left[U(\beta_0)^2\right]. \notag
$$


To relate this to the second derivative of the log-likelihood, we differentiate the identity $E_\beta[U(\theta)] = 0 $  with respect to $\beta$:





$$
\frac{\partial }{\partial \beta} \int \frac{\partial \ln L}{\partial \beta} L dx = 0. \notag
$$



Applying the product rule to the terms inside the integral:


$$
\int \frac{\partial^2 \ln L}{\partial \beta^2} L dx +\int\frac{\partial \ln L}{\partial \beta}\frac{\partial L}{\partial \beta} dx = 0 \notag
$$



We can rewrite the second term using the identity $\frac{\partial L}{\partial \beta} =L \frac{\partial \ln L}{\partial \beta}$:



$$
E_\beta\left[\frac{\partial U(\beta)}{\partial \beta}\right] + E_\beta\left[U(\beta)^2\right] = 0. \notag
$$



Therefore, the variance of the score is:


$$
\text{Var}(U(\beta_0)) = E_{\beta_0}\left[U(\beta_0)^2\right] = -E_{\beta_0}\left[\frac{\partial U(\beta)}{\partial \beta}\right]\bigg\rvert_{\beta_0}. \notag
$$


The observed score $U(0)$ is from an approximately normal distribution


$$
U(0) \sim \mathcal{N}(0, \text{Var}(U(0))). \notag
$$


To test the null hypothesis, we construct a chi-square statistic with one degree of freedom by squaring the standardized score:


$$
\lambda_\text{Score} = \frac{U^2(0)}{\text{Var}(U(0))} \sim \chi_1^2. \notag
$$




#### Calculations of $U(0)$,  Variance of $U(0)$, and $I(0)$ From Survival Data



For the Cox proportional hazards model, the score is 



$$
U(\beta) =\frac{\partial \ell}{\partial \beta} = \sum_{j=1}^K\left[\sum\limits_{i\in D_j}x_i - d_j\frac{\sum\limits_{l\in R(t_j)}x_l \exp(\beta x_l)}{\sum\limits_{l \in R(t_j)}\exp(\beta x_l)}\right] \notag
$$



To evaluate the score under the null hypothesis ($H_0: \beta=0$), we substitute $\beta=0$ into the equation. For a binary covariate where $x\in \{0,1\}$:

- $\sum_{l\in R(t_j)}\exp(0)= n_j$ (the total number at risk)
- $\sum_{l\in R(t_j)}x_l \exp(0) = h_{1j}$ ( the number at risk in Group 1)
- $\sum_{i\in D_j}x_i = d_{1j}$  (the observed failures in Group 1)



The score at $\beta=0$ simplifies to:



$$
U(0)  =\sum_{j=1}^K \left[d_{1j}- d_j\frac{n_{1j}}{n_j} \right] 
 = \sum_{j=1}^K (O_j - E_j). \notag
$$



The Fisher Information is the negative second derivative of the log-likelihood. To find $I(0)$, we calculate:


$$
\begin{align} \notag
\frac{\partial U(\beta)}{\partial \beta}\bigg\rvert_{\beta=0} & = - \sum_{j=1}^K d_j \frac{\left(\sum\limits_{l\in R(t_j)}x_l^2\exp(\beta x_l)\right)\left(\sum\limits_{l \in R(t_j)}\exp(\beta x_l)\right) - \left(\sum\limits_{l\in R(t_j)}x_l \exp(\beta x_l)\right)^2}{\left(\sum\limits_{l \in R(t_j)}\exp(\beta x_l)\right)^2}\Bigg\rvert_{\beta=0} \\ \notag
& = -\sum_{j = 1}^K d_j\left[\frac{n_{1j}}{n_j} -\left(\frac{n_{1j}}{n_j}\right)^2\right] \\ \notag
& = -\sum_{j=1}^K d_j \left(\frac{n_{1j}}{n_j}\right)\left(\frac{n_j - n_{1j}}{n_j}\right) \\ \notag
& = - \sum_{j=1}^K d_j \left(\frac{n_{0j}}{n_j}\right)\left(\frac{n_{1j}}{n_j}\right) \notag
\end{align}.\notag
$$



At $\beta=0$

- $\sum\limits_{l\in R(t_j)}x_l\exp(\beta x_l)=n_{1j}$
- $\sum\limits_{l\in R(t_j)}x_l^2\exp(\beta x_l)=n_{1j}$
- $\sum\limits_{l \in R(t_j)}\exp(\beta x_l)=n_j$
- When multiple failures occur at time $t_j$, all the ties have the same risk set. This is called the  Berslow method.




$$
\begin{align} \notag
\text{Var}(U(0)) & =-E_{\beta}\left[\frac{\partial U(\beta)}{\partial \beta}\bigg\rvert_{\beta=0}\right] \\ \notag
& \approx - \frac{\partial U(\beta)}{\partial \beta}\bigg\rvert_{\beta=0} \\ \notag
& = \sum_{j=1}^K d_j \left(\frac{n_{1j}}{n_j}\right)\left(\frac{n_j - n_{1j}}{n_j}\right) \\ \notag
& =  \sum_{j=1}^K d_j \left(\frac{n_{0j}}{n_j}\right)\left(\frac{n_{1j}}{n_j}\right) \notag
\end{align} \notag
$$



$E_\beta\left[\beta\frac{\partial U(\beta)}{\partial \beta}\bigg\rvert_{\beta=0}\right] \approx \frac{\partial U(\beta)}{\partial \beta}\bigg\rvert_{\beta=0}$ because $\frac{\partial U(\beta)}{\partial \beta}\bigg\rvert_{\beta=0}$ is the sample estimate that is used to approximate the mean over the data distribution.



We use the observed information $I(0)$ as a sample esitmate of the variance:


$$
I(0) = - \frac{\partial U(\beta)}{\partial \beta}\bigg\rvert_{\beta=0} \approx \text{Var}(U(0)) = \sum_j W_j,\notag
$$


where 
$$
W_j = d_j \left(\frac{n_{0j}}{n_j}\right)\left(\frac{n_{1j}}{n_j}\right). \notag
$$




The table below summarizes the statistics of the three methods to test for the difference between the two groups. Note that while the numerator $U^2(0)$ is identical for all three, the denominators differ slightly between the log-rank test and the Cox-based models.



| **Test Method**           | **Statistic ($\chi_1^2$)**                                   | **Denominator Logic**                                        |
| ------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Log-rank Test**         | $\lambda_{\text{Log-rank}} = \frac{U^2(0)}{\sum_j V_j}$      | Hypergeometric Variance: $V_j = \frac{d_j(n_j-d_j)n_{0j}n_{1j}}{n_j^2(n_j-1)}$ |
| **Likelihood Ratio Test** | $\lambda_{\text{Likelihood-ratio}} = \frac{U^2(0)}{\sum_j W_j}$ | Observed Information: $W_j = \frac{d_j n_{0j} n_{1j}}{n_j^2}$ |
| **Score Test**            | $\lambda_{\text{Score}} = \frac{U^2(0)}{\sum_j W_j}$         | Equivalent to the Likelihood Ratio test at $\beta=0$         |



The Cox model-based tests yield identical results when using the Breslow approximation for ties. The log-rank test uses a slightly more conservative variance ($V_j$) derived from the hypergeometric distribution. 



By calculating these statistics and comparing them against the chi-square distribution, we can determine the p-value and decide whether to reject the null hypothesis of no difference between the groups.





### Results

This section summarizes the results of the survival analysis comparing two groups. We provide the theoretical formulas, a step-by-step numerical calculation using a sample dataset, and verification using statistical software.



Both the Log-rank and Likelihood Ratio tests follow a chi-square distribution with one degree of freedom ($\chi_1^2$) under the null hypothesis.



|         Test          |                          Statistic                           |                   P-value                   |
| :-------------------: | :----------------------------------------------------------: | :-----------------------------------------: |
|     Log-rank test     | $$\lambda_\text{Log-rank} = \frac{U^2(0)}{\sum_j V_j}\sim \chi_1^2$$ |     $ P(x \geq\lambda_\text{Log-rank})$     |
| Likelihood ratio test | $$\lambda_\text{Likelihood-ratio} = \frac{U^2(0)}{I(0)}= \frac{U^2(0)}{\sum_j W_j} \sim \chi_1^2$$ | $ P(x \geq\lambda_\text{Likelihood-ratio})$ |



The components of the statistics are calculated based on the failures and risk sets at each distinct event time $j$.

|     Parameter     |                           Formula                            |
| :---------------: | :----------------------------------------------------------: |
|      $U(0)$       | $$\sum_{j=1} \left[d_{ij}- d_j\frac{n_{ij}}{n_j} \right]$$, $i=0, 1$ |
|   $\sum_j V_j$    | $$\sum_j \frac{d_j}{n_j}\frac{n_j - d_j}{n_j}\frac{n_{0j}n_{1j}}{n_j -1}$$ |
| $I(0)=\sum_j W_j$ | $$\sum_{j=1} d_j \left(\frac{n_{0j}}{n_j}\right)\left(\frac{n_{1j}}{n_j}\right)$$ |



#### Numerical Calculation



To compare the methods, we generated survival data for two groups ($x=0$ and $x=1$) with four subjects each.

| **Subject** | **Group ($x$)** | **Time ($t$)** | **Status**   |
| ----------- | --------------- | -------------- | ------------ |
| 1           | 0               | 2              | 1 (Event)    |
| 2           | 0               | 4              | 1 (Event)    |
| 3           | 0               | 4              | 1 (Event)    |
| 4           | 0               | 7              | 0 (Censored) |
| 5           | 1               | 3              | 1 (Event)    |
| 6           | 1               | 4              | 1 (Event)    |
| 7           | 1               | 6              | 1 (Event)    |
| 8           | 1               | 8              | 0 (Censored) |



There are four distinct event times in the dataset: $t=\{2, 3, 4, 6\}$. The counts at these event times are as follows.

| **Time ($t_j$)** | **Group 0 Risk ($n_{0j}$)** | **Group 1 Risk ($n_{1j}$)** | **Total Risk ($n_j$)** | **Group 0 Events ($O_{0j}$)** | **Group 1 Events ($O_{1j}$)** | **Total Events ($d_j$)** |
| ---------------- | --------------------------- | --------------------------- | ---------------------- | ----------------------------- | ----------------------------- | ------------------------ |
| **2**            | 4                           | 4                           | 8                      | 1                             | 0                             | 1                        |
| **3**            | 3                           | 4                           | 7                      | 0                             | 1                             | 1                        |
| **4**            | 3                           | 3                           | 6                      | 2                             | 1                             | 3                        |
| **6**            | 1                           | 2                           | 3                      | 0                             | 1                             | 1                        |



#### Theoretical Calculations

Using the counts from the event-time breakdown, we calculate the observed and expected failures along with the variance components. 

| Time ($t_j$) | **Observed ($O_{0j}$)** | **Observed ($O_{1j}$)** | Expectation ($E_{0j}$) | Expectation ($E_{1j}$) | $V_j$ Log-rank            | $W_j$ Likelihood Ratio  |
| ------------ | ----------------------- | ----------------------- | ---------------------- | ---------------------- | ------------------------- | ----------------------- |
| 2            | 1                       | 0                       | 1/2                    | 1/2                    | 1/4                       | 1/4                     |
| 3            | 0                       | 1                       | 3/7                    | 4/7                    | 12/49                     | 12/49                   |
| 4            | 2                       | 1                       | 3/2                    | 3/2                    | 9/20                      | 3/4                     |
| 6            | 0                       | 1                       | 1/3                    | 2/3                    | 2/9                       | 2/9                     |
| Total        | 3                       | 3                       | $\frac{58}{21}=2.7619$ | $\frac{68}{21}=3.2381$ | $\frac{5147}{4410}=1.167$ | $\frac{647}{441}=1.467$ |



##### Log-rank Test Result

$$
\lambda_{\text{Log-rank}} = \frac{\left(\sum_j \left(O_{1j}-E_{1j}\right)\right)^2}{\sum_j V_j} = \frac{\left(\sum_j \left(O_{0j}-E_{0j}\right)\right)^2}{\sum_j V_j} = \frac{0.2381^2}{1.167} = 0.0486. \notag
$$



The p-value = 0.8255.



##### Likelihood Ratio Test Result


$$
\lambda_{\text{Likelihood-ratio}} = \frac{\left(\sum_j \left(O_{1j}-E_{1j}\right)\right)^2}{\sum_j W_j} = \frac{\left(\sum_j \left(O_{0j}-E_{0j}\right)\right)^2}{\sum_j W_j} = \frac{0.2381^2}{1.467} = 0.0386 \notag
$$


The p-value = 0.8442.



#### Statistical Software Verification

We verified these calculations using the `survival` package in R.



```R
library(survival)
data <- read_csv("data.csv", show_col_types = FALSE)
data
```





##### Log-rank Test with `survdiff` Function

```R
with(data, survdiff(Surv(Time, Status)~ Group))
```

|         |  N   | Observed | Expected | (O-E)^2/E | (O-E)^2/V |
| :-----: | :--: | :------: | :------: | :-------: | :-------: |
| Group=0 |  4   |    3     |   2.76   |  0.0205   |  0.0486   |
| Group=1 |  4   |    3     |   3.24   |  0.0175   |  0.0486   |

 Chisq= 0.04857  on 1 degrees of freedom, p= 0.8256.



##### Lieklihood Ratio Test with `coxph` Function

```R
cox.model <- with(data, coxph(Surv(Time, Status) ~ Group,ties = 'breslow'))

summary(cox.model)$logtest
```



|       |  coef  | exp(coef) | se(coef) |     z     |    p     |
| ----- | :----: | :-------: | :------: | :-------: | :------: |
| Group | -0.162 |   0.851   |  0.823   | -0.196366 | 0.844324 |

Likelihood ratio test=0.0385  on 1 df,  p=0.844,  n= 8, number of events= 6. 



The theoretical derivations and software calculations align almost perfectly.

- The **Log-rank test** produced a $\chi^2$ of **0.0486** ($p=0.8256$).
- The **Likelihood Ratio test** produced a $\chi^2$ of **0.0385** ($p=0.844$).



### Conclusion

This analysis demonstrates the mathematical equivalence between the non-parametric log-rank test and the likelihood test of the Cox Proportional Hazards model. 



The difference between the two tests arises solely from the denominator. The Log-rank test uses a variance estimate derived from the hypergeometric distribution, while the likelihood ratio test uses the Fisher Information estimated via the Breslow method.



While both tests yield nearly identical results in practice, they differ significantly in conceptual straightforwardness.



The likelihood ratio and score tests are conceptually easier to derive. They rely on standard calculus. In contrast, the log-rank test relies on counting progress. Proving its asymptotic properties requires the Martingale Central Limit Theorem. This framework is significantly more complex. Finally, the likelihood framework is more flexible, allowing for easy expansion into multivariate models. 





#### References

Likelihood ratio test. In Wikipedia.  https://en.wikipedia.org/wiki/Likelihood-ratio_test

Fisher Information.  In Wikipedia. https://en.wikipedia.org/wiki/Fisher_information

Wilk's Theorem.  In Wikipedia.  https://en.wikipedia.org/wiki/Wilks%27_theorem





