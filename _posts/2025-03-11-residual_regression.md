---

title: "Estimating True Functional Dependencies Between Observed Variables with Known Causal Relationships"
date: 2025-03-14
typora-root-url: ./..
---



This post presents a method to estimate the true functional dependencies between observed variables, assuming the causal relationships between them are known. Using a two-step regression approach, we show how to isolate and quantify the sensitivity of a target variable to its predictors, even when these predictors are interdependent. We demonstrate this approach through three models that vary in the complexity of their functional relationships, providing both linear and nonlinear examples. The results indicate that the method is effective in recovering the true relationships, even in the presence of complex dependencies.



### Introduction



In multivariate regression, we often need to quantify the sensitivity of a target variable to its predictors, especially when those predictors are correlated. In such cases, standard regression methods may lead to misleading results. Here, we explore an approach to address this issue by assuming that the causal relationships between the predictors and the target variable are known, which can either be derived from domain knowledge or inferred from the data-generating process.  



#### Problem Statement

Consider the causal model shown in Figure 1, where $U$ and $V$ represent unobserved latent variables, and $X$, $Y$, and $Z$ are observed variables. The arrows indicate causal relationships, with the labels next to the arrows representing their respective strengths. The $\delta$ terms denote additive noise.



<figure>
  <center>
  <img src="/assets/images/residual_regression-diagram_model3.svg" width="300">
   </center>
  <center>
    <figcaption> Figure 1. Causal model of the data-generating process.
    </figcaption>
  </center>
</figure>



The relationships in Figure 1 can be expressed as


$$
\begin{align}\notag
X &= U + \delta_1 \\ 
Y & = V + f(U) +\delta_2 \\ \notag
Z &= g(U) + h(V) +\delta_3
\label{eqn_causal}
\end{align}
$$


where $f$, $g$, and $h$ are some functions. 



<figure>
  <center>
  <img src="/assets/images/residual_regression-diagram-xyz.svg" width="170">
   </center>
  <center>
    <figcaption> Figure 2. Causal relationships among <i>X</i>, <i>Y</i>, and <i>Z</i>.
    </figcaption>
  </center>
</figure>



As a result, the causal relationship among the observed variables is illustrated in Figure 2.



The question we aim to answer is: **Given the causal relationship in Figure 1, how can we estimate $g(U)$ and $h(V)$ from the observed values of $X$, $Y$, and $Z$?**



#### Method

To estimate the sensitivities of $Z$ with respect to $U$ and $V$, we propose a two-step regression method: 

Step 1. Regress $Y$ on $X$ and obtain the residual $Y_r$. This removes the influence of $X$ or $Y$, isolating the unique contribution of $V$.

Step 2. Regress $Z$ on $X$ and $Y_r$. The coefficients from this regression provide the sensitivity of $Z$ to $U$ and $V$. 

The sensitivity can be analyzed using tools like Accumulated Local Estimate (ALE) for both linear and nonlinear relationships.



### Results

We illustrate the proposed method with three models, each varying in the functional forms of $f$, $g$, and $h$, as listed in the table below. The goal is to find the functions $g$ and $h$ with the two-step regression.

|  Model  | $f(U)$ |  $g(U)$   | $h(V)$ |
| :-----: | :----: | :-------: | :----: |
| Model 1 | $3 U$  |   $2 U$   | $-3 V$ |
| Model 2 | $U^3$  |   $2 U$   | $-3 V$ |
| Model 3 | $U^3$  | $\sin(U)$ | $V^2$  |



#### Model 1

In this model, all functions are linear: $f(U) = 3 U$, $g(U)= 2 U$, and $h(V)=-3 V$.  The noise terms are set as  $\delta_1 = \delta_2 = 0$, and $\delta_3 \sim 0.25\mathcal{N}(0,1)$.  The causal diagram is shown in Figure 3. 





<figure>
  <center>
  <img src="/assets/images/residual_regression-diagram.svg" width="300">
   </center>
  <center>
    <figcaption> Figure 3. Causal diagram of Model 1.
    </figcaption>
  </center>
</figure>

The following Python generetes the data for Model 1.

```python
n = 500
d1 = 0.
d2 = 0.
d3 = 0.25
alpha = 3
gamma1 = 2
gamma2 = -3
np.random.seed(123)
U = stats.norm.rvs(size=n)
V = stats.norm.rvs(size=n)
noise1 = stats.norm.rvs(size=n) 
noise2 = stats.norm.rvs(size=n) 
noise3 = stats.norm.rvs(size=n) 
X = U + d1*noise1
Y = alpha*U + V + d2*noise2
Z = gamma1*U + gamma2*V + d3*noise3
```



The pairwise scatter plots of $X$, $Y$, and $Z$ is shown in Figure 4.



<figure>
  <center>
  <img src="/assets/images/residual_regression-pairplot_model1.svg" width="500">
   </center>
  <center>
    <figcaption> Figure 4. Pairwise scatter plots for Model 1.
    </figcaption>
  </center>
</figure>



Since the relationships are linear, ordinary least squares regression is used. The results are summarized in the table below. 



|                     |      Model       |    Slope $a$     |    Slope $b$    |
| :-----------------: | :--------------: | :--------------: | :-------------: |
| Original Parameters | $Z = a X + b Y$  | $10.96\pm 0.034$ | $-2.99\pm0.011$ |
|       Step 1        |     $Y=a X$      | $3.01\pm 0.045$  |                 |
|       Step 2        | $Z = aX + b Y_r$ | $1.96\pm 0.011$  | $-2.99\pm0.011$ |

The linear regression using the original parameters gives a slope $b$ of $-2.99$, which is very close to the true value of $-3$. This outcome is expected since $X$ is the confounder and is controlled for in the regression. As a result,  the coefficient for $Y$ (or $V$) is correct. However, the coefficient for $X$ (or $U$) differs significantly from the true value ($-10.96$ compared to $2$ ).



After removing the effect of $X$ on $Y$ in Step 1, the coefficients in Step 2 become accurate for both $X$ (or $U$) and $Y$ (or $V$).



#### Model 2



In this model, $f(U) = U^3$ introduces a nonlinear relationship between $X$ and $Y$. The causal diagram is shown in Figure 5.


$$
\begin{align}\notag
Y &= f(U) + V\\ \notag
&= U^3 + V. \notag
\end{align}
$$




<figure>
  <center>
  <img src="/assets/images/residual_regression-diagram_model2.svg" width="300">
   </center>
  <center>
    <figcaption> Figure 5. Causal diagram of Model 2. 
    </figcaption>
  </center>
</figure>



The following code generate the data from Model 2.

```python
n = 500
d1 = 0.
d2 = 0.
d3 = 0.25
alpha = 3
gamma1 = 2
gamma2 = -3
np.random.seed(123)
U = stats.norm.rvs(size=n)
V = stats.norm.rvs(size=n)
noise1 = stats.norm.rvs(size=n) 
noise2 = stats.norm.rvs(size=n) 
noise3 = stats.norm.rvs(size=n) 
X = U + d1*noise1
Y = U**3 + V + d2*noise2
Z = gamma1*U + gamma2*V + d3*noise3
```





<figure>
  <center>
  <img src="/assets/images/residual_regression-pairplot_model2.svg" width="500">
   </center>
  <center>
    <figcaption> Figure 6. Pairwise scatter plots for Model 2.
    </figcaption>
  </center>
</figure>



The scatter plots for Model 2 (Figure 6) show the nonlinear relationships between the variables. Due to the nonlinearity, we use a neural network with a single hidden layer for the regression. The ALE plots (Figure 7) indicate that the sensitivity of $Z$ to $Y$ is accurate, but the sensitivity to $X$ ($U$) is somewhat off.





<figure>
  <center>
  <img src="/assets/images/residual_regression-model2_Y~X+Y_ale_plots.svg" width="500">
   </center>
  <center>
    <figcaption> Figure 7. ALE plots of Z ~ X +Y model using an MLP regressor.
    </figcaption>
  </center>
</figure>



Next, the regression of $Y$ on $X$ (Figure 8) yields the residual $Y_r$. Regressing $Z$ on $X$ and $Y_r$ leads to the ALE plots in Figure 9, which show a much better match to the true sensitivities.



<figure>
  <center>
  <img src="/assets/images/residual_regression-model2_Y~X_regression.svg" width="500">
   </center>
  <center>
    <figcaption> Figure 8.  Y~X regression with an MLP regressor. Blue dots are observed data and the red line is the predicted values by the regression model.
    </figcaption>
  </center>
</figure>







<figure>
  <center>
  <img src="/assets/images/residual_regression-model2_Y~X+Y_r_ale_plots.svg" width="500">
   </center>
  <center>
    <figcaption> Figure 9. ALE plots of Z ~ X +Y<sub>r</sub> model using MLP regressor. Red lines represent the true dependence.
    </figcaption>
  </center>
</figure>



Finally, we use linear regression on $Z$ and the residual $Y_r$ (Figure 10), yielding results that are very close to the true values.



<figure>
  <center>
  <img src="/assets/images/residual_regression-model2_ols_Z~X+Yr.svg" width="400">
   </center>
  <center>
    <figcaption> Figure 10. Result of the linear regression of Z on X and Y<sub>r</sub>.
    </figcaption>
  </center>
</figure>



#### Model 3



In Model 3, the functions $f(U) = U^3$, $g(U) = \sin(U)$, and $h(V) = V^2$ introduce more complex nonlinearities. The causal diagram is shown in Figure 11.



<figure>
  <center>
  <img src="/assets/images/residual_regression-diagram_model3.svg" width="300">
   </center>
  <center>
    <figcaption> Figure 11. Causal diagram of Model 3.
    </figcaption>
  </center>
</figure>

The code below generates data and Figure 12 shows the pairwise scatter plots among $X$, $Y$, and $Z$. 

```python
n = 5000
d1 = 0.
d2 = 0.
d3 = 0.25
alpha = 3
gamma1 = 2
gamma2 = -3
np.random.seed(123)
U = stats.norm.rvs(size=n)
V = stats.norm.rvs(size=n)
noise1 = stats.norm.rvs(size=n) 
noise2 = stats.norm.rvs(size=n) 
noise3 = stats.norm.rvs(size=n) 
X = U + d1*noise1
Y = U**3 + V + d2*noise2
Z = np.sin(U) + V**2 + d3*noise3
```



<figure>
  <center>
  <img src="/assets/images/residual_regression-pairplot_model3.svg" width="500">
   </center>
  <center>
    <figcaption> Figure 12. Pairwise scatter plots for Model 3.
    </figcaption>
  </center>
</figure>

The regression of ZZ on XX and YY using a neural network regressor reveals that the sensitivities of ZZ to XX and YYdiffer significantly from the true values, despite showing some resemblance to them.

<figure>
  <center>
  <img src="/assets/images/residual_regression-model3_Y~X+Y_ale_plots.svg" width="500">
   </center>
  <center>
    <figcaption> Figure 13. ALE plots of Z ~ X +Y model using MLP regressor.
    </figcaption>
  </center>
</figure>

After regressing $Y$ on $X$, the residual $Y_r$ is used to compute the final regression results (Figure 14), which closely match the true sensitivities.



<figure>
  <center>
  <img src="/assets/images/residual_regression-model3_Y~X+Y_r_ale_plots.svg" width="500">
   </center>
  <center>
    <figcaption> Figure 14. ALE plots of Z ~ X +Y<sub>r</sub> model using MLP regressor. Red lines represent the true dependence.
    </figcaption>
  </center>
</figure>



### Conclusion

This work demonstrates an effective method for estimating the true functional dependencies between observed variables when the causal structure is known. By using a two-step regression procedure, we can remove the effects of correlated predictors and accurately estimate the relationships between the target and predictor variables. This method is applicable to both linear and nonlinear models and can be further extended to more complex causal relationships.

