---


title: "Using Regression to Check Variable Dependence in Three Types of Directed Acyclic Graphs"
typora-root-url: ./..
---

This article explores how regression can be used to determine the dependence between variables in three types of directed acyclic graphs (DAGs): pipe, confounder, and collider. The theoretical analysis of these graphs can be found in the linked [blog post](https://kezhaozhang.github.io/2023/03/20/DAG.html).



## Pipe DAG



<figure>
  <center>
  <img src="/assets/images/dag_pipe.svg" width="300">
   </center>
  <center>
  </center>
</figure>

In the pipe DAG, variables $X$ and $Z$ are **independent** when conditioned on Y; mathematically, this can be expressed as 



$$
P(X,Z \mid Y) = P(X\mid Y) P(Z\mid Y).
$$



We verify this using regression by regressing Z on both X and Y. If the dependence of Z on X is flat, then X and Z are independent conditioned on Y.

We demonstrate this using a data set with non-linear dependencies and create three models: linear regression, gradient boosting, and GAM (generalized additive models). 

```python
n = 500
noise = 0.2
X = np.random.randn(n)
Y = X**2 + noise**np.random.rand(n)
Z = np.sin(Y/2) + noise*np.random.rand(n)

```



Figure 1. shows that without conditioning on $Y$, $X$ and $Z$ are dependent with a nonlinear relationship.


<figure>
  <center>
  <img src="/assets/images/dag_pipe_pairplor.svg" width="600">
   </center>
  <center>
    <figcaption> Figure 1. Pairwise correaltion plot among <i>X</i>, <i>Y</i>, and <i>Z</i>. It is clear that <i>X</i> and <i>Z</i> have nonlinear and nonmonotoic relationship.
    </figcaption>
  </center>
</figure>



```python
import numpy as np
from alibi.explainers import ALE, plot_ale
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from pygam import LinearGAM, s

models = [LinearRegression(), 
          GradientBoostingRegressor(learning_rate=0.001), 
          LinearGAM(s(0)+s(1), max_iter=1000, tol=0.0000001)]
[m.fit(np.stack((X,Y) , axis=1), Z) for m in models]
fig, axes = plt.subplots(3,2, sharey=True, figsize=(6, 10))
for description, m, ax in zip(['Linear Regression', 'GradientBoostingRegressor', 'GAM'], models, axes):
    ale = ALE(m.predict, feature_names=['X','Y'], target_names=['Z'])
    ale_exp = ale.explain(np.stack((X,Y), axis=1), min_bin_points=10)
    #fig, ax = plt.subplots()
    plot_ale(ale_exp, ax=ax, n_cols=2)
    ax[0].set_title(description);
```

The ALE (accumulated local effects) plots in Figure 1 reveal that X and Z are independent when Y is included in the regression in the case of non-linear models, while the linear model shows dependence of Z on X, as it cannot account for the nonlinearity in the data.

<figure>
  <center>
  <img src="/assets/images/dag_pipe_ale.svg" width="600">
   </center>
  <center>
    <figcaption> Figure 2. ALE plots of <i>Z</i>'s dependnce on <i>X</i> and <i>Y</i>, for three regression models: linear regression (top), Gradient Boosting Regressor (middle) and GAM (bottom). For the nonlinear regression models <i>X</i> and <i>Z</i> are independent, indicated by the flat line, whereas the linear model shows dependnce between <i>X</i> and <i>Z</i> because the model is incapable of handle the nonlineariaty in the data.
    </figcaption>
  </center>
</figure>





## Confounder DAG

<figure>
  <center>
  <img src="/assets/images/dag_confounder.svg" width="150">
   </center>
  <center>
  </center>
</figure>

In the DAG with a confounder, $X$ and $Y$ are **independent** when conditioned on $Z$;  mathematically, this is



$$
P(X,Y \mid Z) = P(X\mid Z) P(Y\mid Z),
$$



Again we generate some data. The relationships among $X$, $Y$ and $Z$ are linear.

```python
n = 500
noise = 0.2
Z = np.random.randn(n)
X = 3*Z + noise*np.random.randn(n)
Y = -Z + noise**np.random.rand(n)
```



Figure 3 shows that without conditioning on the confounder $Z$, $X$ and $Y$ are depenent.

<figure>
  <center>
  <img src="/assets/images/dag_confounder_pairplor.svg" width="600">
   </center>
  <center>
    <figcaption> Figure 3. Pairwise correaltion plot among <i>X</i>, <i>Y</i>, and <i>Z</i>. It is clear that <i>X</i> and <i>Z</i> have linear relationship.
    </figcaption>
  </center>
</figure>



We regress $Y$ on $X$ and $Z$. As shown in Figure 4, when conditioned on the confounder $Z$, $X$ and $Y$ are independent.

<figure>
  <center>
  <img src="/assets/images/dag_confounder_ale.svg" width="600">
   </center>
  <center>
    <figcaption> Figure 4. ALE plots of <i>Z</i>'s dependnce on <i>X</i> and <i>Y</i>, for three regression models: linear regression (top), Gradient Boosting Regressor (middle) and GAM (bottom). For all regression models <i>X</i> and <i>Z</i> are independent, indicated by the flat line.
    </figcaption>
  </center>
</figure>



## Collider DAG

<figure>
  <center>
  <img src="/assets/images/dag_collider.svg" width="150">
   </center>
  <center>
  </center>
</figure>
Finally, in the DAG with a collider, $X$ and $Y$ are **dependent** when conditioned on $Z$; mathematically expressed as 



$$
P(X,Y \mid Z) \neq P(X\mid Z) P(Y\mid Z).
$$



We generate some data with linear relationships among $X$, $Y$ and $Z$.

```python
n = 500
noise = 0.2
X = np.random.randn(n)
Y = np.random.randn(n)
Z = X + 2*Y + noise*np.random.randn(n)
```

Figure 5 shows that without condiitoing on the collider $Z$, $X$ and $Y$ are independent.

<figure>
  <center>
  <img src="/assets/images/dag_collider_pairplor.svg" width="600">
   </center>
  <center>
    <figcaption> Figure 5. Pairwise correaltion plot among <i>X</i>, <i>Y</i>, and <i>Z</i>. It is clear that <i>X</i> and <i>Y</i> are independent.
    </figcaption>
  </center>
</figure>

We regress $Y$ on $X$ and $Z$. As shown in Figure 6, when conditioned on the collider $Z$, $X$ and $Y$ are dependent.

<figure>
  <center>
  <img src="/assets/images/dag_collider_ale.svg" width="600">
   </center>
  <center>
    <figcaption> Figure 6. ALE plots of <i>Z</i>'s dependnce on <i>X</i> and <i>Y</i>, for three regression models: linear regression (top), Gradient Boosting Regressor (middle) and GAM (bottom). For all regression models <i>X</i> and <i>Y</i> are dependent, when conditioned on the collider <i>Z</i>.
    </figcaption>
  </center>
</figure>



In this note, regression models and ALE plots were used to analyze variable dependence in DAGs, with simulation results aligned with theoretical analysis.
