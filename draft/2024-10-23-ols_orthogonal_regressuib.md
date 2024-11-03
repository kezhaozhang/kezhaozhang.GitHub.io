---

title: "OLS and Orthogonal Linear Regression"
date: 2024-10-20
typora-root-url: ./..
---







### OLS and Orthogonal Linear Regression




$$
\begin{align}\notag
x &= v + \epsilon \\ \notag
y &= w\cdot v + \eta\\
\end{align}
$$


where $v$ is the true noiseless value, and


$$
\begin{align}\notag
\epsilon &\sim \mathcal{N}(0, \sigma_x^2)\\ \notag
\eta &\sim \mathcal{N}(0, \sigma_y^2)
\end{align}
$$



$$
\mathrm{Var}(v)=\sigma^2
$$


In a linear regression


$$
y = a x +b
$$




#### OLS


$$
\begin{align}
a & = \frac{\mathrm{COV}(x,y)}{\mathrm{Var}(x)}=\frac{w \sigma^2}{\sigma^2+\sigma_x^2}=\frac{w}{1+\left(\frac{\sigma_x}{\sigma}\right)^2} \\
b & = \langle y\rangle - a\langle x \rangle, \label{eqn:ols_b}
\end{align}
$$



where $\langle \rangle$ denotes mean. $\mathrm{COV}$ and $\mathrm{Var}$ are covariance and variance, respectively. 

Equation ($\ref{eqn:ols_b}$) means that the fitted line goes through the mean of the data.





Mean squared error
$$
\mathrm{MSE} =w^2 \sigma^2 +\sigma_y^2-\frac{w^2\sigma^4}{\sigma^2+\sigma_x^2}=\sigma^2\left[\left(\frac{\sigma_y}{\sigma}\right)^2+w^2\frac{\left(\frac{\sigma_x}{\sigma}\right)^2}{1+\left(\frac{\sigma_x}{\sigma}\right)^2}\right]
$$


R squared



#### Orthogonal Fit



The orthogonal fit minimizes the sum of the square distances of the data points to the line. 



Denote the data point by 


$$
\vec{P} = \left(\begin{array}{c}x\\y\end{array}\right) \notag
$$


and a unit vector $\hat{n}$


$$
\hat{n} = \left(\begin{array}{c} \alpha\\\beta\end{array}\right), \notag
$$

and


$$
\hat{n}\cdot\hat{n} = \alpha^2 + \beta^2=1. \label{eqn:n_unit_vector}
$$




The fitted line is defined by


$$
\vec{P}\cdot \hat{n}=b,
$$
where $b$ is constant.

The perpendicular distance from a data point $\vec{P}$ to the line is


$$
d = \mid \vec{P}\cdot \hat{n}-b\mid. \notag
$$

To find the paraemters, $\hat{n}$ and $b$ that define the fitted line, we minimize the Lagrangian



$$
\mathcal{L} = \frac{1}{N}\sum_{i=1}^N \mid\vec{P}_i\cdot\hat{n}-b\mid^2 + \lambda (\hat{n}\cdot\hat{n}-1),
$$



where the first term is the total squared distances, and the second term is the constraint that $\hat{n}$ is a unit vector.



First, we show that the orthogonal fitted line passes through the mean of the data. 


$$
\frac{\partial \mathcal{L}}{\partial b}= -\frac{2}{N}\sum_{i=1}^N(\vec{P_i}\cdot\hat{n}-b)=0\notag
$$
We have 
$$
b = \frac{1}{N}\sum_{i=1}^N \vec{P_i}\cdot\hat{n} = \langle\vec{P}\rangle\cdot\hat{n}. \label{eqn:or_mean}
$$
Therefore, the mean of the data points, $\langle \vec{P}\rangle$, is on the fitted line.



$\hat{n}$ statisfies


$$
\frac{\partial \mathcal{L}}{\partial \hat{n}}=0
$$

$$
\frac{1}{N}\sum_{i=1}^N (\vec{P_i}\cdot\hat{n} - b)\vec{P_i} +\lambda \hat{n}=0
$$

Plug Equation ($\ref{eqn:or_mean}$),


$$
\frac{1}{N}\sum_{i=1}^N (\vec{P_i}-\langle \vec{P}\rangle)\cdot\hat{n}(\vec{P_i}-\langle\vec{P}\rangle) +\lambda \hat{n}=0
$$


Write in terms of $\alpha$, $\beta$, $x$ and $y$,


$$
\begin{align}
\alpha \mathrm{Var}(x)+\beta\mathrm{Cov}(x,y)+\alpha\lambda &= 0 \\
\alpha \mathrm{Cov}(x,y)+\beta\mathrm{Var}(y)+\beta\lambda &= 0
\end{align}
$$


And with the condition


$$
\frac{\partial \mathcal{L}}{\partial \lambda}=0
$$


or
$$
\alpha^2+\beta^2 = 1
$$




Solving for $\alpha$, $\beta$, and $\lambda$,  we have




<figure>
  <center>
  <img src="/assets/images/" width="400">
   </center>
  <center>
    <figcaption> Figure 1. 
    </figcaption>
  </center>
</figure>









### References

Simon, H. A. (1955). On a Class of Skew Distribution Functions. *Biometrika*, 42(3/4), 425-440. https://www.jstor.org/stable/2333389?origin=JSTOR-pdf



