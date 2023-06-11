---
title: "Singularity in Covariance Matrix in Gaussian Process Regression"
date:2023-06-07
typora-root-url: ./..
---



This post discusses the issue of singularity in the covariance matrix when performing Gaussian Process regression, particularly when dealing with a large number of training data points, as shown in a previous [post](https://kezhaozhang.github.io/2023/05/31/Gaussian-process-regression.html). We explore two approaches to handle this numerical problem: adjusting the kernel parameters and introducing jitter to the diagonal of the covariance matrix. Additionally, we evaluate the use of low-rank matrix approximations for the covariance matrix.

### Data

Figure 1 depicts the training data points and the true function that will be utilized in the Gaussian Process regression.

<figure>
  <center>
  <img src="/assets/images/gpr_cov_data.svg" width="600">
   </center>
  <center>
    <figcaption> Figure 1. Training data points and true function for Gaussian Process regression.
    </figcaption>
  </center>
</figure>


### The rank of the Covariance Matrix



For our analysis, we employ the squared exponential or radial basis kernel. The covariance matrix is calculated using the following equation:


$$
K(\mathbf{x}_i, \mathbf{x}_y) = \alpha \exp\left({-\frac{\mid \mathbf{x}_i-\mathbf{x}_j\mid^2}{2 l^2}}\right).
\notag
$$


#### Low Rank with Large Sample Size of Training Data

The rank of the covariance matrix strongly depends on the number of training data points, as demonstrated in Figure 2. As the sample size increases, it becomes evident that the covariance matrix is not full rank, especially for sample sizes exceeding 20.

<figure>
  <center>
  <img src="/assets/images/gpr_cov_cov_rank.svg" width="600">
   </center>
  <center>
    <figcaption> Figure 2. Covariance matrix rank plotted against sample size. The squared exponential kernel with a length scale of <i>l</i>=1 and amplitude <i>&alpha;</i>=1 was applied.
    </figcaption>
  </center>
</figure>



#### Small Length Scale is Necessary for Full Rank

When the sample size of the training data increases, the differences in the x-values among the training data decrease, resulting in more similarities among the elements of the covariance matrix. Consequently, the rows of the covariance matrix become more correlated, leading to a reduced rank. In the extreme case of an infinitely large length scale, the rows of the covariance matrix become identical, resulting in a rank of 1.

To ensure a full-rank covariance matrix, the length scale ($l$) needs to be reduced. Figure 3 demonstrates the rank of the covariance matrix for 40 training data points as a function of the length scale. It is observed that the covariance matrix attains full rank when the length scale is smaller than 0.44.

<figure>
  <center>
  <img src="/assets/images/gpr_cov_rank_vs_l.svg" width="600">
   </center>
  <center>
    <figcaption> Figure 3. Covariance matrix rank plotted against length scale. 40 data points in Figure 1 are used to calculate the covariance matrix.
    </figcaption>
  </center>
</figure>

Regression results using various length scale values are presented in Figure 4. It is evident that when the length scale is small (less than 0.2), the prediction exhibits strong oscillations, indicating suboptimal length scale values.

<figure>
  <center>
  <img src="/assets/images/gpr_cov_fit_vs_length.svg" width="1000">
   </center>
  <center>
    <figcaption> Figure 4. Regression results with different length scale values. The amplitude factor <i>&alpha;</i>=1.
    </figcaption>
  </center>
</figure>



#### Optimization of Kernel Hyperparameters $l$ and $\alpha$

The kernel parameters, length scale ($l$), and amplitude ($\alpha$), can be optimized to maximize the marginal likelihood. Figure 5 illustrates a contour plot of the log marginal likelihood as a function of the length scale and amplitude, considering a maximum length scale of 0.44 to ensure a full-rank covariance matrix. It is evident that while smaller length scales guarantee a full-rank covariance matrix, they may not optimize the marginal likelihood.

<figure>
  <center>
  <img src="/assets/images/gpr_cov_loglikelihood_alpha_l_contourf.svg" width="600">
   </center>
  <center>
    <figcaption> Figure 5. Contour plot of log marginal likelihood as a function of length scale and amplitude. The maximum value of the log marginal likelihood is located at <i>i=0.44</i> and <i>&alpha;=4.1</i>.
    </figcaption>
  </center>
</figure>

Regression results using the optimal length scale ($l = 0.44$) and different amplitude ($\alpha$) values are presented in Figure 6. Overall, the regression predictions are almost identical for all $\alpha$ values. The primary difference lies in the uncertainty interval, which appears smaller for smaller $\alpha$ values. This discrepancy arises because the covariance matrix of the predicted value ($\mathbf{y}_2$) conditioned on the observation ($\mathbf{y}_1$) is directly proportional to $\alpha$, while the mean is not (see Equation $\ref{eqn:cond_prob}$).


$$
\mathbf{y}_2 \mid \mathbf{y}_1 \sim \mathcal{N}(\mathbf{\mu}_1+\Sigma_{21}\Sigma_{11}^{-1} (\mathbf{y}_1-\mathbf{\mu}_1),  
\Sigma_{22}-\Sigma_{21}\Sigma_{11}^{-1}\Sigma_{12})
\label{eqn:cond_prob}
$$


<figure>
  <center>
  <img src="/assets/images/gpr_cov_fit_vs_alpha.svg" width="1000">
   </center>
  <center>
    <figcaption> Figure 6. Regression results with different amplitude values. The length scale <i>l=0.44</i>.
    </figcaption>
  </center>
</figure>



The question arises as to whether the uncertainty estimate truly reflects the uncertainty of the model prediction. We will address this query in a separate post.



### Adding Jitter to Handle Low-Rank Covariance Matrix

To address the issue of a low-rank covariance matrix, one effective approach is to introduce jitter to the diagonal of the covariance matrix (Neal 1997):


$$
K_{\sigma}(\mathbf{x}_i, \mathbf{x}_y) = K(\mathbf{x}_i, \mathbf{x}_y) + \sigma^2 \mathrm{I}. \label{eqn:cov_jitter}
$$


Figure 7 depicts the rank of the covariance matrix for a set of 40 training data points, considering different amounts of added jitter. The value of $\sigma^2$ is is measured in terms of the machine epsilon for 64-bit floating-point numbers: `np.finfo(np.float64).eps = 2.220446049250313e-16`. In this specific example, when $\sigma^2 > 465\times \mathrm{eps} =1.03\times 10^{-13}$ the covariance matrix attains full rank.

<figure>
  <center>
  <img src="/assets/images/gpr_cov_rank_vs_sigma2.svg" width="600">
   </center>
  <center>
    <figcaption> Figure 7. Rank of the covariance matrix for 40 training data points, utilizing the squared exponential kernel with <i>l=1</i> and <i>&alpha;=1</i>. Jitter, denoted by <i>&sigma;<sup>2</sup></i>, is added in varying amounts, with its unit representing the machine epsilon for 64-bit floating-point numbers.  
    </figcaption>
  </center>
</figure>

Figure 8 showcases the regression results obtained with different jitter values, represented by $\sigma^2$. Remarkably accurate regression outcomes are achieved with small jitter values, such as $\sigma^2\le 10^{-6}$. However, it is worth noting that for small $\sigma^2$ values, the estimated uncertainty appears unusually small. This phenomenon is likely an artifact of the numerical calculations rather than an accurate reflection of the model's true uncertainty.



<figure>
  <center>
  <img src="/assets/images/gpr_cov_fit_vs_sigma2.svg" width="1000">
   </center>
  <center>
    <figcaption> Figure 8. Regression results obtained with various jitter values, denoted by <i>&sigma;<sup>2</sup></i>. The length scale is set to <i>l=1</i>, and the amplitude factor is <i>&alpha;</i>=1.
    </figcaption>
  </center>
</figure>



### Approximating Covariance Matrix with A Low-Rank Matrix

In the preceding sections, we observed that the covariance matrix of the training data can possess a low rank. We are interested in evaluating the accuracy of approximating the covariance matrix with a low-rank matrix in Gaussian Process regression. The utilization of a low-rank matrix for approximation has been established as a means to reduce the computational requirements associated with the covariance matrix (Williams and Seeger, 2001).

The covariance matrix $\Sigma$ can be decomposed using Singular Value Decomposition (SVD):


$$
\Sigma = U \Lambda U^T.
$$


To construct the low-rank matrix approximation, we utilize the $n$ columns of $U$ corresponding to the $n$ largest singular values.

The covariance matrix $\Sigma$ is formulated according to Equation ($\ref{eqn:cov_jitter}$). Employing a low-rank matrix approximation can yield excellent regression results, as demonstrated in Figure 9 and Figure 10. Smaller jitter values $\sigma^2$ necessitate a larger number of components (higher $n$) for satisfactory results. In Figure 9, for $\sigma^2=10^{-6}$, a minimum value of $n=18$ is required for desirable regression outcomes. Conversely, when dealing with a larger jitter value $\sigma^2=0.1$, as shown in Figure 10, a smaller $n\geq 10$ suffices to achieve good results.



<figure>
  <center>
  <img src="/assets/images/gpr_cov_lowrank_prediction_sigma2=1e-6.svg" width="1200">
   </center>
  <center>
    <figcaption> Figure 9. Regression results obtained with low-rank matrix approximation to the covariance matrix. The matrix rank is represented by <i>n</i>. The covariance kernel is computed using a squared exponential kernel with a length scale of <i>l=1</i> and an amplitude factor of <i>&alpha;</i>=1. A jitter of <i>&sigma;<sup>2</sup>=10<sup>-6</sup></sub></i> is added to the diagonal of the covariance matrix.
    </figcaption>
  </center>
</figure>



<figure>
  <center>
  <img src="/assets/images/gpr_cov_lowrank_prediction_sigma2=1e-1.svg" width="1200">
   </center>
  <center>
    <figcaption> Figure 10. Regression results obtained with low-rank matrix approximation to the covariance matrix. The matrix rank is represented by <i>n</i>. The covariance kernel is computed using a squared exponential kernel with a length scale of <i>l=1</i> and an amplitude factor of <i>&alpha;</i>=1. A jitter of <i>&sigma;<sup>2</sup>=0.1</i> is added to the diagonal of the covariance matrix. 
    </figcaption>
  </center>
</figure>



### References

- Neal, R. M. (1997). Monte Carlo Implementation of Gaussian Process Models for Bayesian Regression and Classification. *arXiv*. https://doi.org/https://arxiv.org/abs/physics/9701026v2
- Williams, C. K. I., & Seeger, M. (2001). Using the Nyström Method to Speed Up Kernel Machines. In T. K. Leen, T. G. Dietterich, & V. Tresp (Eds.), *Advances in Neural Information Processing Systems 13 (NIPS 2000)* (pp. 682-688). MIT Press. http://papers.nips.cc/paper/1866-using-the-nystrom-method-to-speed-up-kernel-machines.pdf