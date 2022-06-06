---
title: "Hamiltonian Monte Carlo vs. Metropolis"
date: 2022-04-12
typora-root-url: ../../kezhaozhang.GitHub.io
---

This note compares Metropolis and Hamiltonian Monte Carlo algorithms, using autocorrelation and effective sample size as metrics. Unimodal target distribution is used in this note, and multimodal target distribution will be discussed in a future note.



### Hamiltonian System

Hamiltonian Monte Carlo (HMC) uses Hamiltonian dynamics to generate distant candidate samples with a higher probability of acceptance than the Metropolis algorithm. 



The Hamilton is the energy of the system:

 
$$
H(q, p) = U(q) + K(p),
$$


where the potential energy $U(q)$ and the kinetic energy $K(q)$ are functions of the position $q$ and the momentum $p$, which are governed by the Hamiltonian dynamics:


$$
\begin{align}
\frac{d q_i}{dt} &=\frac{\partial H}{\partial p_i} \\
\frac{d p_i}{dt} & = -\frac{\partial H}{\partial q_i}
\end{align}
$$


The probability of $H(p, q)$ is


$$
P(p, q)= \frac{1}{Z}\exp\left(-H(p,q)\right),
$$


where $Z$ is the normalization factor.



Suppose the pdf of the target distribution is $f(q)$, and if we set $U(q) = -\log(f)$ and $K(p)=\frac{1}{2}p^Tp$,  then


$$
\begin{align}
P(p, q) & =  \frac{1}{Z}\exp(-H)\\
        & \propto   \exp\left(-U(q)\right)\exp\left(-K(p)\right)\\
        & = f(q) \varphi(p),
\end{align}
$$
where $\varphi(p) = \frac{1}{\sqrt{2\pi}}\exp(-\frac{p^2}{2})$ is the pdf of a normal distribution. The probability is decomposed into the target distribution $f$ and a normal distribution $\varphi$. 



Using  $U(q) = -\log(f)$ and $K(p)=\frac{1}{2}p^Tp$,  we generate samples with the Hamiltonian dynamics. In each proposed sample generation, $p$ is drawn from a normal distribution: $p\sim \mathcal{N}(0,1)$. The trajectory on the Hamiltonian has the same energy and probability. Parameter space can be explored in a larger size when the new proposed candidate follows the Hamiltonian dynamics.



### Leapfrog Integration

Leapfrog is a numerical integration method. In a one-dimensional case:


$$
p(t+\frac{\varepsilon}{2}) = p(t) -\frac{\varepsilon}{2} \frac{dU(q(t))}{dq}\\
q(t+\varepsilon) = q(t)+ \varepsilon p(t+\frac{\varepsilon}{2})  \\

p(t+\varepsilon) = p(t+\frac{\varepsilon}{2}) -\frac{\varepsilon}{2}\frac{dU(q(t+\varepsilon))}{d q}
$$


where $\varepsilon$ is the integration time step size. A comparison of the leapfrog and Euler methods for the system $H(p,q) = \frac{1}{2} (q^2 + p^2)$  in Figure 1 demonstrates the robustness of the leapfrog method.

<figure>
  <center>
  <img src="/assets/images/leapfrog_euler_50_0.25.svg" width="700">
   </center>
  <center>
  <figurecaption>
  Figure 1. Trajectories of leapfrog and Euler methods. The Euler method diverges, whereas the leapfrog method is close to the exact solution. 
  </figurecaption>
  </center>
</figure>



### Code Implementation

The algorithms are implemented for one-dimensional distribution.

#### Hamilton Monte Carlo

The implementation is based on the R code in Neal (2001).

```python
from autograd import grad
import autograd.numpy as np

def hmc(U, epsilon, L, current_q):
    ''' HMC based on Radford Neal's R implementation
    Inputs:
    	U: function of -log(f), where f is the target distribution
    	epsilon: step size in leapfrog integration
    	L: number of steps in leapfrog integration
    	current_q: q of the current step
    Outputs:
    	q: the value of q after HMC sampling
    	0 or 1: indicator of whether the proposed sample is accepted (1) or rejected (0)
    '''
    grad_U = grad(U)
    
    q = current_q
    p = np.random.randn()
    current_p = p
    
    # Make a half step for momentum at the beginning
    p = p - epsilon * grad_U(q)/2
    
    # Alternate full steps for position and momentum
    for i in range(L):
        # Make a full step for the position
        q += epsilon * p
        # Make a full step for the momentum, except at the end of the trajectory
        if i<L-1:
            p -= epsilon * grad_U(q)
    #Make a  half step for momentum at the end
    p -= epsilon * grad_U(q)/2
    
    # Negate momentum at the end of trajectory to make the proposal symmetric
    p = -p
    
    #Evaluation potential and kinetic energies at start and end of trajectory
    current_U = U(current_q)
    current_K = current_p**2/2
    proposed_U = U(q)
    proposed_K = p**2/2
    
    # Accept or reject the proposal
    if np.random.rand() < np.exp(current_U - proposed_U + current_K - proposed_K):
        return q, 1 ## new q and acceptance
    else:
        return current_q, 0 ## new q and rejection
```

#### Metropolis

The proposed sample $x'$ is generated from a uniform distribution centered around the current value $x$:

 
$$
x' \sim \mathrm{Uniform}(x-\delta, x+\delta),
$$


where $\delta$ controls the range of the random distribution.



```python
def metropolis(target_fn, x0, N, delta):
    '''Metropolis sampling
    Inputs:
        target_fn: callable function of the target distribution
        x0: initial value 
        N: number of samples
        delta: the size of the random perturbation to generate the next proposed sample
    Output:
        samples of length N including the user-supplied initial value
        number of proposed samples that have been accepted
    '''
    
    x_current = x0
    trace = [x_current]
    acceptance_counter = 0
    
    for i in range(N-1):
        x = x_current + (1- 2*np.random.rand())*delta  # proposal
        if np.random.rand() < target_fn(x)/target_fn(x_current):
            x_current = x
            acceptance_counter += 1
        trace.append(x_current)
    return np.array(trace), acceptance_counter
```



### Result

The target distribution is a normal distribution with $\mu=3$ and $\sigma=1.2$. We show that the Hamiltonian Monte Carlo is more efficient than the Metropolis algorithm.

#### Acceptance Rate

The Metropolis and Hamiltonian Monte Carlo have different acceptance rates for the proposed samples. The acceptance rate depends on the uniform distribution range from which the sample is drawn for the Metropolis algorithm. As shown in Figure 2, the larger the range, the lower the acceptance rate.  For the Hamiltonian Monte Carlo, for various step sizes and numbers of steps in the leapfrog integration, the acceptance rate is almost 1.

<figure>
  <center>
  <img src="/assets/images/metropolis_hmc_acceptance_rate_metropolis.svg" width="600">
   </center>
  <center>
  <figurecaption>
  Figure 2. The acceptance rate of the Metropolis algorithm as a function of &#948.
  </figurecaption>
  </center>
</figure>



#### Autocorrelation

The autocorrelation vs. lag plots are shown in Figure 3 for Metropolis and Figure 4 for Hamiltonian Monte Carlo. In both cases, the autocorrelation depends on the setting of the sampling. Therefore, the sampling setting can be optimized to reduce autocorrelation.



A noticeable difference is that autocorrelation is positive for all lags in Metropolis, but there is negative autocorrelation at odd lags in some cases for the Hamiltonian Monte Carlo. 

<figure>
  <center>
  <img src="/assets/images/mcmc_metropolis_autocorr_metropolis.svg" width="700">
   </center>
  <center>
  <figurecaption>
  Figure 3. Autocorrelation of the traces generated by the Metropolis algorithm for various values of &#948. Autocorrelation is positive.
  </figurecaption>
  </center>
</figure>



<figure>
  <center>
  <img src="/assets/images/mcmc_metropolis_autocorr_hmc.svg" width="850">
   </center>
  <center>
  <figurecaption>
  Figure 4. Autocorrelation of the traces generated by the Hamiltonian Monte Carlo algorithm for various step sizes (&#949) and numbers of steps (<i>L</i>) in the leapfrog integration. Autocorrelation is negative on odd lags in some cases.
  </figurecaption>
  </center>
</figure>

##### Area Under Autocorrelation Envelop

The area under the envelop curve (AUC) is calculated to quantify the autocorrelation. When there is negative autocorrelation, the absolute value is used. 

```python
# Calulation of the area under the autocorrelation envelop curve for a trace
import numpy as np
import arviz as az
max_lag = 20

np.trapz(np.abs(az.autocorr(trace)[:maxlag]))
```



A smaller area under the autocorrelation curve indicates more efficient sampling. For the Metropolis algorithms, the best $\delta \sim 5$ (figure 5). For the Hamiltonian Monte Carlo, it appears that the less autocorrelation occurs when the product of leapfrog step size $\epsilon$ and integration number of steps  $L$ is around $2$, $6$, and $10$ ($\epsilon L = 2, 6, \mathrm{or}\  10$) (Figure 6).

<figure>
  <center>
  <img src="/assets/images/mcmc_metropolis_auc_vs_delta_metropolis.svg" width="600">
   </center>
  <center>
  <figurecaption>
  Figure 5. The area under the autocorrelation curve of the traces generated by the Metropolis algorithm for various &#948. The minimum AUC=1.9 at &#948=5.
  </figurecaption>
  </center>
</figure>

<figure>
  <center>
  <img src="/assets/images/mcmc_metropolis_auc_vs_eL_hmc.svg" width="600">
   </center>
  <center>
  <figurecaption>
  Figure 6. The area under the autocorrelation curve of the traces generated by the Hamiltonian Monte Carlo algorithm for various &#949<i>L</i>. The minimum AUC=0.72 at &#949<i>L</i>=2. Here ESS is relative, which is the ratio of effective sample size and the sample size.
  </figurecaption>
  </center>
</figure>

The minimum AUC of the Hamiltonian Monte Carlo is much smaller (0.72 when $\varepsilon L=2$) than that of the Metropolis algorithm (1.9 when $\delta=5$). This suggests that a fine-tuned Hamiltonian Monte Carlo sampling is more efficient than the Metropolis algorithm.

#### Effective Sample Size (ESS)

Intuitively, the effective sample size should be consistent with the area under the autocorrelation curve. Indeed this is the case for the Metropolis algorithm, as shown in Figure 7.

<figure>
  <center>
  <img src="/assets/images/mcmc_metropolis_ess_auc.svg" width="650">
   </center>
  <center>
  <figurecaption>
  Figure 7. The strong correlation between the area under the autocorrelation curve and the effective sample size (ESS) of the traces generated by the Metropolis algorithm.
  </figurecaption>
  </center>
</figure>



The effective sample size in the Hamiltonian Monte Carlo is larger in the sample size in many cases (see Figure 8). This is due to the negative autocorrelations on odd lags. *Stan Reference Manual* briefly discusses this in the *Effective Sample Size* section. 

<figure>
  <center>
  <img src="/assets/images/mcmc_metropolis_ess_vs_eL_hmc.svg" width="650">
   </center>
  <center>
  <figurecaption>
  Figure 8. The strong correlation between the area under the autocorrelation curve and the effective sample size (ESS) of the traces generated by the Hamiltonian Monte Carlo. Here ESS is relative, which is the ratio of effective sample size and the sample size.
  </figurecaption>
  </center>
</figure>



### Conclusion and Discussion

The simulations in this note show that the setting of the Monte Carlo sampling should be optimized. For example, Figure 9 and Figure 10 show that sampling performance depends on the setting for both the Metropolis algorithm and the Hamiltonian Monte Carlo.

<figure>
  <center>
  <img src="/assets/images/metropolis_hmc_trace_delta_0.5_metropolis.svg" width="750">
  <img src="/assets/images/metropolis_hmc_trace_delta_5_metropolis.svg" width="750">
   </center>
  <center>
  <figurecaption>
  Figure 9. Traces generated by the Metropolis algorithm. Top: &#948=0.5; Bottom: &#948=5. The former is not as good as the latter: more discrepancy between the sample  and the target distributions; and stronger autocorrelation. 
  </figurecaption>
  </center>
</figure>

 

<figure>
  <center>
  <img src="/assets/images/metropolis_hmc_trace_e=0.2_L=20_hmc.svg" width="750">
  <img src="/assets/images/metropolis_hmc_trace_e=0.1_L=20_hmc.svg" width="750">
   </center>
  <center>
  <figurecaption>
  Figure 10.  Traces generated by the Hamiltonian Monte Carlo. Top: &#949=0.2 and <i>L</i>=20; Bottom: &#949=0.1 and <i>L</i>=20. The former is not as good as the latter: more discrepancy between the sample and the target distributions; and larger nonuniformity. 
  </figurecaption>
  </center>
</figure>

In addition, the Hamiltonian Monte Carlo is more efficient (after the sampling setting optimization) than the Metropolis algorithm in terms of sample acceptance rate and autocorrelation. 



However, Hamiltonian Monte Carlo encounters challenges when the target distribution is multimodal and struggles to move between the modes. Tempered methods can be used to improve the sampling in the multimodal case.  We will discuss this in a future blog.

<figure>
  <center>
  <img src="/assets/images/metropolis_hmc_bimodal_hmc.svg" width="650">
   </center>
  <center>
  <figurecaption>
  Figure 11. An example of the Hamiltonian Monte Carlo generated samples unable to move between modes for a bimodal distribution.
  </figurecaption>
  </center>
</figure>

## References

Neal, R.M. (2011). MCMC Using Hamiltonian Dynamics. In S. Brooks, A. Gelman, G.L. Jones & X. Meng (Eds.),  *Handbook of Markov Chain Monte Carlo* (pp.113-162). Chapman & Hall/CRC.

Effective Sample Size. *Stan Reference Manual* (https://mc-stan.org/docs/2_21/reference-manual/effective-sample-size-section.html)
