---
title: "Hamiltonian Monte Carlo"
date: 2022-04-12
---

This note has heavily borrowed from Neal (2011)

Hamiltonian Monte Carlo (HMC) is a 



The Hamilton of a system is the sum of its potential energy $U(q)$ of the position $q$ and its kinetic energy $K(q)$ of the momentum $p$:
$$
H(q, p) = U(q) + K(p).
$$


Hamiltonian dynamics follow the following equations:
$$
\begin{array}{ccc}

\frac{d q_i}{dt} &=&\frac{\partial H}{\partial p_i} \\

\frac{d p_i}{dt} & =& -\frac{\partial H}{\partial q_i}

\end{array}
$$


A Hamiltonian system has the following properties:

- time reversibility
- conservation of H
- conservation of volume in phase space

The trajectory on the Hamiltonian has the same energy and probability. Parameter space can be explored in a larger size when the new proposed candidate follows the Hamiltonian dynamics.



Leapfrog integration is a particular approach to writing two coupled first-order ordinary differential equations with finite differences. 

![leapfrog](http://cvarin.github.io/CSci-Survival-Guide/images/leapfrog.gif)

Numerical integration with LeapFrog algorithm.

- Leapfrog algorithm

- Advantage:
  $$
  P(p, q)= \frac{1}{Z}\exp(-H)\propto \exp(-U(q))\exp(-K(p))\\
  
  P(q) = \exp(-U(q))\\
  
  U(q)= -\log(P(q)) \\
  
  K(p) = \frac{1}{2} p^T p\\
  $$
  $\frac{d q}{d t} = \frac{\partial H}{\partial p} = \frac{d K(p)}{d p}=p$

$$
\begin{array}\\
\frac{d q}{d t} &=& \frac{\partial H}{\partial p} = \frac{d K(p)}{d p}=p \\
\frac{d p}{d t} &=& -\frac{\partial H}{\partial q} = -\frac{d U(q)}{d q}
\end{array}
$$

For simplicity without loss of generality, use a one-dimensional case for the Leapfrog algorithm.
$$
p(t+\frac{\Delta}{2}) = p(t) -\frac{\Delta}{2} \frac{dU(q(t))}{dq}\\
q(t+\Delta) = q(t)+ \Delta p(t+\frac{\Delta}{2})  \\

p(t+\Delta) = p(t+\frac{\Delta}{2}) -\frac{\Delta}{2}\frac{dU(q(t+\Delta))}{d q}
$$

$$
p(t+\frac{\varepsilon}{2}) = p(t) -\frac{\varepsilon}{2} \frac{dU(q(t))}{dq}\\

q(t+\varepsilon) = q(t)+ \varepsilon p(t+\frac{\varepsilon}{2})  \\

p(t+\varepsilon) = p(t+\frac{\varepsilon}{2}) -\frac{\varepsilon}{2}\frac{dU(q(t+\varepsilon))}{d q}
$$



##### Reference

Neal, R.M. (2011). MCMC Using Hamiltonian Dynamics. In S. Brooks, A. Gelman, G.L. Jones & X. Meng (Eds.),  *Handbook of Markov Chain Monte Carlo* (pp.113-162). Chapman & Hall/CRC.



http://cvarin.github.io/CSci-Survival-Guide/leapfrog.html

