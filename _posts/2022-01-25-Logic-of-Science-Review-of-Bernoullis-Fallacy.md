---
title: "Logic of Science: Review of Bernoulli's Fallacy"
---

The book is well researched and very lucid. The arguments for the Bayesian approach are convincing. It's a beneficial book before reading Jaynes' *Probability Theory*.

**Bernoulli's Fallacy:  Statistical Illogic and the Crisis of Modern Science**

**Aubrey Clayton**

**Columbia University Press (2021)**



<figure>
 <center>
  <img src="/assets/images/fallacy_book_cover.jpeg", height="400">
 </center>
</figure>



I have read a few books on Bayesian analysis. They all focus on the technical details of the inference methods such as MCMC and software for probabilistic programming such as Stan and PyMC.  These books have helped me to become a Bayesian enthusiast.  Indeed, using Bayesian inference with probabilistic programming software is liberating, and life becomes more straightforward. Instead of relying on various complicated tests and tricks, we build the data model and draw conclusions from the resultant posterior distributions.

As a next logical step, I want to learn more about Bayesian methods' theoretical and historical background and have planned to read Jaynes' *Probability Theory: The Logic of Science*. Then I found Aubrey Clayton's delightful and lucid book, *Bernoulli's Fallacy*. It compares frequentist and Bayesian methods and demonstrates the deficiencies of the frequentist approach and the history that leads to the dominance of the frequentist methods in current statistics and sciences.   

The fallacy of the frequentist methods, named Bernoullis' Fallacy by the author because of Bernoulli's pioneering work on the frequentist approach to probability, is confusing sampling probability with inferential probability. The former is the probability of data given the hypothesis (likelihood), and the latter is the probability of the hypothesis given the data (posterior). The frequentist method uses the former to infer the latter. 

One key reason frequentists reject the Bayesian approach is their objection to the subjective priors.  They believe that probability is a quality of objects and inference is to obtain such property objectively. To conform to objectivity, they define probability as the frequency of repeated measurements or proportions in a population. 

Objectivity can also explain the adoption of the frequentist methods early on by the "soft" social sciences, which needed the "objectivity" to justify their validity. In addition, the author argues that Galton, Pearson, and Fisher, the most important progenitors of frequentist statistics, advocated the frequentist methods because they wanted to put an objectivity facade to their eugenics agenda. Since I'm not very familiar with eugenics history and its interaction with statistics, I'm not sure the author's case is compelling. 

Using various examples and references, for example, the replication problem, the author convincingly makes the case that the frequentist methods are illogical and the Bayesian methods are the logic of science. Fundamentally, the probability "measures the plausibility of a proposition given some assumed information." It is "a function of our knowledge or lack thereof and not an inherently measurable quantity of a physical system. Frequencies are measurable; probabilities are not."

I like that both the author and Jaynes consider Bayesian methods as the logic of science. Bayesians treat probability as the subjective model of the world. The inference is a process of updating our understanding of the world based on observed data. In this sense, the Bayesian approach is similar to the development of theories in physics, where the models are updated iteratively as new experimental data are collected. An example is the theory of gravity: neither Netown's classical theory nor Einstein's theory of relativity is the absolute truth of nature. However, they both reflect the best knowledge and understanding of the nature of gravity with the data available when the theories were developed. They both have successfully solved problems in the domains where they are valid. And we know there will be a new theory of gravity (e.g., unification of gravity and quantum mechanics) in the future as more discoveries will be made.  

The similarity extends to causal inference, where our model is our best understanding of the causal relationships in the system, and we use data and statistical inference to validate and update the causal model iteratively.

In this sense, all of our models are approximations to the truth; the Bayesian approach is the most logical tool that helps us refine our understanding and makes our models closer to reality.

In summary, *Bernoulli's Fallacy* is well researched and easy to understand. The arguments for the Bayesian approach are convincing. It's a beneficial book before reading Jaynes' *Probability Theory*.