---
Title: A Simple Non-Bayesian Solution to Monty Hall Problem
---



This short note describes a simple non-Bayesian solution to the Monty Hall problem. A charming Bayesian analysis can be found in the book *Bernoulli's Fallacy*.

[The Monty Hall problem](https://en.wikipedia.org/wiki/Monty_Hall_problem) is stated as follows on Wikipedia:

> Suppose you're on a game show, and you're given the choice of three doors: Behind one door is a car; behind the others, goats. You pick a door, say No. 1, and the host, who knows what's behind the doors, opens another door, say No. 3, which has a goat. He then says to you, "Do you want to pick door No. 2?" Is it to your advantage to switch your choice?



## Solution

The probability of selecting the door with the car behind *without switching* is simply $\frac{1}{3}$.

The probability of getting the car after switching is the weighted average of two scenarios: the car was selected in the first place, and the car was not chosen in the first place. The associated probabilities are listed below.

|                 Scenario                  |  Probability  | Probability of getting the car after switching |                             Note                             |
| :---------------------------------------: | :-----------: | :--------------------------------------------: | :----------------------------------------------------------: |
|     Car selected (car behind Door #1)     | $\frac{1}{3}$ |                       0                        |                                                              |
| Car NOT selected (Car not behind Door #1) | $\frac{2}{3}$ |                       1                        | The probability of getting the car is $1$ because the car is behind the only remaining door (Door #2). |

The overall probability of getting the car *after switching* is 
$$
\frac{1}{3}\times0+\frac{2}{3}\times 1 = \frac{2}{3}. \notag
$$
Therefore, switching increases the probability of getting the car to $\frac{2}{3}$ from the probability of $\frac{1}{3}$ without switching.