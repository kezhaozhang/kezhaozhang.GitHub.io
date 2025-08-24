---

title: "Causal Inference using Front Door Criterion"
date: 2025-08-19
typora-root-url: ./..
---



In this note, we calculate the causal effect using the front door criterion with data from an example in the book Causal Inference in Statistics: A Primer by Pearl, Glymour, and Jewell.



### Problem Statement

We consider the classic question: Does smoking cause lung cancer?



 Table 1 presents simulated data of two groups--smokers and nonsmokers--each with 400000 individuals with high risks of lung cancer. For each subject,  both lung cancer and tar deposits in the lung were recorded. The same data are regrouped in Table 2.



<figure>
  <center>
    <figcaption> 
      Table 1
    </figcaption>
  </center>
  <center>
  <img src="/assets/images/frontdoor_table1.svg" width="550">
   </center>
 </figure>





<figure>
  <center>
    <figcaption> 
      Table 2
    </figcaption>
  </center>
  <center>
  <img src="/assets/images/frontdoor_table2.svg" width="450">
   </center>
</figure>
 

At first glance, the cancer rates appear lower among smokers than nonsmokers across the all groupings. Naively, this suggests that smoking does not cause lung cancer. However, medical knowledge indicatres that tar deposits are a direct cause of lung cancer. 

Comparing the tar vs. no-tar groups shows a clear association: higher tar levels led to higher cancer rates, regrardless of smoking status. The question becomes: 

If smoking increases tar deposits, and tar increases cancer risk, what is the causal effect of smoking on cancer?

We answer this using the front door criterion.




### The Front Door Criterion



To answer the question in the preceding section, we need a causal model for lung cancer. One such model is shown in Figure 1. Smoking ($X$) influences tar deposit ($Z$), which in turn affects lung cancer ($Y$).  



<figure>
  <center>
  <img src="/assets/images/frontdoor_scm.svg" width="450">
   </center>
  <center>
    <figcaption> 
      Figure 1. Causal model with tar deposits as a mediator.
    </figcaption>
  </center>
</figure>



The key properties of the model are:

-  $Z$ blocks all direct paths from $X$ to $Y$.
-  There is no unblocked backdoor path from $X$ to $Z$. 
- All backdoor paths from $Z$ to $Y$ are blocked by $X$. $

Thus, $Z$ satisfies the front-door criterion. 



In the model, there are two causal relationships:

- $X\rightarrow Z$:  $P(Z=z\mid \mathrm{do}(X=x))$
- $Z\rightarrow Y$: $P(Y=y\mid \mathrm{do}(Z=z))$



Combining the two and marginalizing over $Z$, the causal effect of $X$ on $Y$ is then


$$
P(Y=y\mid \mathrm{do}(X=x))=\sum_z P(Y=y\mid  \mathrm{do}(Z=z))P(Z=z\mid  \mathrm{do}(X=x)) \label{eqn_PY_doX}
$$



Further, 


$$
P(Z=z\mid \mathrm{do}(X=x))= P(Z=z\mid X=x) \label{eqn:conditional}
$$


because there is no backdoor path between $X$ and $Z$.  And 


$$
P(Y=y\mid \mathrm{do}(Z=z))=\sum_{x'}P(Y=y\mid Z=z, X=x')P(X=x') \label{eqn:backdoor}
$$


as a result of the backdoor criterion (see the appendix to this note for a derivation). 



Substituting Equations ($\ref{eqn:conditional}$) and ($\ref{eqn:backdoor}$) into Equation ($\ref{eqn_PY_doX}$) gives the front-door formula:


$$
\begin{align} \notag
P(Y=y\mid \mathrm{do}(X=x))& =\sum_z P(Y=y\mid  \mathrm{do}(Z=z))P(Z=z\mid  \mathrm{do}(X=x)) \\
 & = \sum_{z}\sum_{x'} P(Y=y\mid Z=z, X=x')P(X=x')P(Z=z\mid X=x) \label{eqn:front-door}
\end{align}
$$




###  Numerical Calculations 



Using Table 1, we compute the proabilities needed for Equation ($\ref{eqn:front-door}$):



- Tar distributiin given smoking: $P(Z=z\mid X=x)$

  |    $X$     | $P(Z=\text{Tar}\mid X)$ | $P(Z=\text{No Tar}\mid X)$ |
  | :--------: | :---------------------: | :------------------------: |
  |  Smokers   |         $0.95$          |           $0.05$           |
  | NonSmokers |         $0.05$          |           $0.95$           |

  

- Marginal distributions of smokers vs. nonsmokers: $P(X=x')$

  |    $X$     | $P(X)$ |
  | :--------: | :----: |
  |  Smokers   | $0.5$  |
  | NonSmokers | $0.5$  |



- Cancer risk by tar and smoking: $P(Y=y\mid Z=z, X=x')$




|  $Z$   |    $X$     | $P(Y=\mathrm{Cancer}\mid Z, X)$ |
| :----: | :--------: | :-----------------------------: |
|  Tar   |  Smokers   |              0.15               |
|  Tar   | Nonsmokers |              0.95               |
| No Tar |  Smokers   |               0.1               |
| No Tar | Nonsmokers |               0.9               |


#### 

|  $Z$   |    $X$     | $P(Y=\mathrm{No\ \ Cancer}\mid Z, X)$ |
| :----: | :--------: | :-----------------------------------: |
|  Tar   |  Smokers   |                 0.85                  |
|  Tar   | Nonsmokers |                 0.05                  |
| No Tar |  Smokers   |                  0.9                  |
| No Tar | Nonsmokers |                  0.1                  |



Applying Equation ($\ref{eqn:front-door}$) , we obtain:


$$
P(Y=\mathrm{Cancer}\mid X=\mathrm{Smokers}) = 0.5475 \notag
$$

$$
P(Y=\mathrm{Cancer}\mid X=\mathrm{Nonsmokers}) = 0.5025 \notag
$$

$$
P(Y=\mathrm{No\ \ Cancer}\mid X=\mathrm{Smokers}) = 0.4525 \notag
$$

$$
P(Y=\mathrm{No\ \ Cancer}\mid X=\mathrm{Nonsmokers}) = 0.4975 \notag
$$



Thus, the causal effect of smoking on cancer is:


$$
\begin{align}\notag
& P(Y=\mathrm{Cancer}\mid X=\mathrm{Smokers}) - P(Y=\mathrm{Cancer}\mid X=\mathrm{Nonsmokers}) \\ \notag
& = 0.5475-0.5025\\ \notag
& =0.045,
\end{align}
$$


a $4.5\%$ increase in cancer risk due to smoking.



Similarly,  benefit of not smoking is:


$$
P(Y=\mathrm{No\ \ Cancer}\mid X=\mathrm{Nonsmokers}) - P(Y=\mathrm{No\ \ Cancer}\mid X=\mathrm{Smokers}) =0.045.\notag
$$


The result is consistent with the observed $5\%$ higher cancer rate in tar vs. no-tar groups (Table 2), validating the model.



### Conclusion

The key insight is that without modeling tar as a mediator, the raw data misleadingly suggest smoking reduces cancer risk. By introducing tar deposits as an intermediate cause, the front-door criterion allows us to isolate the true causal effect: smoking increases cancer risk by approximately 4.5%.

This example illustrates two critical lessons:

1. Causal models are essential—statistical associations alone can be misleading.
2. The front-door criterion provides a principled way to estimate causal effects, even in the presence of confounding.



### Appendix: Backdoor Criterion



The following is a brief derivation of the backdoor criterion as in Equation ($\ref{eqn:backdoor}$), which expresess $P(Y=y\mid\text{do}(Z=z))$ using observable probabilties.



Consider DAG in figure 2, where $Z$ is a confounder of $X$ and $Y$. To estimate $P(Y\mid \text{do}(X))$, we apply the back-door criterion by blocking all backdoor pathes from $X$ to $Y$. When we intervene on $X$, we cut the incoming edge edge $Z\to X$, resulting in the modified DAG of Figure 3.



<figure>
  <center>
  <img src="/assets/images/frontdoor_backdoor-SCM.svg" width="200">
   </center>
  <center>
    <figcaption> Figure 2. Original DAG.
    </figcaption>
  </center>
</figure>



<figure>
  <center>
  <img src="/assets/images/frontdoor_backdoor-SCM_modified.svg" width="200">
   </center>
  <center>
    <figcaption> Figure 3. Modified DAG after do(<em>X=x</em>).
    </figcaption>
  </center>
</figure>



Let $P_m$ denote probabilities in the modified DAG, then:


$$
\begin{align} \tag{A1}
P(y|\mathrm{do}(x)) & = P_m(y|x) \\ \label{eqn:bayes} \tag{A2}
& = \frac{P_m(y,x)}{P_m(x)} \\ \label{eqn:totality}\tag{A3}
& = \frac{\sum_z P_m(y,z,x)}{P_m(x)} \\ \label{eqn:bayes2}\tag{A4}
& = \frac{\sum_z P_m(y|z,x)P_m(z,x)}{P_m(x)}\\ \label{eqn:product}\tag{A5}
& = \frac{\sum_z  P_m(y|z,x) P_m(z) P_m(x)}{P_m(x)}\\ \label{eqn:cancel_px}\tag{A6}
& = \sum_z P_m(y|z, x) P_m(z)\\ \label{eqn:final}\tag{A7}
& = \sum_z P(y|z, x) P(z)
\end{align}
$$
Step-by Step Explanation

- (A1): Definition of intervention: condiitoning in the modfieid graph.
- (A2): Application of Bayes' rule.
- (A3): Expansion using the law of total probability over $Z$.
- (A4): Another application of Bayes' rule.
- (A5): $P_m(z,x)=P_m(z) P_m(x)$ because $Z$ and $X$ are independent in the modified DAG.
- (A6):  Cancellation of $P_m(x)$ from numerator and denominator.
- (A7): Replacement of $P_m$ terms with observable distributions in the original DAG. $P_m(z)=P(z)$ because intervening on $X$ does not affect how $Z$ is generated. $P_m(y\mid z, x)=P(y\mid z, x)$ becasue the conditional dependence of $Y$ on $(Z, X)$ remains unchanged.



### References

Pearl J., Glymour, M., and Jewell, P.  (2016). Causal Inference in Statistics: A Primer. John Wiley & Sons Ltd





