---

title: "Mean and Variance Network"
date: 2025-02-07
typora-root-url: ./..
---



We use a feedforward neural network to analyze heteroscedastic data in nonlinear regression. The network is trained in two steps, each using a different loss function, to estimate the dependent variable's mean and variance. 



### Data

We generate data that displays heteroscedasticity for the regression. The dependent variable $y$ is affected by the predictor $x$, with the noise $\epsilon$ coming from a normal distribution in which the variance varies based on $x$. 


$$
y = \mu(x) + \mathcal{N}(0, \sigma^2(x)), \label{eq_y_vs_x}
$$


Where 


$$
\begin{align} \notag
\mu(x) = & e^{2x}\\ 
\sigma(x)= &0.4 e^{2x}. \notag
\end{align}
$$


The following Wolfram language code generates data with 1000 samples.

```mathematica
SeedRandom[1234];
x = RandomVariate[NormalDistribution[], 1000];
y0 = Exp[2*x];
y = y0 + 0.4*y0*RandomVariate[NormalDistribution[], 1000]; (* sigma depends on x *)
y = Standardize[y]; (* standardize y for better performance *)
```



Figure 1 shows the relationships between the dependent variable $y$ (gray dots), as well as the mean $\mu$ and standard deviation $\sigma$  (solid lines),  and the predictor $x$.



<figure>
  <center>
  <img src="/assets/images/mvnet_data_mu_sigma.svg" width="500">
   </center>
  <center>
    <figcaption> Figure 1. Simulated data (gray dots) for regression. The solid lines are the mean &mu; and variation &sigma; as functions of <i>x</i>.
    </figcaption>
  </center>
</figure>





### Maximum Likelihood Estimate (MLE) of $\mu$ and $\sigma$



We use maximum likelihood method to estimate $\mu$ and $\sigma$ in Equation ($\ref{eq_y_vs_x}$). For observed data $(x_i, y_i)$,  the likelihood is given by


$$
L = \prod_i p(y_i\mid \mu_i, \sigma_i), \notag
$$


where $p$ is the probability density function (PDF) of the normal distribution:


$$
p(y_i\mid \mu_i, \sigma_i)=\frac{1}{\sqrt{2\pi}\sigma_i}\exp\left(-\frac{(y_i-\mu_i)^2}{2\sigma_i^2}\right). \notag
$$


To maximize the likelihood $L$, we minimize the negative of the logarithm of $L$, or $-\log L$.


$$
-\log L =\left( \sum_i \log\sigma_i + \frac{(y_i-\mu_i)^2}{2\sigma_i^2}\right), \label{eq_logL_both}
$$


where we have excluded the constant term that does not depends on $\sigma_i$ and $\mu_i$. 



In case where the $\sigma$ is assumed to be independent of $x$,  we have:


$$
-\log L \propto \sum_i (y_i - \mu_i)^2. \label{eq_logL_mean}
$$


In this situation, the maximum likelihood estimate aligns with the least squares regression method.





### MLE with a Feedforward Neural Network



The structure of the feedforward neural network is illustrated in Figure 2. 



<figure>
  <center>
  <img src="/assets/images/mvnet_net_out.svg" width="600">
   </center>
  <center>
    <figcaption> Figure 1. The feedforward neural network. <i>Tanh</i> activation is used after the first linear layer. The output is a vector of size 2, where the first element is &mu; and the second element is &sigma;.
    </figcaption>
  </center>
</figure>



The loss function calculates the negative logarithm of the likelihood for two cases: 

(a) When both $\mu$ and $\sigma$ are included, which corresponds to Equation ($\ref{eq_logL_both}$).

(b) When only $\mu$ is included, corresponding to Equation ($\ref{eq_logL_mean}$). 



The definition of the loss function in the Wolfram language is presented below,  and the corresponding diagrams are shown in Figure 2.



```mathematica
loss[type_] := Module[{func},
    func = Switch[type, "Both", #2 + (#3-#1)^2*Exp[-2*#2]/2&, "Mean", (#3-#1)^2&];
    NetGraph[
        <| "\[Mu]"->PartLayer[1],
           "\[Simga]"->PartLayer[2],
           "loss"->ThreadingLayer[func] |>,
          { { "[\Mu]", "\[Sigma]", NetPort["Target"]}->"loss"}, "Input"->2]
          ]
```





<figure>
  <center>
  <img src="/assets/images/mvnet_losslayer.svg" width="600">
   </center>
  <center>
    <figcaption> Figure 2. Loss function: (a) when both &mu; and &sigma; are included, and (b) when only &mu; is incldued.
    </figcaption>
  </center>
</figure>



### Result

We train the network in two approaches: (1) estimating $\mu$ and $\sigma$ simultaneously in a single training session and (2) training in two steps, each with a different loss function.



#### 1. Simultaneous Estimating $\mu$ and $\sigma$ in a Single Training

In the first approach, we train the neural network to estimate $\mu$ and $\sigma$ simultaneously, using a loss function that incorporates both parameters.  However, as shown in Figure 3, this approcah performs poorly, as neither estiamted $\mu$ nor $\sigma$ converges to the actual values.  



<figure>
  <center>
  <img src="/assets/images/mvnet_train_both.svg" width="800">
   </center>
  <center>
    <figcaption> Figure 3. Results of training the network to estimate &mu; and &sigma; simultaneously. Both estmated &mu; and &sigma; deviate significnatly from the true values.
    </figcaption>
  </center>
</figure>



This issue has been noted in previous studies. One proposed solution is to train the network in two steps. In the first step, the network is trained to estimate $\mu$ only, assuming a constant $\sigma$. This initial phase is known as the "warm-up" training.  In the second step, the network is further trained for both $\mu$ and $\sigma$. 



#### 2. Two-Step Training

The results of the first step, where only $\mu$ is estimated, are shown in Figure 4. The estimated $\mu$ is close to the actual value,  but $\sigma$ is completely inaccurate, as expected since the loss function in this step assumes a constant $\sigma$. 

<figure>
  <center>
  <img src="/assets/images/mvnet_train_mu_step1.svg" width="800">
   </center>
  <center>
    <figcaption> Figure 4. Results of Step 1 training. &mu; is close to the actual value,  but &sigma; is entirely inaccurate. 
    </figcaption>
  </center>
</figure>



In Step 2, the trained network is further trained by including both $\mu$ and $\sigma$ in the loss function and estimate them simultaneously. Figure 5 shows that both $\mu$ and $\sigma$ align well with their actual values after this training phase. 

<figure>
  <center>
  <img src="/assets/images/mvnet_train_both_step2.svg" width="800">
   </center>
  <center>
    <figcaption> Figure 5. After Step 2 training, both &mu; and &sigma; converge to their actual values.
    </figcaption>
  </center>
</figure>



Through the two-step training process, the neural network achieves accurate estimates for both $\mu$ and $\sigma$.



### Code

The Wolfram language code for the 

```mathematica
(***** generate data with heteroscedacity for regression *****)
SeedRandom[1234];
x = RandomVariate[NormalDistribution[], 1000];
y0 = Exp[2*x];
y = y0 + 0.4*y0*RandomVariate[NormalDistribution[], 1000]; (* sigma depends on x *)
y = Standardize[y]; (* standardize y for better performance *)

(* data for training *)
data = <|"Input"->x, "Output"->Y|>;


(***** feedforward network *****)
net = NetChain[{20, Tanh, 2}, "Input"->"Real"];

(***** Loss Function *****)
loss[type_] := Module[{func},
    func = Switch[type, "Both", #2 + (#3-#1)^2*Exp[-2*#2]/2&, "Mean", (#3-#1)^2&];
    NetGraph[
        	<| "\[Mu]"->PartLayer[1],
           "\[Simga]"->PartLayer[2],
           "loss"->ThreadingLayer[func] |>,
          { { "[\Mu]", "\[Sigma]", NetPort["Target"]}->"loss"}, "Input"->2]
          ]
          
(***** training using loss function that includes both mu and sigma *****)
resultBoth = NetTrain[net, data, All, LossFunction->loss["Both"]];

(***** two step training *****)
(* Step 1: training using loss function with mu only *)
resultStep1 = NetTrain[net, data, All, LossFunction->loss["Mean"]];
(* Step 2: further training trained net from Step 1 using loss function with both mu and sigma *)
trainedNet = resultStep1["TrainedNet"];
resultStep2 = NetTrain[trainedNet, data, All, LossFunction->loss["Both"]];
```



### References

Sluijterman L., Cator, E., and Heskes, T.  (2023). Optimal Training of Mean Variance Estimation Neural Networks. https://arxiv.org/pdf/2302.08875



