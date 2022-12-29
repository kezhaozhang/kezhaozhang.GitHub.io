---
title: "Effect of Noise in Data On Regression: Linear Model vs. Neural Network"
typora-root-url: ./..
---



We have observed that the performance of the linear model for regression is equivalent to or better than more complex nonlinear models like the neural network in cases where the data is noisy. In this note, we compare a linear model and a feed-forward neural network for regression with various amounts of noise in the data. 



The result shows that when the noise is low, the neural network outperforms the linear model, but when the noise is high, the linear model becomes better.



This analysis suggests two approaches to improve regression performance: data cleaning and a more complex model. When the noise in data is low, improvement can be achieved with a more complex model. However,  such improvement diminishes as the noise gets stronger. For sufficiently noisy data, the simpler model can outperform a complex one. In this case, obtaining cleaner data, e.g., by reducing variations in the measurement system, is the most effective way to improve regression performance. 



## Data

We use generated data for the regression analysis according to the following equation:


$$
y = \beta_0 x_0 + \alpha x_0^2 + \beta_1 x_1 + \beta_2 x_2 + \mathcal{N}(0, \sigma^2), \label{eqn_y}
$$


where $\mathcal{N}(0,\sigma^2)$ is a random variable from a normal distribution with a variance of $\sigma^2$;  $\beta_0 = -1$, $\beta_1=-0.5$, $\beta_2=1.2$, $\alpha=1$, and $\sigma^2=0.5$.



Because of the quadratic term $\alpha x_0^2$,  the model of $y$ is nonlinear.

```python
import numpy as np
alpha = 1
beta = np.array([-1, -0.5, 1.2])
np.random.seed(89)
X = np.random.randn(1000, 3) 
y = np.matmul(X, beta) + alpha*X[:,0]**2 + np.random.randn(1000)*0.5
```



<figure>
  <center>
  <img src="/assets/images/ols_nnet_data_pairplots.svg" width="600">
   </center>
  <center>
  <figurecaption>
  Figure 1. Pairplots of the generated data. 
  </figurecaption>
  </center>
</figure>



### Regression Models

A linear regression model and a feedforward neural network are used to regress $y$ on $x_0, x_1$, and $x2$. The predictors  $x_0, x_1$, and $x2$ are standardized before being fed into the regressors. 



```python
import torch
from torch import nn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

class NNRegressor(nn.Module):
    def __init__(self, hidden_layer_sizes, activation='ReLU', max_iter = 1000, learning_rate=0.001 ):
        super().__init__()
        
        self.activation_dict = {'ReLU': nn.ReLU,
                                'Sigmod': nn.Sigmoid,
                                'Tanh': nn.Tanh
                               }
        
        self.max_iter = max_iter
       
        self.loss = nn.MSELoss()
        
 
            
        self.hidden_layer_sizes = hidden_layer_sizes
        
        self.stack = nn.ModuleList()
        self.activation = activation
        self.learning_rate = learning_rate
        
        self.fitted = False
        
    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        return x
    
    
    def fit(self, X, y):
        
        XX = torch.tensor(X, dtype=torch.float)
        yy = torch.tensor(y, dtype=torch.float).reshape(-1,1)
        if not self.fitted:  # haven't been fitted
            n_samples, n_features = XX.shape
    
            
            sizes = self.hidden_layer_sizes
            if hasattr(sizes, '__iter__'):
                layer_sizes = list(sizes)
            else:
                layer_sizes = [sizes]
           
            
            n_in = n_features
            for n in layer_sizes:
                n_out = n
                self.stack.append(nn.Linear(in_features=n_in, out_features=n_out))
                self.stack.append(self.activation_dict[self.activation]())
                n_in = n
            
            self.stack.append(nn.Linear(in_features=n, out_features=1)) #output layer
            
            self.optimizer = torch.optim.Adam(self.stack.parameters(), 	lr=self.learning_rate) # Optimizer
            
            self.fitted = True
        
        for i in range(self.max_iter):
            self.optimizer.zero_grad()
            loss = self.loss(self.forward(XX), yy)
            loss.backward()
            self.optimizer.step()
            
        return self
    
    def predict(self, X):
        if not self.fitted:
            print("Fit the model before predict.")
            return
        return self.forward(torch.tensor(X, dtype=torch.float)).detach().numpy()
    
    def get_params(self, deep=True):
        return {'hidden_layer_sizes': self.hidden_layer_sizes, 
                'max_iter': self.max_iter,
                'learning_rate': self.learning_rate,
                'activation': self.activation
               }

# models:
lm = make_pipeline(StandardScaler(), LinearRegression())
nnet = make_pipeline(StandardScaler(), 
                     NNRegressor(hidden_layer_sizes=(20,), activation='Tanh', max_iter=1000, learning_rate=0.01))
```



### Effect of Noise on Regression Performance

The data is made noisier by adding normally distributed noise to the predictors 


$$
X = \left [
\begin{array}{c} 
x_0\\
x_1\\
x_2
\end{array}
\right], \notag
$$


resulting in the new predictors: 


$$
XX = X + \mathcal{N}(0, noise). \notag
$$


Then we regress $y$ on $XX$.

```python
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from sklearn.model_selection import cross_validate
import pandas as pd

scorers = {'r2': make_scorer(r2_score), 'mse': make_scorer(mean_squared_error)}

scores = pd.DataFrame()
for noise in [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]:
    XX = X + np.random.randn(*X.shape)*noise
    
    print(f"noise={noise}")
    
    lm_scores = cross_validate(lm, XX, y, scoring=scorers, cv=5)
    
    nnet_scores = cross_validate(nnet, XX, y, scoring=scorers, cv=5)
 
    ols_r2 = lm_scores['test_r2']
    scores = pd.concat([pd.DataFrame({'noise': noise, 'model': 'ols', 'metric': 'r2', 'value': ols_r2}), scores])
    ols_rmse = np.sqrt(lm_scores['test_mse'])
    scores = pd.concat([pd.DataFrame({'noise': noise, 'model': 'ols', 'metric': 'rmse', 'value': ols_rmse}), scores])
    nnet_r2 = nnet_scores['test_r2']
    scores = pd.concat([pd.DataFrame({'noise': noise, 'model': 'nnet', 'metric': 'r2', 'value': nnet_r2}), scores])
    nnet_rmse = np.sqrt(nnet_scores['test_mse'])
    scores = pd.concat([pd.DataFrame({'noise': noise, 'model': 'nnet', 'metric': 'rmse', 'value': nnet_rmse}), scores])
 
```



The performance of the linear and the neural network regressors is evaluated with five-fold cross-validation.  Both root mean squared error and $R^2$ are calculated.



As shown in figure 2, when the noise $<1$, the neural network (nnet) outperforms the linear model (ols) with lower RMSE and higher $R^2$. However, then the noise is large ($>1$), the linear model is slightly better than the neural network.

<figure>
  <center>
  <img src="/assets/images/ols_nnet_rmse_r2_vs_noise.svg" width="750">
   </center>
  <center>
  <figurecaption>
    Figure 2. RMSE and R<sup>2</sup> for various amounts of noises. The dots are the mean of the 5-fold cross-validations, and the vertical bars represent a 95% interval. The neural network outperforms the linear model when noise is low (noise <1) but becomes worse than the linear model when noise is high (>1).
  </figurecaption>
  </center>
</figure>





In the low-noise region, the neural network performs better than the linear model because it captures the nonlinearity in the Equation ($\ref{eqn_y}$). However, when the noise is strong, the intrinsic nonlinearity in the data is overshadowed by the noise, and the neural network model overfits and becomes inferior to the linear model.

