# pybuc
`pybuc` ((Py)thon (B)ayesian (U)nobserved (C)omponents) is a feature-limited version of R's Bayesian structural time series package, `bsts`, written by Steven L. Scott. The source paper can be found [here](https://people.ischool.berkeley.edu/~hal/Papers/2013/pred-present-with-bsts.pdf) or in the *papers* directory of this repository. While there are plans to expand the feature set of `pybuc`, currently there is no roadmap for the release of new features. The current version of `pybuc` includes the following options for modeling and forecasting a structural time series: 


- Stochastic or non-stochastic level
- Stochastic or non-stochastic slope (assuming a level state is specified)
- Multiple stochastic or non-stochastic "dummy" seasonality
- Multiple stochastic or non-stochastic trigonometric seasonality
- Regression with static coefficients


Note that the way `pybuc` estimates regression coefficients is methodologically different than `bsts`. The former uses a standard Gaussian prior, whereas the latter uses a Bernoulli-Gaussian mixture commonly known as the spike-and-slab prior. The main benefit of using a spike-and-slab prior is its promotion of coefficient-sparse solutions, i.e., variable selection, when the number of predictors in the regression component exceeds the number of observed data points.

Fast computation is achieved using [Numba](https://numba.pydata.org/), a high performance just-in-time (JIT) compiler for Python.

# Model

A structural time series model with level, trend, seasonal, and regression components takes the form: 

$$
y_t = \mu_t + \gamma_t + \mathbf x_t^\prime \boldsymbol{\beta} + \epsilon_t
$$ 

where $\mu_t$ specifies an unobserved dynamic level component, $\gamma_t$ an unobserved dynamic seasonal component, $\mathbf x_t^\prime \boldsymbol{\beta}$ a partially unobserved regression component (the regressors $\mathbf x_t$ are observed, but the coefficients $\boldsymbol{\beta}$ are not), and $\epsilon_t \sim N(0, \sigma_{\epsilon}^2)$ an unobserved irregular component. The equation describing the outcome $y_t$ is commonly referred to as the observation equation, and the transition equations governing the evolution of the unobserved states are known as the state equations.

## Level and slope

The unobserved level evolves according to the following general transition equations:

$$
\begin{align}
    \mu_{t+1} &= \mu_t + \delta_t + \eta_{\mu, t} \\ 
    \delta_{t+1} &= \delta_t + \eta_{\delta, t} 
\end{align}
$$ 

where $\eta_{\mu, t} \sim N(0, \sigma_{\eta_\mu}^2)$ and $\eta_{\delta, t} \sim N(0, \sigma_{\eta_\delta}^2)$ for all $t$. The state equation for $\delta_t$ represents the local slope at time $t$. If $\sigma_{\eta_\mu}^2 = \sigma_{\eta_\delta}^2 = 0$, then the level component in the observation equation, $\mu_t$, collapses to a deterministic intercept and linear time trend.

## Seasonality

### Dummy form
The seasonal component, $\gamma_t$, can be modeled in two ways. One way is known as the "dummy" variable approach. Formally, the seasonal effect on the outcome $y$ is modeled as 

$$
\sum_{j=0}^{S-1} \gamma_{t-j} = \eta_{\gamma, t} \iff \gamma_t = -\sum_{j=1}^{S-1} \gamma_{t-j} + \eta_{\gamma, t},
$$ 

where $j$ indexes the number of periods in a seasonal cycle, $S$ is the number of periods in a seasonal cycle, and $\eta_{\gamma, t} \sim N(0, \sigma_{\eta_\gamma}^2)$ for all $t$. Intuitively, if a time series exhibits periodicity, then the sum of the periodic effects over a cycle should, on average, be zero.

### Trigonometric form
Another way to model seasonality is through a trigonometric representation, which exploits the periodicity of sine and cosine functions. Specifically, seasonality is modeled as

$$
\gamma_t = \sum_{j=1}^h \gamma_{j, t}
$$

where $j$ indexes the number of harmonics to represent seasonality of periodicity $S$ and $1 \leq h \leq \lfloor S/2 \rfloor$ is the highest desired number of harmonics. The state transition equations for each harmonic, $\gamma_{j, t}$, are represented by a real and imaginary part, specifically

$$
\begin{align}
    \gamma_{j, t+1} &= \cos(\lambda_j) \gamma_{j, t} + \sin(\lambda_j) \gamma_{j, t}^* + \eta_{\gamma_j, t} \\
    \gamma_{j, t+1}^* &= -\sin(\lambda_j) \gamma_{j, t} + \cos(\lambda_j) \gamma_{j, t}^* + \eta_{\gamma_j^* , t}
\end{align}
$$

where frequency $\lambda_j = 2j\pi / S$. It is assumed that $\eta_{\gamma_j, t}$ and $\eta_{\gamma_j^ * , t}$ are distributed $N(0, \sigma^2_{\eta_\gamma})$ for all $j, t$.

## Regression
There are two ways to configure the model matrices to account for a regression component with static coefficients. The most common way (Method 1) is to append $\mathbf x_t^\prime$ to $\mathbf Z_t$ and $\boldsymbol{\beta}_t$ to the state vector, $\boldsymbol{\alpha}_t$ (see state space representation below), with the constraints $\boldsymbol{\beta}_0 = \boldsymbol{\beta}$ and $\boldsymbol{\beta}_t = \boldsymbol{\beta}_{t-1}$ for all $t$. Another, less common way (Method 2) is to append $\mathbf x_t^\prime \boldsymbol{\beta}$ to $\mathbf Z_t$ and 1 to the state vector. 

While both methods can be accommodated by the Kalman filter, Method 1 is a direct extension of the Kalman filter as it maintains the observability of $\mathbf Z_t$ and treats the regression coefficients as unobserved states. Method 2 does not fit naturally into the conventional framework of the Kalman filter, but it offers the significant advantage of only increasing the size of the state vector by one. In contrast, Method 1 increases the size of the state vector by the size of $\boldsymbol{\beta}$. This is significant because computational complexity is quadratic in the size of the state vector but linear in the size of the observation vector.

The unobservability of $\mathbf Z_t$ under Method 2 can be handled with maximum likelihood or Bayesian estimation by working with the adjusted series 

$$
y_t^* \equiv y_t - \tau_t = \mathbf x_ t^\prime \boldsymbol{\beta} + \epsilon_t
$$

where $\tau_t$ represents the time series component of the structural time series model. For example, assuming a level and seasonal component are specified, this means an initial estimate of the time series component $\tau_t = \mu_t + \gamma_t$ and $\boldsymbol{\beta}$ has to be acquired first. Then $\boldsymbol{\beta}$ can be estimated conditional on $\mathbf y^ * \equiv \left(\begin{array}{cc} y_1^ * & y_2^ * & \cdots & y_n^ * \end{array}\right)^\prime$.

`pybuc` uses Method 2 for estimating static coefficients.

## State space representation (example)
The unobserved components model can be rewritten in state space form. For example, suppose level, slope, seasonal, regression, and irregular components are specified, and the seasonal component takes a trigonometric form with periodicity $S=4$ and $h=2$ harmonics. Let $\mathbf Z_t \in \mathbb{R}^{1 \times m}$, $\mathbf T \in \mathbb{R}^{m \times m}$, $\mathbf R \in \mathbb{R}^{m \times q}$, and $\boldsymbol{\alpha}_ t \in \mathbb{R}^{m \times 1}$ denote the observation matrix, state transition matrix, state error transformation matrix, and unobserved state vector, respectively, where $m$ is the number of state equations and $q$ is the number of state parameters to be estimated (i.e., the number of stochastic state equations, which is defined by the number of positive state variance parameters). 

There are $m = 1 + 1 + h * 2 + 1 = 7$ state equations and $q = 1 + 1 + h * 2 = 6$ stochastic state equations. There are 6 stochastic state equations because the state value for the regression component is not stochastic; it is 1 for all $t$ by construction. The observation, state transition, and state error transformation matrices may be written as

$$
\begin{align}
    \mathbf Z_t &= \left(\begin{array}{cc} 
                        1 & 0 & 1 & 0 & 1 & 0 & \mathbf x_t^{\prime} \boldsymbol{\beta}
                        \end{array}\right) \\
    \mathbf T &= \left(\begin{array}{cc} 
                        1 & 1 & 0 & 0 & 0 & 0 & 0 \\
                        0 & 1 & 0 & 0 & 0 & 0 & 0 \\
                        0 & 0 & \cos(2\pi / 4) & \sin(2\pi / 4) & 0 & 0 & 0 \\
                        0 & 0 & -\sin(2\pi / 4) & \cos(2\pi / 4) & 0 & 0 & 0 \\
                        0 & 0 & 0 & 0 & \cos(4\pi / 4) & \sin(4\pi / 4) & 0 \\
                        0 & 0 & 0 & 0 & -\sin(4\pi / 4) & \cos(4\pi / 4) & 0 \\
                        0 & 0 & 0 & 0 & 0 & 0 & 1
                        \end{array}\right) \\
    \mathbf R &= \left(\begin{array}{cc} 
                    1 & 0 & 0 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 & 0 & 0 \\
                    0 & 0 & 1 & 0 & 0 & 0 \\
                    0 & 0 & 0 & 1 & 0 & 0 \\
                    0 & 0 & 0 & 0 & 1 & 0 \\
                    0 & 0 & 0 & 0 & 0 & 1 \\
                    0 & 0 & 0 & 0 & 0 & 0
                    \end{array}\right)
\end{align}
$$

Given the definitions of $\mathbf Z_t$, $\mathbf T$, and $\mathbf R$, the state space representation of the unobserved components model above can compactly be expressed as

$$
\begin{align}
    y_t &= \mathbf Z_t \boldsymbol{\alpha}_ t + \epsilon_t \\
    \boldsymbol{\alpha}_ {t+1} &= \mathbf T \boldsymbol{\alpha}_ t + \mathbf R \boldsymbol{\eta}_ t, \hspace{5pt} t=1,2,...,n
\end{align}
$$

where

$$
\begin{align}
    \boldsymbol{\alpha}_ t &= \left(\begin{array}{cc} 
                            \mu_t & \delta_t & \gamma_{1, t} & \gamma_{1, t}^* & \gamma_{2, t} & \gamma_{2, t}^* & 1
                            \end{array}\right)^\prime \\
    \boldsymbol{\eta}_ t &= \left(\begin{array}{cc} 
                            \eta_{\mu, t} & \eta_{\delta, t} & \eta_{\gamma_ 1, t} & \eta_{\gamma_ 1^\*, t} & \eta_{\gamma_ 2, t} & \eta_{\gamma_ 2^\*, t}
                            \end{array}\right)^\prime
\end{align}
$$

and 

$$
\mathrm{Cov}(\boldsymbol{\eta}_ t) = \mathrm{Cov}(\boldsymbol{\eta}_ {t-1}) = \boldsymbol{\Sigma}_ \eta =  \mathrm{diag}(\sigma^2_{\eta_\mu}, \sigma^2_{\eta_\delta}, \sigma^2_{\eta_{\gamma_ 1}}, \sigma^2_{\eta_{\gamma_ 1^\*}}, \sigma^2_{\eta_{\gamma_ 2}}, \sigma^2_{\eta_{\gamma_ 2^\*}}) \in \mathbb{R}^{6 \times 6} \hspace{5pt} \textrm{for all } t=1,2,...,n
$$

# Estimation
`pybuc` mirrors R's `bsts` with respect to estimation method. The observation vector, state vector, and regression coefficients are assumed to be conditionally normal random variables, and the error variances are assumed to be conditionally independent inverse-Gamma random variables. These model assumptions imply conditional conjugacy of the model's parameters. Consequently, a Gibbs sampler is used to sample from each parameter's posterior distribution.

To achieve fast sampling, `pybuc` follows `bsts`'s adoption of the Durbin and Koopman (2002) simulation smoother. For any parameter $\theta$, let $\theta(s)$ denote the $s$-th sample of parameter $\theta$. Each sample $s$ is drawn by repeating the following three steps:

1. Draw $\boldsymbol{\alpha}(s)$ from $p(\boldsymbol{\alpha} | \mathbf y, \boldsymbol{\sigma}^2_\eta(s-1), \boldsymbol{\beta}(s-1), \sigma^2_\epsilon(s-1))$ using the Durbin and Koopman simulation state smoother, where $\boldsymbol{\alpha}(s) = (\boldsymbol{\alpha}_ 1(s), \boldsymbol{\alpha}_ 2(s), \cdots, \boldsymbol{\alpha}_ n(s))^\prime$ and $\boldsymbol{\sigma}^2_\eta(s-1) = \mathrm{diag}(\boldsymbol{\Sigma}_ \eta(s-1))$. Note that `pybuc` implements a correction (based on a potential misunderstanding) for drawing $\boldsymbol{\alpha}(s)$ per "A note on implementing the Durbin and Koopman simulation smoother" (Marek Jarocinski, 2015).
2. Draw $\boldsymbol{\sigma}^2(s) = (\sigma^2_ \epsilon(s), \boldsymbol{\sigma}^2_ \eta(s))^\prime$ from $p(\boldsymbol{\sigma}^2 | \mathbf y, \boldsymbol{\alpha}(s), \boldsymbol{\beta}(s-1))$ using Durbin and Koopman's simulation disturbance smoother.
3. Draw $\boldsymbol{\beta}(s)$ from $p(\boldsymbol{\beta} | \mathbf y^ *, \boldsymbol{\alpha}(s), \sigma^2_\epsilon(s))$, where $\mathbf y^ *$ is defined above.

By assumption, the elements in $\boldsymbol{\sigma}^2(s)$ are conditionally independent inverse-Gamma distributed random variables. Thus, Step 2 amounts to sampling each element in $\boldsymbol{\sigma}^2(s)$ independently from their posterior inverse-Gamma distributions.
