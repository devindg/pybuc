# pybuc
<code/>pybuc</code> ((Py)thon (B)ayesian (U)nobserved (C)omponents) is presently a feature-limited version of R's Bayesian structural time series package, <code/>bsts</code>, written by Steven L. Scott. While there are plans to expand the feature set of <code/>pybuc</code>, currently there is no roadmap for the release of new features. The current version of <code/>pybuc</code> includes the following options for modeling and forecasting a structural time series: 

<ul>
    <li>Stochastic or non-stochastic level</li>
    <li>Stochastic or non-stochastic slope (assuming a level state is specified)</li>
    <li>Stochastic or non-stochastic "dummy" seasonality</li>
    <li>Multiple stochastic or non-stochastic trigonometric seasonality</li>
    <li>Regression with static coefficients</li>
</ul>

In addition to a relatively limited feature set, the way <code/>pybuc</code> estimates regression coefficients is methodologically different than <code/>bsts</code>. The former uses a standard Gaussian prior, whereas the latter uses a Bernoulli-Gaussian mixture known as the spike-and-slab prior. The main benefit of using a spike-and-slab prior is its promotion of coefficient-sparse solutions, i.e., variable selection, when the number of predictors in the regression component exceeds the number of observed data points.

# Model
A structural time series model with level, trend, seasonal, and regression components takes the form:

$$
y_t = \mu_t + \gamma_t + \mathbf x_t^{\prime} \boldsymbol{\beta} + \epsilon_t \\
$$

where $\mu_t$ specifies an unobserved dynamic level component, $\gamma_t$ an unobserved dynamic seasonal component, $\mathbf x_t^\prime \boldsymbol{\beta}$ an unobserved regression component with respect to the coefficients, $\boldsymbol{\beta}$, and $\epsilon_t \sim N(0, \sigma_{\epsilon}^2)$ an unobserved irregular component. The equation describing the outcome $y_t$ is commonly referred to as the observation equation, and the transition equations governing the evolution of the unobserved states are known as the state equations.

## Level and trend
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
Another way to model seasonality is through a trigonometric represenation, which exploits the periodicity of sine and cosine functions. Specifically, seasonality is modeled as

$$
\gamma_t = \sum_{j=1}^h \gamma_{j, t}
$$

where $j$ indexes the number of harmonics to represent seasonality of periodity $S$ and $1 \leq h \leq \lfloor S/2 \rfloor$ is the highest desired number of harmonics. The state transition equations for each harmonic, $\gamma_{j, t}$, are represented by a real and imaginary part, specifically

$$
\begin{align}
    \gamma_{j, t+1} &= \cos(\lambda_j) \gamma_{j, t} + \sin(\lambda_j) \gamma_{j, t}^* + \eta_{\gamma_j, t} \\
    \gamma_{j, t+1}^* &= -\sin(\lambda_j) \gamma_{j, t} + \cos(\lambda_j) \gamma_{j, t}^* + \eta_{\gamma_j^* , t}
\end{align}
$$

where frequency $\lambda_j = 2j\pi / S$. It is assumed that $\eta_{\gamma_j, t}$ and $\eta_{\gamma_j^* , t}$ are distributed $N(0, \sigma^2_{\eta_\gamma})$ for all $j, t$.

## State space representation
The unobserved components model can be rewritten in state space form. For example, suppose level, slope, seasonal, regression, and irregular components are specified, and the seasonal component takes a trigonometric form. Let $\mathbf Z_t \in \mathbb{R}^{1 \times m}$, $\mathbf T \in \mathbb{R}^{m \times m}$, and $\mathbf R \in \mathbb{R}^{m \times u}$ denote the observation, state transition, and state error transformation matrices, respectively, where $m$ is the number of state equations and $u$ is number of state parameters to be estimated. 
