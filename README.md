# pybuc
<code/>pybuc</code> ((Py)thon (B)ayesian (U)nobserved (C)omponents) is presently a feature-limited version of R's Bayesian structural time series package, <code/>bsts</code>, written by Steven L. Scott. While there are plans to expand the feature set of <code/>pybuc</code>, currently there is no roadmap for the release of new features. The current version of <code/>pybuc</code> includes the following model options: 

<ul>
    <li>Stochastic or non-stochastic level</li>
    <li>Stochastic or non-stochastic slope (assuming a level state is specified)</li>
    <li>Stochastic or non-stochastic "dummy" seasonality</li>
    <li>Multiple stochastic or non-stochastic trigonometric seasonality</li>
    <li>Regression with static coefficients</li>
</ul>

In addition to a relatively limited feature set, the way <code/>pybuc</code> estimates regression coefficients is methodologically different than <code/>bsts</code>. The former uses a standard Gaussian prior, whereas the latter uses a Bernoulli-Gaussian mixture known as the spike-and-slab prior. The main benefit of using a spike-and-slab prior is its promotion of coefficient-sparse solutions, i.e., variable selection, when degrees of freedom are negative.

# Model
A structural time series model with level, trend, seasonal, and regression components takes the form:

$$
y_t = \mu_t + \gamma_t + \mathbf x_t^{\prime} \boldsymbol{\beta} + \epsilon_t \\
$$

where $\mu_t$ specifies an unobserved dynamic level component, $\gamma_t$ an unobserved dynamic seasonal component, $\mathbf x_t^\prime \boldsymbol{\beta}$ an unobserved regression component with respect to the coefficients, $\boldsymbol{\beta}$, and $\epsilon_t \sim N(0, \sigma_{\epsilon}^2)$ an unobserved irregular component. The equation describing the outcome $y_t$ is commonly referred to as the observation equation, and the transition equations governing the evolution of the unobserved states are known as the state equations.

The unobserved level evolves according to the following general transition equations:

$$
\begin{align}
    \mu_{t+1} &= \mu_t + \delta_t + \eta_{\mu, t} \\
    \delta_{t+1} &= \delta_t + \eta_{\delta, t}
\end{align}
$$

where $\eta_{\mu, t} \sim N(0, \sigma_{\eta_\mu}^2)$ and $\eta_{\delta, t} \sim N(0, \sigma_{\eta_\delta}^2)$. The state described by $\delta_t$ governs the local slope at time $t$. If $\sigma_{\eta_\mu}^2 = \sigma_{\eta_\delta}^2 = 0$, then the level component in the observation equation, $\mu_t$, collapses to a deterministic intercept and linear time trend.
