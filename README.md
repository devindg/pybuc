# pybuc
<code/>pybuc</code> ((Py)thon (B)ayesian (U)nobserved (C)omponents) is a feature-limited version of R's Bayesian structural time series package, <code/>bsts</code>, written by Steven L. Scott. While there are plans to expand the feature set of <code/>pybuc</code>, currently there is no roadmap for the release of new features. The current version of <code/>pybuc</code> includes the following model options: 

<ul>
    <li>Stochastic or non-stochastic level</li>
    <li>Stochastic or non-stochastic slope (assuming a level state is specified)</li>
    <li>Stochastic or non-stochastic "dummy" seasonality</li>
    <li>Multiple stochastic or non-stochastic trigonometric seasonality</li>
    <li>Regression with static coefficients</li>
</ul>

A model with level, trend, seasonal, and regression components takes the form:

$$
y_t = \mu_t + \gamma_t + \mathbf x_t \boldsymbol{\beta} + \epsilon_t \\
$$

where $\mu_t$ specifies the dynamic level component, $\gamma_t$ the dynamic seasonal component, $\mathbf x_t \boldsymbol{\beta}$ the regression component, and $\epsilon_t \sim N(0, \sigma_{\epsilon}^2)$ the irregular component. The level component takes the general form

$$
\begin{align}
    \mu_{t+1} &= \mu_t + \delta_t + \eta_{\mu, t} \\
    \delta_{t+1} &= \delta_t + \eta_{\delta, t}
\end{align}
$$
