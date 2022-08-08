# pybuc
<code/>pybuc</code> ((Py)thon (B)ayesian (U)nobserved (C)omponents) is a feature-limited version of R's Bayesian structural time series package, <code/>bsts</code>, written by Steven L. Scott. While there are plans to expand the feature set of <code/>pybuc</code>, currently there is no roadmap for the release of new features. The current version of <code/>pybuc</code> includes the following model options: 

<ul>
    <li>Stochastic or non-stochastic level</li>
    <li>Stochastic or non-stochastic slope (assuming a level state is specified)</li>
    <li>Stochastic or non-stochastic "dummy" seasonality</li>
    <li>Multiple stochastic or non-stochastic trigonometric seasonality</li>
    <li>Static regression</li>
</ul>

A model with level, trend, seasonal, and regression components takes the form:

$$
\begin{align}
y_t &= \mu_t + \gamma_t + \mathbf{x}_t^{\prime} \boldsymbol{\beta} + \epsilon_t \\
\mu_{t+1} &= \mu_t + \delta_t + \eta_{\mu, t} \\
\delta_{t+1} &= \delta_t + \eta_{\delta, t}
\end{align}
$$
