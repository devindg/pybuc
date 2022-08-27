from pybuc import buc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.structural import UnobservedComponents


# Convenience function for computing root mean squared error
def rmse(actual, prediction):
    act, pred = actual.flatten(), prediction.flatten()
    return np.sqrt(np.mean((act - pred) ** 2))


# Simulate data
seed = 123
np.random.seed(seed)
hold_out_size = 10
n = 100
veps = 50
veta = 200
beta = np.array([[3.15, -10.24]]).T
eps = np.random.normal(0, np.sqrt(veps), size=n)
eta = np.random.normal(0, np.sqrt(veta), size=n)
x1 = np.random.normal(50, 10, size=n)
x2 = np.random.normal(100, 40, size=n)
x = np.c_[x1, x2]
Z = 1.
T = 1.
R = 1.
mu = np.empty(n + 1)
mu[0] = 1500
y = np.empty(n)
for t in range(n):
    mu[t + 1] = mu[t] + eta[t]

for t in range(n):
    y[t] = mu[t] + x[t].dot(beta) + eps[t]

y = y.reshape(-1, 1)
x = np.atleast_2d(x)

# Create train and test sets
y_train = y[:-hold_out_size, :]
x_train = x[:-hold_out_size, :]
y_test = y[-hold_out_size:, :]
x_test = x[-hold_out_size:, :]

if __name__ == '__main__':
    ''' Fit the airline data using SARIMA(0,1,1)(0,1,1) '''
    sarima = SARIMAX(y_train, exog=x_train, order=(0, 1, 1), trend=[0])
    sarima_res = sarima.fit(disp=False)
    print(sarima_res.summary())

    # Plot in-sample fit against actuals
    plt.plot(y_train)
    plt.plot(sarima_res.fittedvalues)
    plt.title('SARIMA Airline: In-sample')
    plt.show()

    # Get and plot forecast
    sarima_forecast = sarima_res.get_forecast(hold_out_size, exog=x_test).summary_frame(alpha=0.05)
    plt.plot(y_test)
    plt.plot(sarima_forecast['mean'])
    plt.fill_between(sarima_forecast.index,
                     sarima_forecast['mean_ci_lower'],
                     sarima_forecast['mean_ci_upper'], alpha=0.2)
    plt.title('SARIMA: Forecast')
    plt.legend(['Actual', 'Mean', '95% Confidence Interval'])
    plt.show()

    # Print RMSE
    print(f"SARIMA RMSE: {rmse(y_test.flatten(), sarima_forecast['mean'].to_numpy())}")

    ''' Fit the airline data using MLE unobserved components '''
    mle_uc = UnobservedComponents(y_train, exog=x_train, irregular=True,
                                  level=True, stochastic_level=True)

    # Fit the model via maximum likelihood
    mle_uc_res = mle_uc.fit(disp=False)
    print(mle_uc_res.summary())

    # Plot in-sample fit against actuals
    plt.plot(y_train)
    plt.plot(mle_uc_res.fittedvalues)
    plt.title('MLE UC: In-sample')
    plt.show()

    # Plot time series components
    mle_uc_res.plot_components(legend_loc='lower right', figsize=(15, 9), which='smoothed')
    plt.show()

    # Get and plot forecast
    mle_uc_forecast = mle_uc_res.get_forecast(hold_out_size, exog=x_test).summary_frame(alpha=0.05)
    plt.plot(y_test)
    plt.plot(mle_uc_forecast['mean'])
    plt.fill_between(mle_uc_forecast.index,
                     mle_uc_forecast['mean_ci_lower'],
                     mle_uc_forecast['mean_ci_upper'], alpha=0.2)
    plt.title('MLE UC: Forecast')
    plt.legend(['Actual', 'Mean', '95% Confidence Interval'])
    plt.show()

    # Print RMSE
    print(f"MLE UC RMSE: {rmse(y_test.flatten(), mle_uc_forecast['mean'].to_numpy())}")

    ''' Fit the airline data using Bayesian unobserved components '''
    buc.set_seed(seed)
    bayes_uc = buc.BayesianUnobservedComponents(response=y_train,
                                                level=True, stochastic_level=True,
                                                predictors=x_train)
    post = bayes_uc.sample(5000)
    mcmc_burn = 100

    # Print summary of estimated parameters
    for key, value in bayes_uc.summary(post, burn=mcmc_burn).items():
        print(key, ' : ', value)

    # Plot in-sample fit against actuals
    yhat = np.mean(post.filtered_prediction[mcmc_burn:], axis=0)
    plt.plot(y_train)
    plt.plot(yhat)
    plt.title('Bayesian-UC: In-sample')
    plt.show()

    # Plot time series components
    bayes_uc.plot_components(post, burn=mcmc_burn, smoothed=True)
    plt.show()

    # Get and plot forecast
    forecast = bayes_uc.forecast(post, hold_out_size, mcmc_burn, future_predictors=x_test)
    forecast_mean = np.mean(forecast, axis=0)
    forecast_l95 = np.quantile(forecast, 0.025, axis=0).flatten()
    forecast_u95 = np.quantile(forecast, 0.975, axis=0).flatten()

    plt.plot(y_test)
    plt.plot(forecast_mean)
    plt.fill_between(np.arange(forecast_mean.size), forecast_l95, forecast_u95, alpha=0.2)
    plt.title('Bayesian UC: Forecast')
    plt.legend(['Actual', 'Mean', '95% Confidence Interval'])
    plt.show()

    # Print RMSE
    print(f"BAYES-UC RMSE: {rmse(y_test, forecast_mean)}")

