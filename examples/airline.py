from pybuc import buc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.structural import UnobservedComponents
from pathlib import Path


# Convenience function for computing root mean squared error
def rmse(actual, prediction):
    return np.sqrt(np.mean((actual - prediction) ** 2))


# Import airline passenger data
air = pd.read_csv(Path('../examples/data/airline-passengers.csv'), header=0, index_col=0)
air = air.to_numpy().astype(float)
hold_out_size = 12

# Create train and test sets
y_train = air[:-hold_out_size, :]
y_test = air[-hold_out_size:, :]

if __name__ == '__main__':
    ''' Fit the airline data using SARIMA(0,1,1)(0,1,1) '''
    sarima = SARIMAX(y_train, order=(0, 1, 1),
                     seasonal_order=(0, 1, 1, 12),
                     trend=[0])
    sarima_res = sarima.fit(disp=False)
    print(sarima_res.summary())

    # Plot in-sample fit against actuals
    plt.plot(y_train)
    plt.plot(sarima_res.fittedvalues)
    plt.title('SARIMA: In-sample')
    plt.show()

    # Get and plot forecast
    sarima_forecast = sarima_res.get_forecast(hold_out_size).summary_frame(alpha=0.05)
    plt.plot(y_test)
    plt.plot(sarima_forecast['mean'])
    plt.plot(sarima_forecast['mean_ci_lower'])
    plt.plot(sarima_forecast['mean_ci_upper'])
    plt.title('SARIMA: Forecast')
    plt.legend(['Actual', 'Mean', 'LB', 'UB'])
    plt.show()

    # Print RMSE
    print(f"SARIMA RMSE: {rmse(y_test.flatten(), sarima_forecast['mean'].to_numpy())}")

    ''' Fit the airline data using MLE unobserved components '''
    mle_uc = UnobservedComponents(y_train, exog=None, irregular=True,
                                  level=True, stochastic_level=True,
                                  trend=True, stochastic_trend=True,
                                  freq_seasonal=[{'period': 12, 'harmonics': 6}],
                                  stochastic_freq_seasonal=[True])

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
    mle_uc_forecast = mle_uc_res.get_forecast(hold_out_size).summary_frame(alpha=0.05)
    plt.plot(y_test)
    plt.plot(mle_uc_forecast['mean'])
    plt.plot(mle_uc_forecast['mean_ci_lower'])
    plt.plot(mle_uc_forecast['mean_ci_upper'])
    plt.title('MLE UC: Forecast')
    plt.legend(['Actual', 'Mean', 'LB', 'UB'])
    plt.show()

    # Print RMSE
    print(f"MLE UC RMSE: {rmse(y_test.flatten(), mle_uc_forecast['mean'].to_numpy())}")

    ''' Fit the airline data using Bayesian unobserved components '''
    buc.set_seed(123)
    bayes_uc = buc.BayesianUnobservedComponents(outcome=y_train,
                                                level=True, stochastic_level=True,
                                                slope=True, stochastic_slope=True,
                                                seasonal=0, stochastic_seasonal=True,
                                                trig_seasonal=((12, 0), ), stochastic_trig_seasonal=(True, ))

    post = bayes_uc.sample(5000)
    mcmc_burn = 100

    # Print summary of estimated parameters
    mean_sig_obs = np.mean(post.outcome_error_variance[mcmc_burn:])
    mean_sig_lvl = np.mean(post.state_error_variance[:, 0, 0][mcmc_burn:])
    mean_sig_slope = np.mean(post.state_error_variance[:, 1, 1][mcmc_burn:])
    mean_sig_trig = np.mean(post.state_error_variance[:, 2, 2][mcmc_burn:])
    std_sig_obs = np.std(post.outcome_error_variance[mcmc_burn:])
    std_sig_lvl = np.std(post.state_error_variance[:, 0, 0][mcmc_burn:])
    std_sig_slope = np.std(post.state_error_variance[:, 1, 1][mcmc_burn:])
    std_sig_trig = np.std(post.state_error_variance[:, 2, 2][mcmc_burn:])

    print(f"sigma2.irregular: {mean_sig_obs} ({std_sig_obs}) \n"
          f"sigma2.level: {mean_sig_lvl} ({std_sig_lvl}) \n"
          f"sigma2.trend: {mean_sig_slope} ({std_sig_slope}) \n"
          f"sigma2.freq_seasonal_12(6): {mean_sig_trig} ({std_sig_trig})")

    # Plot in-sample fit against actuals
    yhat = np.mean(post.filtered_prediction[mcmc_burn:], axis=0)
    plt.plot(y_train.flatten())
    plt.plot(yhat)
    plt.title('Bayesian-UC: In-sample')
    plt.show()

    # Plot time series components
    bayes_uc.plot_components(post, burn=mcmc_burn, smoothed=True)

    # Get and plot forecast
    forecast = bayes_uc.forecast(post, hold_out_size, mcmc_burn)
    forecast_mean = np.mean(forecast, axis=0)
    forecast_l95 = np.quantile(forecast, 0.025, axis=0)
    forecast_u95 = np.quantile(forecast, 0.975, axis=0)

    plt.plot(y_test)
    plt.plot(forecast_mean)
    plt.plot(forecast_l95)
    plt.plot(forecast_u95)
    plt.title('Bayesian UC: Forecast')
    plt.legend(['Actual', 'Mean', 'LB', 'UB'])
    plt.show()

    # Print RMSE
    print(f"BAYES-UC RMSE: {rmse(y_test, forecast_mean)}")
