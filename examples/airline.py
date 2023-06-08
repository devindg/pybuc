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

# Import airline passenger data
url = "https://raw.githubusercontent.com/devindg/pybuc/master/examples/data/airline-passengers.csv"
air = pd.read_csv(url, header=0, index_col=0)
air = air.astype(float)
air.index = pd.to_datetime(air.index)
hold_out_size = 12

# Create train and test sets
y_train = air.iloc[:-hold_out_size]
y_test = air.iloc[-hold_out_size:]

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
    plt.xticks(rotation=45, ha="right")
    plt.show()

    # Get and plot forecast
    sarima_forecast = sarima_res.get_forecast(hold_out_size).summary_frame(alpha=0.05)
    plt.plot(y_test)
    plt.plot(sarima_forecast['mean'])
    plt.fill_between(sarima_forecast.index,
                     sarima_forecast['mean_ci_lower'],
                     sarima_forecast['mean_ci_upper'], alpha=0.2)
    plt.title('SARIMA: Forecast')
    plt.legend(['Actual', 'Mean', '95% Prediction Interval'])
    plt.show()

    # Print RMSE
    print(f"SARIMA RMSE: {rmse(y_test.to_numpy(), sarima_forecast['mean'].to_numpy())}")

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
    plt.fill_between(mle_uc_forecast.index,
                     mle_uc_forecast['mean_ci_lower'],
                     mle_uc_forecast['mean_ci_upper'], alpha=0.2)
    plt.title('MLE UC: Forecast')
    plt.legend(['Actual', 'Mean', '95% Prediction Interval'])
    plt.show()

    # Print RMSE
    print(f"MLE UC RMSE: {rmse(y_test.to_numpy(), mle_uc_forecast['mean'].to_numpy())}")

    ''' Fit the airline data using Bayesian unobserved components '''
    bayes_uc = buc.BayesianUnobservedComponents(response=y_train,
                                                level=True, stochastic_level=True,
                                                trend=True, stochastic_trend=True,
                                                trig_seasonal=((12, 0),), stochastic_trig_seasonal=(True,),
                                                seed=123)
    post = bayes_uc.sample(5000)
    mcmc_burn = 100

    # Print summary of estimated parameters
    for key, value in bayes_uc.summary(burn=mcmc_burn).items():
        print(key, ' : ', value)

    # Plot in-sample fit against actuals
    yhat = np.mean(post.filtered_prediction[mcmc_burn:], axis=0)
    plt.plot(y_train)
    plt.plot(y_train.index, yhat)
    plt.title('Bayesian-UC: In-sample')
    plt.show()

    # Plot time series components
    bayes_uc.plot_components(burn=mcmc_burn, smoothed=True)
    plt.show()

    # Get and plot forecast
    forecast, _ = bayes_uc.forecast(hold_out_size, mcmc_burn)
    forecast_mean = np.mean(forecast, axis=0)
    forecast_l95 = np.quantile(forecast, 0.025, axis=0).flatten()
    forecast_u95 = np.quantile(forecast, 0.975, axis=0).flatten()

    plt.plot(y_test)
    plt.plot(bayes_uc.future_time_index, forecast_mean)
    plt.fill_between(bayes_uc.future_time_index, forecast_l95, forecast_u95, alpha=0.2)
    plt.title('Bayesian UC: Forecast')
    plt.legend(['Actual', 'Mean', '95% Prediction Interval'])
    plt.show()

    # Print RMSE
    print(f"BAYES-UC RMSE: {rmse(y_test.to_numpy(), forecast_mean)}")
