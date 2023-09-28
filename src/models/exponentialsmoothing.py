import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import itertools

class exponentialsmoothingModel:
    def __init__(self, df: pd.DataFrame):
        df.drop(columns=[df.columns[0]], inplace=True)
        df.index = pd.to_datetime(df.index)
        self.data = df
        l = len(self.data)
        X = self.data.index
        y = self.data[df.columns[0]]
        limit = int(l * 0.7)
        self.X_train, self.X_test, self.y_train, self.y_test = X[0:limit], X[limit:], y[0:limit], y[limit:]

    def create_model(self):
        best_params = self.hyperparameter_optimization()
        model = ExponentialSmoothing(self.y_train, trend=best_params[0], seasonal=best_params[1],
                                     seasonal_periods=best_params[2])
        model_fit = model.fit()
        y_pred = model_fit.forecast(steps=len(self.X_test))
        print(f'The best fit parameters were {best_params}')
        return self.mape(y_pred)

    def mape(self, y_pred):
        abs_diffs = np.abs(self.y_test - y_pred)
        pct_diffs = abs_diffs / self.y_test
        pct_diffs[np.isnan(pct_diffs)] = 0
        MAPE_error = np.mean(pct_diffs) * 100
        return MAPE_error

    def hyperparameter_optimization(self):
        trend_options = ['add', 'mul', None]
        seasonal_options = ['add', 'mul', None]
        seasonal_periods_options = [None, 12]  # Adjust as needed

        best_mape = float("inf")
        best_params = None
        ci = 0

        for trend, seasonal, seasonal_periods in itertools.product(trend_options, seasonal_options,
                                                                   seasonal_periods_options):
            try:
                model = ExponentialSmoothing(self.y_train, trend=trend, seasonal=seasonal,
                                             seasonal_periods=seasonal_periods)
                model_fit = model.fit()
                y_pred = model_fit.forecast(steps=len(self.X_test))
                mape = self.mape(y_pred)

                if mape < best_mape:
                    best_mape = mape
                    best_params = (trend, seasonal, seasonal_periods)
                ci += 1
            except:
                print('Exception error')
                continue

        print(f'Total iterations: {ci}')
        return best_params


if __name__ == "__main__":
    eg_data = pd.read_csv("data/daily/sample_1.csv", index_col="point_timestamp")
    print(eg_data.head(), eg_data.columns[0])
    exp_smoothing_model = exponentialsmoothingModel(eg_data)
    print(f'Exponential Smoothing MAPE error is {exp_smoothing_model.create_model()}')
