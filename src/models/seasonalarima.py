import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools

class seasonalarimaModel:
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
        model = SARIMAX(self.y_train, order=best_params[:3], seasonal_order=best_params[3:])
        model_fit = model.fit(disp=False)
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

        p_range = range(0, 3)  
        d_range = range(0, 3)  
        q_range = range(0, 3)  
        P_range = range(0, 6)  
        D_range = range(0, 3)  
        Q_range = range(0, 3)  
        seasonal_period = 12  # can be changed

        best_aic = float("inf")
        best_params = None
        ci = 0

        for p, d, q, P, D, Q in itertools.product(p_range, d_range, q_range, P_range, D_range, Q_range):
            try:
                seasonal_order = (P, D, Q, seasonal_period)
                order = (p, d, q)
                model = SARIMAX(self.y_train, order=order, seasonal_order=seasonal_order)
                results = model.fit(disp=False)
                aic = results.aic

                if aic < best_aic:
                    best_aic = aic
                    best_params = (p, d, q, P, D, Q)
                ci += 1
            except:
                print('Exception error')
                continue

        print(f'Total iterations: {ci}')
        return best_params


if __name__ == "__main__":
    eg_data = pd.read_csv("data/daily/sample_1.csv", index_col="point_timestamp")
    print(eg_data.head(), eg_data.columns[0])
    sarima_model = seasonalarimaModel(eg_data)
    print(f'Seasonal ARIMA MAPE error is {sarima_model.create_model()}')
