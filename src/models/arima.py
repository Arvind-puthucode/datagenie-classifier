import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
import itertools
from joblib import Parallel, delayed

class arimaModel:
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
        model = ARIMA(self.y_train, order=best_params)
        model_fit = model.fit()
        y_pred = model_fit.forecast(steps=len(self.X_test))[0]
        print(f'The ARIMA model best fit order was {best_params}')
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

        # Generate all possible combinations of p, d, and q
        all_params = list(itertools.product(p_range, d_range, q_range))

        # Use parallel processing to find the best parameters
        results = Parallel(n_jobs=-1)(
            delayed(self._calculate_aic)(params) for params in all_params
        )

        best_aic = float("inf")
        best_params = None

        for aic, params in results:
            if aic < best_aic:
                best_aic = aic
                best_params = params

        return best_params

    def _calculate_aic(self, params):
        p, d, q = params
        try:
            model = ARIMA(self.y_train, order=(p, d, q))
            results = model.fit()
            aic = results.aic
            return aic, (p, d, q)
        except:
            return float('inf'), (p, d, q)

if __name__ == "__main__":
    eg_data = pd.read_csv("data/daily/sample_1.csv", index_col="point_timestamp")
    arima_model = arimaModel(eg_data)
    best_params = arima_model.create_model()
    print(f"The best ARIMA model order is {best_params}")
