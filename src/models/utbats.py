import itertools
from tbats import TBATS
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import ParameterGrid
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class utbatsModel:
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
        # Fit the TBATS model
        tbats_model = TBATS(seasonal_periods=[7, 30.4])  # Daily and monthly seasonality
        tbats_fit = tbats_model.fit(self.y_train)

        # Forecast using the trained model
        y_pred = tbats_fit.forecast(steps=len(self.X_test))

        # Calculate MAPE
        mape = self.mape(y_pred)

        return mape
    def parallel_hyperparameter_optimization(self):
        # Define a grid of hyperparameters to search
        param_grid = {
            'seasonal_periods': [[7, 30.4], [7]],
            'use_box_cox': [True, False],
            'use_trend': [True, False]
        }

        # Generate all combinations of hyperparameters
        param_combinations = list(ParameterGrid(param_grid))

        with ThreadPoolExecutor() as executor:
            futures = []
            for params in param_combinations:
                futures.append(executor.submit(self.fit_tbats, params))

            best_mape = float('inf')
            best_params = None

            for future in futures:
                params, mape = future.result()
                if mape < best_mape:
                    best_mape = mape
                    best_params = params

        return best_params, best_mape

    def fit_tbats(self, params):
        tbats_model = TBATS(seasonal_periods=params['seasonal_periods'], use_box_cox=params['use_box_cox'],
                            use_trend=params['use_trend'])
        tbats_fit = tbats_model.fit(self.y_train)
        y_pred = tbats_fit.forecast(steps=len(self.X_test))
        mape = self.mape(y_pred)

        return params, mape

    
    def mape(self, y_pred):
        abs_diffs = np.abs(self.y_test - y_pred)
        pct_diffs = abs_diffs / self.y_test
        pct_diffs[np.isnan(pct_diffs)] = 0
        MAPE_error = np.mean(pct_diffs) * 100
        return MAPE_error

if __name__ == "__main__":
    eg_data = pd.read_csv("data/daily/sample_1.csv", index_col="point_timestamp")
    tbats_model = utbatsModel(eg_data)
    best_params, best_mape = tbats_model.parallel_hyperparameter_optimization()
    print(f'Best hyperparameters: {best_params}')
    print(f'Best MAPE error: {best_mape}')
