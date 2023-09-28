import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV

class varModel:
    def __init__(self, df: pd.DataFrame):
        # Assuming df has the appropriate structure and columns
        df.drop(columns=[df.columns[0]], inplace=True)
        df.reset_index(inplace=True)
        df.columns = ['ds', 'y']
        self.data = df
        l = len(self.data)
        limit = int(l * 0.7)
        self.train_data = self.data.iloc[:limit]
        self.test_data = self.data.iloc[limit:]

    def create_model(self, lag_order):
        model = VAR(self.train_data)
        model_fitted = model.fit(lag_order)
        y_pred = model_fitted.forecast(y=self.test_data, steps=12)
        return self.mape(self.test_data.values, y_pred)

    @staticmethod
    def mape(y_true, y_pred):
        return mean_absolute_percentage_error(y_true, y_pred)

    def hyperparameter_optimization(self):
        # Define the range of lag orders (p values) to search
        p_values = range(1, 7)  # Adjust the range based on your needs

        # Dictionary to store results
        results = {}

        # Iterate through the lag orders and fit VAR models
        for p in p_values:
            mape_error = self.create_model(p)
            results[p] = mape_error

        # Find the lag order with the lowest MAPE
        best_order = min(results, key=results.get)
        print(f"Best lag order (p) found: {best_order}")

        return best_order


if __name__ == "__main__":
    eg_data = pd.read_csv("data/daily/sample_1.csv", index_col="point_timestamp")
    var_model = varModel(eg_data)
    best_order = var_model.hyperparameter_optimization()
    mape_error = var_model.create_model(best_order)
    print(f'MAPE error for the best lag order ({best_order}): {mape_error}')
