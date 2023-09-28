import pandas as pd
import numpy as np
from arch import arch_model
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV

class garchModel:
    def __init__(self, df: pd.DataFrame):
        df.drop(columns=[df.columns[0]], inplace=True)
        df.index = pd.to_datetime(df.index)
        self.data = df
        l = len(self.data)
        X = self.data.index
        y = self.data[df.columns[0]]
        limit = int(l * 0.7)
        self.X_train, self.X_test, self.y_train, self.y_test = X[:limit], X[limit:], y[:limit], y[limit:]

    def create_model(self, p, q):
        model = arch_model(self.y_train, vol='Garch', p=p, q=q)
        model_fit = model.fit(disp='off')
        forecasts = model_fit.forecast(start=len(self.y_train),horizon=len(self.y_test))
        print(forecasts,model_fit)
        if forecasts.variance.shape[0] == 0:
            # Handle the case where forecasts are empty
            return float('inf')

        y_pred = np.sqrt(forecasts.variance.values[:, 0])
        return self.mape(self.y_test, y_pred)

    def mape(self, y_true, y_pred):
        return mean_absolute_percentage_error(y_true, y_pred)

    def hyperparameter_optimization(self):
        # Define the range of GARCH(p, q) parameters to search
        p_values = range(1, 6)
        q_values = range(1, 6)

        best_mape = float("inf")
        best_params = None

        for p in p_values:
            for q in q_values:
                mape_error = self.create_model(p, q)
                if mape_error < best_mape:
                    best_mape = mape_error
                    best_params = (p, q)

        print(f"Best GARCH(p, q) parameters found: {best_params}")

        return best_params


if __name__ == "__main__":
    eg_data = pd.read_csv("data/daily/sample_1.csv", index_col="point_timestamp")
    garch_model = garchModel(eg_data)
    best_params = garch_model.hyperparameter_optimization()
    if best_params:
        mape_error = garch_model.create_model(*best_params)
        print(f'MAPE error for the best GARCH(p, q) parameters {best_params}: {mape_error}')
    else:
        print("No valid GARCH(p, q) parameters found.")
