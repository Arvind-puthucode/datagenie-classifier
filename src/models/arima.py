import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV
from statsmodels.tsa.arima.model import ARIMA
import itertools
class arimaModel:
    def __init__(self, df: pd.DataFrame):
        df.drop(columns=[df.columns[0]],inplace=True)
        df.index = pd.to_datetime(df.index)
        self.data = df
        l = len(self.data)
        X = self.data.index
        y = self.data[df.columns[0]]
        limit = int(l * 0.7)
        self.X_train, self.X_test, self.y_train, self.y_test = X[0:limit], X[limit:], y[0:limit], y[limit:]

    def create_model(self):
        order = self.hyperparameter_optimization()
        model = ARIMA(self.y_train, order=order)
        model_fit = model.fit()
        y_pred = model_fit.forecast(steps=len(self.X_test))[0]
        print(f'the model best fit order was {order}')
        return self.mape(y_pred)

    def mape(self, y_pred):
        abs_diffs = np.abs(self.y_test - y_pred)
        pct_diffs = abs_diffs / self.y_test
        pct_diffs[np.isnan(pct_diffs)] = 0
        MAPE_error = np.mean(pct_diffs) * 100
        return MAPE_error

    def hyperparameter_optimization(self):

        # Define ranges for p, d, and q
        p_range = range(0, 6)  # for example
        d_range = range(0, 3)  # for example
        q_range = range(0, 6)  # for example

        best_aic = float("inf")
        best_params = None
        ci=0
        # Iterate through all possible combinations of p, d, and q
        for p, d, q in itertools.product(p_range, d_range, q_range):
            try:
                model = ARIMA(self.y_train, order=(p, d, q))
                results = model.fit()
                aic = results.aic

                # Update the best parameters if this model has a lower AIC
                if aic < best_aic:
                    best_aic = aic
                    best_params = (p, d, q)
                ci+=1
            except:
                print('Exception error')
                continue
        print(f'try runned are:{ci} , missed are{36*3-ci}')
        return(best_params)




if __name__ == "__main__":
    eg_data = pd.read_csv("data/daily/sample_1.csv",index_col="point_timestamp")
    print(eg_data.head(),eg_data.columns[0])
    arima_model = arimaModel(eg_data)
    print(f'ARIMA MAPE error is {arima_model.create_model()}')
