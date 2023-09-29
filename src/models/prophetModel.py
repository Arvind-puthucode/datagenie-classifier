import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from prophet import Prophet

class prophetModel:
    def __init__(self, df: pd.DataFrame):
        df.drop(columns=[df.columns[0]], inplace=True)
        df.reset_index(inplace=True)
        df.columns = ['ds', 'y']
        self.data = df
        l = len(self.data)
        limit = int(l * 0.7)
        self.train_data = self.data.iloc[:limit]
        self.test_data = self.data.iloc[limit:]

    def create_model(self):
        model = Prophet()
        model.fit(self.train_data)
        future = model.make_future_dataframe(periods=len(self.test_data))
        forecast = model.predict(future)
        y_pred = forecast['yhat'].tail(len(self.test_data))
        MAPE_error = self.mape(self.test_data['y'], y_pred)
        return MAPE_error

    def mape(self, y_true, y_pred):
        return mean_absolute_percentage_error(y_true, y_pred) * 100

if __name__ == "__main__":
    eg_data = pd.read_csv("data/daily/sample_1.csv", index_col="point_timestamp")
   # print(eg_data.head(), eg_data.columns[0])

    # Ensure the 'point_timestamp' is in datetime format
    eg_data.index = pd.to_datetime(eg_data.index)

    prophet_model = prophetModel(eg_data)
    mape_error= prophet_model.create_model()
    print(f'Prophet MAPE error is {mape_error}')
    