import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from prophet import Prophet

class prophetModel:
    def __init__(self, df: pd.DataFrame):
        df.columns = ['ds', 'y']
        self.data = df
        l = len(self.data)
        limit = int(l * 0.7)
        self.train_data = self.data.iloc[:limit]
        self.test_data = self.data.iloc[limit:]
        print(self.train_data.head(2))

    def create_model(self):
        model = Prophet()
        model.fit(self.train_data)
        future = model.make_future_dataframe(periods=len(self.test_data))
        forecast = model.predict(future)
        y_pred = forecast['yhat'].tail(len(self.test_data))
        mape_error = self.mape(self.test_data['y'], y_pred)
        return mape_error
    def result_json(self):
        model = Prophet()
        model.fit(self.train_data)
                # Make predictions for the training data
        future_train = model.make_future_dataframe(periods=len(self.train_data))
        forecast_train = model.predict(future_train)
        y_pred_train = forecast_train['yhat'].tail(len(self.train_data))

        # Calculate MAPE for the training data

        # Assuming self.test_data contains your test data

        # Make predictions for the test data
        future_test = model.make_future_dataframe(periods=len(self.test_data))
        forecast_test = model.predict(future_test)
        y_pred_test = forecast_test['yhat'].tail(len(self.test_data))

        l1,l2=y_pred_train.tolist(),y_pred_test.tolist()
        l1.extend(l2)
        l3,l4=self.train_data['y'].values.tolist(),self.test_data['y'].values.tolist()
        l3.extend(l4)
        print(len(l1),len(l3),len(l2),len(l4))
        mape_error_test = self.mape(self.test_data['y'], y_pred_test)

        return {"mape":mape_error_test,"point_timestamp":self.data['ds'].values.tolist()
                ,"y_pred":l1,
                "y_test":l3}
    
        # Calculate MAPE for the test data

    def mape(self, y_true, y_pred):
        return mean_absolute_percentage_error(y_true, y_pred) * 100

if __name__ == "__main__":
    eg_data = pd.read_csv("data/daily/sample_1.csv", index_col=0)
   # print(eg_data.head(), eg_data.columns[0])

    # Ensure the 'point_timestamp' is in datetime format
    #eg_data.index = pd.to_datetime(eg_data.index)

    prophet_model = prophetModel(eg_data)
    mape_error= prophet_model.create_model()
    print(f'Prophet MAPE error is {mape_error}')
    