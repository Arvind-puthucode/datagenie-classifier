import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping

class LSTMModel:
    def __init__(self, df: pd.DataFrame):
        self.datearr = df.loc[:, 'point_timestamp']
        df.drop(columns=[df.columns[0]], inplace=True)
        df.index = pd.to_datetime(df.index)
        self.data = df
        l = len(self.data)
        X = self.data.index
        y = self.data[df.columns[0]]
        limit = int(l * 0.7)
        self.x_train, self.x_test, self.y_train, self.y_test = X[0:limit], X[limit:], y[0:limit], y[limit:]

    def create_model(self):
        # Prepare the data
        self.x_train, self.y_train = self.prepare_data(self.x_train, self.y_train)
        self.x_test, self.y_test = self.prepare_data(self.x_test, self.y_test)

        # Build and train the LSTM model
        model, y_pred, y_pred_train = self.train_lstm_model(self.x_train, self.y_train, self.x_test)

        # Calculate MAPE
        mape = self.mape(y_pred)

        l1, l2 = y_pred_train.tolist(), y_pred.tolist()
        l1.extend(l2)
        l3, l4 = self.y_train.tolist(), self.y_test.tolist()
        l3.extend(l4)

        print(len(l1), len(l3), len(l2), len(l4))
        return {"mape": mape, "point_timestamp": self.datearr.tolist(), "y_pred": l1, "y_test": l3}

    def train_lstm_model(self, x_train, y_train, x_test):
        model = Sequential()
        model.add(LSTM(units=100, activation='relu', input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
        model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, self.y_test), callbacks=[early_stop])

        y_pred = model.predict(x_test)
        y_pred = self.inverse_transform(y_pred)

        y_pred_train = model.predict(x_train)
        y_pred_train = self.inverse_transform(y_pred_train)

        return model, y_pred, y_pred_train

    def prepare_data(self, X, y, sequence_length=10):
        scaler = MinMaxScaler()
        y = np.array(y).reshape(-1, 1)
        y = scaler.fit_transform(y)

        x_seq, y_seq = [], []
        for i in range(len(X) - sequence_length):
            x_seq.append(y[i:i + sequence_length, 0])
            y_seq.append(y[i + sequence_length, 0])

        x_seq, y_seq = np.array(x_seq), np.array(y_seq)
        x_seq = np.reshape(x_seq, (x_seq.shape[0], x_seq.shape[1], 1))

        return x_seq, y_seq

    def inverse_transform(self, y):
        scaler = MinMaxScaler()
        scaler.fit_transform(self.y_test.reshape(-1, 1))
        y_inverse = scaler.inverse_transform(y)
        return y_inverse

    def mape(self, y_pred):
        self.y_test_no_zeros = np.where(self.y_test == 0, 1e-10, self.y_test)
        abs_diffs = np.abs(self.y_test_no_zeros - y_pred)
        pct_diffs = abs_diffs / self.y_test_no_zeros
        pct_diffs[np.isnan(pct_diffs)] = 0
        mape_error = np.mean(pct_diffs) * 100
        return mape_error

if __name__ == "__main__":
    eg_data = pd.read_csv("data/daily/sample_1.csv", index_col="point_timestamp")
    lstm_model = LSTMModel(eg_data)
    print(f'LSTM MAPE error is {lstm_model.create_model()}')
