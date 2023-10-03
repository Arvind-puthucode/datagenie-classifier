import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping

class lstmModel:
    def __init__(self, df: pd.DataFrame):
        self.datearr=df.loc[:,'point_timestamp']
        df.drop(columns=[df.columns[0]], inplace=True)
        df.index = pd.to_datetime(df.index)
        self.data = df
        l = len(self.data)
        X = self.data.index
        y = self.data[df.columns[0]]
        limit = int(l * 0.7)
        self.X_train, self.X_test, self.y_train, self.y_test = X[0:limit], X[limit:], y[0:limit], y[limit:]

    
    def create_model(self):
        # Prepare the data
        X_train, y_train = self.prepare_data(self.X_train, self.y_train)
        X_test, y_test = self.prepare_data(self.X_test, self.y_test)

        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(units=100, activation='relu', input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))  # Dropout for regularization
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Early stopping to prevent overfitting
        early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

        # Train the model
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop])

        # Predict using the trained model
        y_pred = model.predict(X_test)
        y_pred = self.inverse_transform(y_pred)

        # Calculate MAPE
        mape = self.mape(y_pred)

        return mape

    def result_json(self):
        # Prepare the data
        self.X_train, self.y_train = self.prepare_data(self.X_train, self.y_train)
        self.X_test, self.y_test = self.prepare_data(self.X_test, self.y_test)

        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(units=100, activation='relu', input_shape=(self.X_train.shape[1], 1)))
        model.add(Dropout(0.2))  # Dropout for regularization
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Early stopping to prevent overfitting
        early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

        # Train the model
        model.fit(self.X_train, self.y_train, epochs=50, batch_size=32, validation_data=(self.X_test, self.y_test), callbacks=[early_stop])

        # Predict using the trained model
        y_pred = model.predict(self.X_test)
        y_pred = self.inverse_transform(y_pred)

        # Predict using the trained model
        y_pred_train = model.predict(self.X_train)
        y_pred_train = self.inverse_transform(y_pred_train)
        
        # Calculate MAPE
        mape = self.mape(y_pred)
        l1,l2=y_pred_train.tolist(),y_pred.tolist()
        l1.extend(l2)
        l3,l4=self.y_train.tolist(),self.y_test.tolist()
        l3.extend(l4)
        print(len(l1),len(l3),len(l2),len(l4))
        return {"mape":mape,"point_timestamp":self.datearr.tolist()
                ,"y_pred":l1,
                "y_test":l3}
    
    
    def prepare_data(self, X, y, sequence_length=10):
        # Normalize the data
        scaler = MinMaxScaler()
        y = np.array(y).reshape(-1, 1)
        y = scaler.fit_transform(y)

        # Create sequences for LSTM
        X_seq, y_seq = [], []
        for i in range(len(X) - sequence_length):
            X_seq.append(y[i:i+sequence_length, 0])
            y_seq.append(y[i+sequence_length, 0])

        X_seq, y_seq = np.array(X_seq), np.array(y_seq)

        # Reshape for LSTM
        X_seq = np.reshape(X_seq, (X_seq.shape[0], X_seq.shape[1], 1))

        return X_seq, y_seq

    def inverse_transform(self, y):
        # Inverse transform the normalized values
        scaler = MinMaxScaler()
        scaler.fit_transform(self.y_test.reshape(-1, 1))
        y_inverse = scaler.inverse_transform(y)
        return y_inverse

    def mape(self, y_pred):
        self.y_test_no_zeros = np.where(self.y_test == 0, 1e-10, self.y_test)

        abs_diffs = np.abs(self.y_test_no_zeros - y_pred)
        pct_diffs = abs_diffs / self.y_test_no_zeros
        pct_diffs[np.isnan(pct_diffs)] = 0
        MAPE_error = np.mean(pct_diffs) * 100
        return MAPE_error


if __name__ == "__main__":
    eg_data = pd.read_csv("data/daily/sample_1.csv", index_col="point_timestamp")
    lstm_model = lstmModel(eg_data)
    print(f'LSTM MAPE error is {lstm_model.create_model()}')
