import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.stattools import jarque_bera
import statsmodels.api as sm
import pandas as pd
import pywt
from scipy.stats import skew, kurtosis

class Parameters:
    def __init__(self, data_set: pd.DataFrame):
        self.data = data_set

    def measure_trend(self, timeseries):
        # Fit a linear regression model to the time series
        x = np.arange(len(timeseries)).reshape(-1, 1)
        model = LinearRegression()
        model.fit(x, timeseries)
        return model.coef_[0]
    def measure_seasonality(self, timeseries):
        # Calculate the power spectrum
        fft_values = np.fft.fft(timeseries)
        power_spectrum = np.abs(fft_values) ** 2

        # Identify the dominant frequency
        max_freq_index = np.argmax(power_spectrum[1:len(timeseries) // 2]) + 1
        dominant_freq = max_freq_index / len(timeseries)

        # Calculate the ratio of power at the dominant frequency to total power
        power_ratio = np.max(power_spectrum) / np.sum(power_spectrum)
        return power_ratio
    def measure_autocorrelation(self, timeseries):
        # Calculate autocorrelation
        acf_values = acf(timeseries, nlags=10, fft=False)
        max_autocorr = np.max(np.abs(acf_values))
        return max_autocorr

    def measure_stationarity(self, timeseries):
        # Perform ADF test for stationarity
        result = adfuller(timeseries)
        return result[1] <= 0.05  # True if stationary, False if not

    def measure_heteroscedasticity(self, timeseries):
        # Perform ARCH test for heteroscedasticity
        _, p_value, _, _ = het_arch(timeseries)
        return p_value <= 0.05  # True if heteroscedastic, False if not

    def measure_residual_patterns(self, timeseries):
    # Fit an ARIMA model to detect residual patterns
    # Using ARIMA(1, 0, 0) as an example order, you can adjust the order as needed
        model = ARIMA(timeseries, order=(1, 0, 0))
        model_fit = model.fit()
        residuals = model_fit.resid
        _, jb_p_value, _, _ = jarque_bera(residuals)
        return jb_p_value < 0.05  # True if patterns in residuals (reject else)

    def measure_outliers(self, timeseries,threshold=3):
         # Calculate mean and standard deviation
        mean_val = np.mean(timeseries)
        std_dev = np.std(timeseries)

        # Calculate Z-scores for each data point
        z_scores = np.abs((timeseries - mean_val) / std_dev)

        # Identify outliers based on the threshold
        outliers = z_scores > threshold

        # Count the number of outliers
        num_outliers = np.sum(outliers)

        return num_outliers

    def measure_non_linearity(self, timeseries):
        # Calculate skewness and kurtosis as indicators of non-linearity
        skewness = skew(timeseries)
        kurt = kurtosis(timeseries)
        return 'Non-Linear' if abs(skewness) > 1 or abs(kurt) > 1 else 'Linear'

    def measure_periodicity(self, timeseries, threshold=0.2):
        # Calculate autocorrelation
        acf_values = acf(timeseries, nlags=10, fft=False)

        # Check if any autocorrelation exceeds the threshold
        has_periodicity = np.any(acf_values > threshold)

        return 'Yes' if has_periodicity else 'No'

    def measure_magnitude_of_fluctuations(self, timeseries):
        # Calculate standard deviation as a measure of fluctuations
        return np.std(timeseries)

    def measure_time_series_length(self, timeseries):
        # Calculate the length of the time series
        return len(timeseries)

    def get_params(self, timeseries):
        return {
            'Trend': self.measure_trend(timeseries),
            'Seasonality': self.measure_seasonality(timeseries),
            'Autocorrelation': self.measure_autocorrelation(timeseries),
            'Stationarity': self.measure_stationarity(timeseries),
            'Heteroscedasticity': self.measure_heteroscedasticity(timeseries),
            'Residual Patterns': self.measure_residual_patterns(timeseries),
            'Outliers': self.measure_outliers(timeseries),
            'Non-Linearity': self.measure_non_linearity(timeseries),
            'Periodicity': self.measure_periodicity(timeseries),
            'Magnitude of Fluctuations': self.measure_magnitude_of_fluctuations(timeseries),
            'Time Series Length': self.measure_time_series_length(timeseries)
        }

if __name__ == "__main__":
    eg_df = pd.read_csv("data/daily/sample_1.csv", index_col=0)
    eg_df = eg_df.set_index(eg_df.columns[0])
    eg_df = eg_df.fillna(eg_df.mean())
    eg1 = Parameters(eg_df)
    timeseries = (eg1.data)[eg_df.columns[0]]
    parameters = eg1.get_params(timeseries)
    print(parameters, "parameters")
