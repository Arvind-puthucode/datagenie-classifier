import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.stattools import jarque_bera
import statsmodels.api as sm
from scipy.stats import skew, kurtosis
import pandas as pd

class ParametersManual:
    def __init__(self, data_set):
        self.df = data_set

    def measure_autocorrelation(self, timeseries):
        # Calculate autocorrelation
        acf_values = acf(timeseries, nlags=10, fft=False)
        max_autocorr = np.max(np.abs(acf_values))
        return max_autocorr

    def measure_trend(self, timeseries):
        # Measure trend based on the rate of change
        avg_diff = np.mean(np.diff(timeseries))
        if avg_diff > 0:
            return 3  # Positive trend
        elif avg_diff < 0:
            return 1  # Negative trend
        else:
            return 2  # No trend

    def measure_seasonality(self, timeseries):
        # Measure seasonality based on Fourier Transform
        fft_values = np.fft.fft(timeseries)
        power_spectrum = np.abs(fft_values) ** 2
        max_freq_index = np.argmax(power_spectrum[1:len(timeseries) // 2]) + 1
        dominant_freq = max_freq_index / len(timeseries)
        if dominant_freq > 0:
            return 3  # Seasonal pattern
        else:
            return 1  # No clear seasonality

    def measure_noise_level(self, timeseries):
        # Measure noise level based on standard deviation
        std_dev = np.std(timeseries)
        if std_dev < 5:
            return 1  # Low noise
        elif std_dev < 20:
            return 2  # Moderate noise
        else:
            return 3  # High noise

    def measure_temporal_dependencies(self, timeseries):
        # Measure temporal dependencies based on autocorrelation
        acf_values = acf(timeseries, nlags=10, fft=False)
        max_autocorr = np.max(np.abs(acf_values))
        if max_autocorr > 0.5:
            return 3  # Strong temporal dependencies
        elif max_autocorr > 0.2:
            return 2  # Moderate temporal dependencies
        else:
            return 1  # Weak temporal dependencies

    def measure_data_stationarity(self, timeseries):
        # Measure stationarity using Augmented Dickey-Fuller test
        result = adfuller(timeseries)
        if result[1] <= 0.05:
            return 1  # Stationary
        else:
            return 2  # Non-stationary

    def measure_model_complexity(self, timeseries):
        # Measure model complexity based on the length of the time series
        if len(timeseries) > 100:
            return 3  # Complex model
        elif len(timeseries) > 50:
            return 2  # Moderate complexity
        else:
            return 1  # Simple model

    def measure_data_size_and_availability(self, timeseries):
        # Measure data size and availability based on the length of the time series
        if len(timeseries) > 500:
            return 3  # Sufficient data, suitable for large models
        elif len(timeseries) > 200:
            return 2  # Moderate data, suitable for medium-sized models
        else:
            return 1  # Limited data, suitable for simpler models

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

    def measure_outliers(self, timeseries, threshold=3):
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

    def measure_historical_behavior(self, timeseries):
        # Calculate the average rate of change
        avg_change = np.mean(np.diff(timeseries))

        # Categorize historical behavior based on the average rate of change
        if avg_change > 0.5:
            return 3  # Strongly increasing historical behavior
        elif avg_change < -0.5:
            return 1  # Strongly decreasing historical behavior
        else:
            return 2  # Relatively stable historical behavior

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

    def get_params(self):
        self.df = self.df.set_index(self.df.columns[0])
        timeseries = self.df[self.df.columns[0]]
        
        existing_params = {
            'Trend': self.measure_trend(timeseries),
            'Seasonality': self.measure_seasonality(timeseries),
            'Autocorrelation': self.measure_autocorrelation(timeseries),
            'Stationarity': self.measure_data_stationarity(timeseries),
            'Heteroscedasticity': self.measure_heteroscedasticity(timeseries),
            'Residual Patterns': self.measure_residual_patterns(timeseries),
            'Outliers': self.measure_outliers(timeseries),
            'Non-Linearity': self.measure_non_linearity(timeseries),
            'Periodicity': self.measure_periodicity(timeseries),
            'Magnitude of Fluctuations': self.measure_magnitude_of_fluctuations(timeseries),
            'Time Series Length': self.measure_time_series_length(timeseries),
            'Historical Behavior': self.measure_historical_behavior(timeseries),
            'Model Complexity': self.measure_model_complexity(timeseries),
            'Data Size and Availability': self.measure_data_size_and_availability(timeseries)
            # Add more parameters here...
        }
        
        return existing_params

if __name__ == "__main__":
    eg_df = pd.read_csv("data/daily/sample_1.csv", index_col=0)
    eg1_manual = ParametersManual(eg_df)
    parameters_manual = eg1_manual.get_params()
    print(parameters_manual, "manual parameters")
