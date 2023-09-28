import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pywt
from scipy.stats import entropy

from statsmodels.tsa.stattools import adfuller, acf
import statsmodels.api as sm


class Parameters:
    def __init__(self, data_set: pd.DataFrame):
        self.data = data_set
    # statistical measures

    def get_cv(self, timeseries):
        """Function to calculate the coefficient of variation (CV) of a given time series"""
        mean = np.mean(timeseries)
        std = np.std(timeseries)
        cv = std / mean
        return cv

    # Define a function to test for stationarity using the ADF test
    def test_stationarity(self, timeseries):
        # Perform ADF test
        result = adfuller(timeseries)
        adf_stat = result[0]
        # use critical value for 1% significance level
        crit_val = result[4]['1%']
        norm_adf_stat = np.abs(adf_stat / crit_val)
        return norm_adf_stat

    # measure for trend ie slope as a indicator

    def measure_trend(self, timeseries):
        # Fit linear regression to the time series
        cycle, trend = sm.tsa.filters.hpfilter(timeseries)
        # Calculate total variance
        total_var = timeseries.var()
        # Calculate variance of trend component
        trend_var = trend.var()
        # Calculate trendness
        trendness = trend_var / total_var
        return trendness
    """auto correlation features
    """

    def auto_corr_measure(self, timeseries):
        # calculate the ACF
        acf_values, confint = acf(timeseries, nlags=10, alpha=0.05, fft=False)
        max_autocorr = np.max(np.abs(acf_values))
        return max_autocorr

    def seasonality_measure(self, timeseries):
        # calculate the power spectrum
        fft_values = np.fft.fft(timeseries)
        power_spectrum = np.abs(fft_values) ** 2

        # identify the dominant frequency
        max_freq_index = np.argmax(power_spectrum[1:len(timeseries)//2]) + 1
        dominant_freq = max_freq_index / len(timeseries)

        # calculate the ratio of power at the dominant frequency to total power
        power_ratio = np.max(power_spectrum) / np.sum(power_spectrum)
        return power_ratio

    def wavelet_rms(self, timeseries):
        # perform wavelet transform
        coeffs = pywt.wavedec(timeseries, 'db1')
        # calculate RMS of coefficients
        rms = 0
        for i in range(len(coeffs)):
            rms += (sum(j*j for j in coeffs[i]))/len(coeffs[i])
        rms = rms**(1/2)
        # Calculate maximum possible RMS value
        max_val = np.sqrt(len(timeseries) * np.max(timeseries)**2)
        # Normalize the RMS value
        norm_rms = rms / max_val
        return norm_rms
    
    def measure_entropy(self, timeseries):
        # Calculate the histogram of the time series
        hist, _ = np.histogram(timeseries, bins='auto', density=True)

        # Calculate the Shannon entropy of the histogram
        entropy_val = entropy(hist)

        return entropy_val
    def get_params(self,timeseries):
        return {'p1': self.get_cv(timeseries),
                 'p2' : self.test_stationarity(timeseries),
                    "p3": self.measure_trend(timeseries),
                    "p4": self.auto_corr_measure(timeseries),
                    "p5": self.seasonality_measure(timeseries),
                    "p6": self.wavelet_rms(timeseries),
                    "p7": self.measure_entropy(timeseries),
                }

if __name__ == "__main__":
    eg_df = pd.read_csv("sample_1.csv",index_col=0)
    eg_df=eg_df.set_index(eg_df.columns[0])
    eg_df= eg_df.fillna(eg_df.mean())
    eg1 = Parameters(eg_df)
    timeseries = (eg1.data)[eg_df.columns[0]]
    parameters={'p1': eg1.get_cv(timeseries),
                       "p2": eg1.test_stationarity(timeseries),
                       "p3": eg1.measure_trend(timeseries),
                       "p4": eg1.auto_corr_measure(timeseries),
                       "p5": eg1.seasonality_measure(timeseries),
                       "p6": eg1.wavelet_rms(timeseries),
                       "p7": eg1.measure_entropy(timeseries),
                }
    print(parameters, "parameters")
