import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os

# Create a directory to store the generated datasets
if not os.path.exists('data/generated'):
    os.makedirs('data/generated')

# Number of datasets to create
num_datasets = 100

# Generate and save each dataset
for i in range(num_datasets):
    # Generate synthetic time series data suitable for exponential smoothing
    np.random.seed(i)  # Seed with a unique value for each dataset

    # Generate synthetic sequential data with an exponential smoothing pattern
    n = np.random.randint(100, 500)  # Number of data points
    t = pd.date_range(start='2020-01-01', periods=n)

    # Generate data following exponential smoothing
    alpha = np.random.uniform(0.1, 0.9)  # Smoothing parameter
    beta = np.random.uniform(0.1, 0.9)  # Trend parameter
    gamma = np.random.uniform(0.1, 0.9)  # Seasonality parameter
    data_points = ExponentialSmoothing(np.random.normal(scale=5, size=n)).fit(smoothing_level=alpha, smoothing_slope=beta, smoothing_seasonal=gamma).fittedvalues

    # Create a DataFrame with the generated data
    df_exp_smooth = pd.DataFrame({
        'index': range(n),
        'point_timestamp': t,
        'point_value': data_points
    })

    # Save to a CSV file
    filename = f'data/generated/exponentialsmoothing/exponentialsmoothing_{i + 1}.csv'
    df_exp_smooth.to_csv(filename, index=False)

    print(f'Dataset {i + 1} saved to {filename}')
