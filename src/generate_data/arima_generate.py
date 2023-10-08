import numpy as np
import pandas as pd
import os

# Create a directory to store the generated datasets
if not os.path.exists('data/generated'):
    os.makedirs('data/generated')

# Number of datasets to create
num_datasets = 100

# Generate and save each dataset
for i in range(num_datasets):
    # Generate synthetic time series data suitable for ARIMA
    np.random.seed(i)  # Seed with a unique value for each dataset

    # Generate synthetic sequential data with an autoregressive pattern
    rng = np.random.default_rng(seed=5)
    # Generate a random integer between 100 and 500 (inclusive)
    n = rng.integers(100, 501)  # 501 is used to include 500 in the range 
    t = pd.date_range(start='2020-01-01', periods=n)
    random_numbers = rng.normal(scale=5, size=n)
    data_points = 50 + 0.9 * np.arange(n) + random_numbers  # Autoregressive pattern with noise

    # Create a DataFrame with the generated data
    df_arima = pd.DataFrame({
        'index': range(n),
        'point_timestamp': t,
        'point_value': data_points
    })

    # Save to a CSV file
    filename = f'data/generated/arima/arima_data_{i + 1}.csv'
    df_arima.to_csv(filename, index=False)

    print(f'Dataset {i + 1} saved to {filename}')
