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
    # Generate synthetic time series data suitable for Prophet
    np.random.seed(i)  # Seed with a unique value for each dataset

    # Generate synthetic sequential data with daily seasonality and trend
    n = np.random.randint(100, 500)  # Number of data points
    t = pd.date_range(start='2020-01-01', periods=n, freq='D')
    data_points = 50 + 20 * np.sin(2 * np.pi * np.arange(n) / 7) + np.random.normal(scale=5, size=n)  # Daily sinusoidal pattern with noise

    # Create a DataFrame with the generated data
    df_prophet = pd.DataFrame({
        'index': range(n),
        'point_timestamp': t,
        'point_value': data_points
    })

    # Save to a CSV file
    filename = f'data/generated/prophetModel/prophet_data_{i + 1}.csv'
    df_prophet.to_csv(filename, index=False)

    print(f'Dataset {i + 1} saved to {filename}')
