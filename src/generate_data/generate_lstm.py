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
    # Generate synthetic time series data suitable for LSTM
    rng = np.random.default_rng(i)  # Seed with a unique value for each dataset

    # Generate synthetic sequential data with a sinusoidal pattern
    n = rng.integers(100, 501)  # Number of data points
    t = pd.date_range(start='2020-01-01', periods=n)
    sinusoidal_pattern = 50 + 20 * np.sin(np.linspace(0, 10 * np.pi, n))
    noise = rng.normal(scale=5, size=n)
    data_points = sinusoidal_pattern + noise  # Sinusoidal pattern with noise

    # Create a DataFrame with the generated data
    df_lstm = pd.DataFrame({
        'index': range(n),
        'point_timestamp': t,
        'point_value': data_points
    })

    # Save to a CSV file
    filename = f'data/generated/lstm/lstm_data_{i + 1}.csv'
    df_lstm.to_csv(filename, index=False)

    print(f'Dataset {i + 1} saved to {filename}')
