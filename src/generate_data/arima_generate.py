import numpy as np
import pandas as pd
import os

# Create a directory to store the generated datasets
if not os.path.exists('data/generated'):
    os.makedirs('data/generated')

# Number of datasets to create
num_datasets = 20

# Generate and save each dataset
for i in range(num_datasets):
    # Generate synthetic time series data suitable for ARIMA
    np.random.seed(i)  # Seed with a unique value for each dataset
    n =np.random.randint(100,500)  # Number of data points
    t = np.arange(n)
    data_arima = 50 + 0.5 * t + np.random.normal(scale=10, size=n)  # Linear trend with noise

    # Create a DataFrame with the generated data
    df_arima = pd.DataFrame({
        'point_timestamp': pd.date_range(start='2020-01-01', periods=n),
        'point_value': data_arima
    })

    # Add an index (row number) column
    df_arima.insert(0, 'index', range(n))

    # Save to a CSV file
    filename = f'data/generated/arima_data_{i + 1}.csv'
    df_arima.to_csv(filename, index=False)

    print(f'Dataset {i + 1} saved to {filename}')
