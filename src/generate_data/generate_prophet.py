import numpy as np
import pandas as pd
import os

# Create a directory to store the generated datasets
if not os.path.exists('data/generated/prophetModel'):
    os.makedirs('data/generated/prophetModel')

# Number of datasets to create
num_datasets = 20

# Generate and save each dataset
for i in range(num_datasets):
    # Generate a random number of data points for this dataset
    n = np.random.randint(100, 500)  # Adjust the range as needed

    # Generate synthetic time series data suitable for Prophet
    np.random.seed(i)  # Seed with a unique value for each dataset
    t = np.arange(n)
    data_prophet = 50 + 10 * np.sin(0.05 * t)  # Sinusoidal pattern

    # Create a DataFrame with the generated data
    df_prophet = pd.DataFrame({
        'point_timestamp': pd.date_range(start='2020-01-01', periods=n),
        'point_value': data_prophet
    })

    # Add an index (row number) column
    df_prophet.insert(0, 'index', range(n))

    # Save to a CSV file
    filename = f'data/generated/prophetModel/prophet_data_{i + 1}.csv'
    df_prophet.to_csv(filename, index=False)

    print(f'Dataset {i + 1} with {n} data points saved to {filename}')
