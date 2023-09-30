import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from parameter import Parameters
from timeSeriesModel import TimeSeriesModel

# Path to the data folder
parent_dir=""
colab_run=False
if colab_run==True:
    parent_dir="/content/timeseries/timeseries-classifier"
main_dir = "data"

# Define a function to process a single CSV file
def process_csv(subfolder, csv_file):
    subfolder_path = os.path.join(parent_dir+main_dir, subfolder)
    df_name = f'{subfolder}/{csv_file[:-4]}'
    df = pd.read_csv(os.path.join(subfolder_path, csv_file), index_col=0)
    df['point_value'].fillna(0, inplace=True)
    data=pd.read_csv(os.path.join(subfolder_path, csv_file), index_col='point_timestamp')
    print(f'\nProcessing {df_name}, head: {df.head(1)}')
    p1 = Parameters(df)
    params = p1.get_params(df)
    best_model, error_model = TimeSeriesModel(data).create_all_models()
    params['best_model'] = best_model

    return params

# Define the function for training the classifier and saving it
def train_and_save_classifier():
    params_data = []
    subfolders = []

    for subfolder in subfolders:
        subfolder_path = os.path.join(parent_dir+main_dir, subfolder)
        csv_files = [f for f in os.listdir(subfolder_path) if f.endswith('.csv')]
        for csv_file in csv_files:
            params = process_csv(subfolder, csv_file)
            if params is not None:
                params_data.append(params)

    if not params_data:
        print('No valid datasets were processed.')
        return

    df = pd.DataFrame(params_data)
    # Check if the file exists
    csv_file_path = f'{parent_dir+main_dir}/train_params.csv'
    if os.path.isfile(csv_file_path):
        # Append to the existing CSV file

        df.to_csv(csv_file_path, mode='a', header=False, index=False,columns=df.columns)
    else:
        # If the file doesn't exist, create a new one
        df.to_csv(csv_file_path, index=False)

    print("Processing completed.")


if __name__ == "__main__":
    train_and_save_classifier()
