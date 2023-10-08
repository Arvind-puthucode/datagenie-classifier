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
    try:
        subfolder_path = os.path.join(parent_dir+main_dir, subfolder)
        df_name = f'{subfolder}/{csv_file[:-4]}'
        df = pd.read_csv(os.path.join(subfolder_path, csv_file), index_col=0)
        
        df['point_value'].fillna(value=0,inplace=True)
        return df,df_name
    except:
        return (None,None)
def paramsforgenerate(df,df_name,model):
 #   df['point_value'].fillna(0, inplace=True)
    print(f'\nProcessing {df_name}, head: {df.head(1)}')
    p1 = Parameters(df)
    params = p1.get_params(df)
    best_model = model
    params['best_model'] = best_model
    return params

def calcualtemodel(df,df_name):
    try:
        df['point_value'].fillna(0, inplace=True)
        print(f'\nProcessing {df_name}, head: {df.head(1)}')
        p1 = Parameters(df)
        params = p1.get_params(df)
        best_model,_ = TimeSeriesModel(df).create_all_models()
        params['best_model'] = best_model

        return params
    except:
        print('df no point',df.head())

# Define the function for training the classifier and saving it
def train_and_save_classifier():
    params_data = []
    subfolders = []
    #'daily','weekly',hourly,'monthly'

    for subfolder in subfolders:
        subfolder_path = os.path.join(parent_dir+main_dir, subfolder)
        csv_files = [f for f in os.listdir(subfolder_path) if f.endswith('.csv')]
        for csv_file in csv_files:
            dfdummy,dfname=process_csv(subfolder, csv_file)
            params=calcualtemodel(dfdummy,dfname)
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

def train_generated_data():
    params_data = []
    #'arima','lstm',,'prophetModel'
    subfolders = ['exponentialsmoothing']

    global main_dir
    main_dir='data/generated'
    for subfolder in subfolders:
        subfolder_path = os.path.join(parent_dir+main_dir, subfolder)
        csv_files = [f for f in os.listdir(subfolder_path) if f.endswith('.csv')]
        for csv_file in csv_files:
            dfd,dfn = process_csv(subfolder, csv_file)
            if dfd is None and dfn is None:
                continue
            params=paramsforgenerate(dfd,dfn,subfolder)
            if params is not None:
                params_data.append(params)
    if not params_data:
        print('No valid datasets were processed.')
        return

    df = pd.DataFrame(params_data)
    # Check if the file exists
    main_dir='data'
    csv_file_path = f'{parent_dir+main_dir}/train_params.csv'
    if os.path.isfile(csv_file_path):
        # Append to the existing CSV file

        df.to_csv(csv_file_path, mode='a', header=False, index=False,columns=df.columns)
    else:
        # If the file doesn't exist, create a new one
        df.to_csv(csv_file_path, index=False)

    print("Processing completed.")


    
if __name__ == "__main__":
   # train_and_save_classifier()
    train_generated_data()