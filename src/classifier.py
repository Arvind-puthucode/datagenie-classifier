import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from parameter import Parameters
from timeSeriesModel import TimeSeriesModel
# Path to the data folder
main_dir = "/data/"

# Define the function for training the classifier and saving it

# Path to the data folder
main_dir="/data"
    
def train_and_save_classifier():
    params_data=[]
    subfolders = ['daily', 'hourly', 'weekly', 'monthly']
    dfs = {}
    for subfolder in subfolders:
        # Get the path to the subfolder
        subfolder_path = os.path.join(main_dir, subfolder)
        print('\n sub_path \n',subfolder_path,'\n')
        csv_files = [f for f in os.listdir(subfolder_path) if f.endswith('.csv')]
        for csv_file in csv_files:
            df_name = f'{subfolder}\{csv_file[:-4]}'
            print(f'\n\n df_name is {df_name}\n\n',)
            df = pd.read_csv(os.path.join(subfolder_path, csv_file),index_col=0)
            df=df.set_index(df.columns[0])
            df= df.fillna(df.mean())

            print(df_name,df.head(1)) 
            p1=Parameters(df)
            timeseries = (df)[df.columns[0]]
            params=p1.get_params(timeseries)
            best_model,error_model=TimeSeriesModel(df).create_all_models()
            params['best_model']=best_model
            params_data.append(params)

    df = pd.DataFrame(params_data)
    df.to_csv(f'{main_dir}/train_params.csv')
    # Save the trained classifier to a file
    print(f'Classifier for {subfolder} saved successfully.')

if __name__ == "__main__":
    train_and_save_classifier()
    