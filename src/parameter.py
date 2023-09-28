import numpy as np
import pandas as pd
import tsfresh
from manual_params import ParametersManual
class Parameters:
    def __init__(self, data_set: pd.DataFrame):
        self.data = data_set
    def measure_tsfresh_features(self, df):
        # Extract relevant TSFresh features
        tsfresh_features = tsfresh.extract_features(df,column_id="point_timestamp", column_sort="point_timestamp", column_value="point_value")
        # Initialize a dictionary to hold TSFresh features
        tsfresh_dict = {}
        i=0
        #print(tsfresh_features.columns,tsfresh_features.shape,tsfresh_features.iloc[:2,:].describe())
        for feature_name in tsfresh_features.columns:
            # Get the specified TSFresh feature (replace with actual feature names)
            tsfresh_feature = tsfresh_features[feature_name].mean()
            if not np.isnan(tsfresh_feature) and tsfresh_feature != 0:
                tsfresh_dict[f'tsfeature{i+1}'] = tsfresh_feature
            i+=1
        return tsfresh_dict     
    
    def get_params(self, df):
        tsfresh_features = self.measure_tsfresh_features(df)
        new_params={}
        manual_params=ParametersManual(data_set=df).get_params()
        new_params.update(manual_params)
        new_params.update(tsfresh_features)
        return new_params


if __name__ == "__main__":
    eg_df = pd.read_csv("data/daily/sample_1.csv", index_col=0)
    eg1 = Parameters(eg_df)
    print(eg_df.head())
    parameters = eg1.get_params(eg_df)
    print(parameters, "parameters")
