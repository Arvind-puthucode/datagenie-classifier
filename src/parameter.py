import numpy as np
import pandas as pd
import tsfresh
from manual_params import ParametersManual

class Parameters:
    def __init__(self, data_set: pd.DataFrame):
        self.data = data_set
        self.NUM_TSFRESH_FEATURES = 47
    def rename_ts_columns(self,df):
        ts_columns = [col for col in df.columns if col.startswith('ts')]
        ts_count = 1

        for col in ts_columns:
            df.rename(columns={col: f'tsfeature{ts_count}'}, inplace=True)
            ts_count += 1

        return df
    def add_missing_tsfresh_features(self, tsfresh_dict):
        # If the number of TSFresh features is less than the desired number, add more features with value 1
        for i in range(len(tsfresh_dict) + 1, self.NUM_TSFRESH_FEATURES + 1):
            tsfresh_dict[f'tsfeature{i}'] = 1.0
        return tsfresh_dict
    def measure_tsfresh_features(self, df):
        tsfresh_dict = {}
        
        try:
            # Extract relevant TSFresh features
            tsfresh_features = tsfresh.extract_features(df, column_id="point_timestamp", column_sort="point_timestamp", column_value="point_value")

            i = 0
            for feature_name in tsfresh_features.columns:
                # Get the specified TSFresh feature (replace with actual feature names)
                tsfresh_feature = tsfresh_features[feature_name].mean()
                if not np.isnan(tsfresh_feature) and tsfresh_feature != 0:
                    tsfresh_dict[f'tsfeature{i+1}'] = tsfresh_feature
                i += 1

        except Exception as e:
            print(f"Error extracting TSFresh features: {str(e)}")

        return tsfresh_dict

    def get_params(self, df):
        tsfresh_features = self.measure_tsfresh_features(df)
        new_params = {}
        manual_params = ParametersManual(data_set=df).get_params()
        tsfresh_features=self.add_missing_tsfresh_features(tsfresh_features)

        new_params.update(manual_params)
        new_params.update(tsfresh_features)
        
        return new_params

if __name__ == "__main__":
    eg_df = pd.read_csv("data/daily/sample_1.csv", index_col=0)
    eg1 = Parameters(eg_df)
    print(eg_df.head(),eg_df.index)
    parameters = eg1.get_params(eg_df)
    print(parameters, "parameters")
