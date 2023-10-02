import numpy as np
import pandas as pd
import tsfresh
from manual_params import ParametersManual

class Parameters:
    def __init__(self, data_set: pd.DataFrame):
        self.data = data_set

    def rename_ts_columns(self, df):
        ts_columns = [col for col in df.columns if col.startswith('ts')]
        ts_count = 1

        for col in ts_columns:
            df.rename(columns={col: f'tsfeature{ts_count}'}, inplace=True)
            ts_count += 1

        return df

    def measure_tsfresh_features(self, df):
        tsfeature_dict = {}

        try:
            # Extract relevant TSFresh features
            tsfresh_features = tsfresh.extract_features(df, column_id="point_timestamp", column_sort="point_timestamp", column_value="point_value")

            # Replace NaN values with 0
            tsfresh_features.fillna(0, inplace=True)

            # Flatten the tsfresh_features DataFrame and compute the mean for each feature
            tsfresh_features_mean = tsfresh_features.mean()
            tsfresh_features_mean_normalized = (tsfresh_features_mean - tsfresh_features_mean.mean()) / tsfresh_features_mean.std()

            # Multiply the normalized tsfeature values by 0.1
            tsfresh_features_mean_normalized *= 0.1
            i = 0
            for feature_name in tsfresh_features_mean.index:
                tsfeature_dict[f'tsfeature{i+1}'] = tsfresh_features_mean_normalized[feature_name]
                
                i += 1

        except Exception as e:
            print(f"Error extracting TSFresh features: {str(e)}")

        return tsfeature_dict

    def get_params(self, df):
        tsfeature_dict = self.measure_tsfresh_features(df)

        # Fill NaN values with 0 for tsfresh features
        for key, value in tsfeature_dict.items():
            if pd.isnull(value):
                tsfeature_dict[key] = 0

        new_params = {}
        manual_params = ParametersManual(data_set=df).get_params()
        new_params.update(manual_params)
        new_params.update(tsfeature_dict)

        return new_params

if __name__ == "__main__":
    eg_df = pd.read_csv("data/daily/sample_1.csv", index_col=0)
    eg1 = Parameters(eg_df)
    print(eg_df.head(), eg_df.index)
    parameters = eg1.get_params(eg_df)
    print(parameters, "parameters")
