import pandas as pd
import json
# Assume your CSV data is already loaded into a DataFrame called df
if __name__ == "__main__":

    df =pd.read_csv('data/test/test_4.csv')
    # Convert to the desired JSON format
    json_data = df.apply(lambda row: {
        "point_timestamp": row['point_timestamp'],
        "point_value": float(row['point_value'])
    }, axis=1).tolist()

    output_file_path = 'data/test/test_4.json'

    # Write the JSON data to the output file
    with open(output_file_path, 'w') as output_file:
        json.dump(json_data, output_file)

    print('JSON data has been saved to', output_file_path)
