from flask import Flask, request, jsonify
import pandas as pd
import joblib
import json


from parameter import Parameters
from sklearn import preprocessing
import importlib
import logging
from flask_cors import CORS
app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:8000"}})


# Load the trained classifier

def preprocess_input(input_data):
    # Convert the input JSON to a DataFrame
    df = pd.DataFrame(input_data)
    
    # Set 'point_timestamp' as the index
    return df

def generate_predictions(data):
    
    # Predict using the loaded classifier for all models
    obj=Parameters(data)
    p=obj.get_params(data)
    x=pd.DataFrame(p,index=[0])
    le = preprocessing.LabelEncoder()
    string_columns = x.select_dtypes(include=['object']).columns
        # Iterate through string columns and encode them
    for col in string_columns:
        print('col',col)    
        x[col] = le.fit_transform(x[col])
        
    classifier = joblib.load('trained_classifier.joblib')
    print(x.head(),x.columns)
    
    print("Number of Estimators:", classifier.n_estimators)

    best_model=(classifier.predict(x))[0]
    print(best_model)
    bm_v=best_model
    # Create an object of the best_model class
    module = importlib.import_module(f'models.{best_model}')
    print('best model is \n\n\n\n',best_model)
    if best_model == 'prophetModel':
        best_model = 'prophetModel'
    else:
         best_model+='Model'
    best_model_class = getattr(module, best_model)
    best_model_object = best_model_class(data)  # Assuming the class doesn't require any arguments for initialization

    # Call the result_json attribute of the best_model object
    result_json = best_model_object.result_json()
    print(result_json)
    # Extract mape, y_pred, and y_test from the original data
    mape = result_json["mape"]
    y_pred = result_json["y_pred"]
    y_test = result_json["y_test"]
    dates=result_json["point_timestamp"]
    # Construct the desired JSON format
    desired_json = {
        "model": bm_v,  # Replace with the actual model name
        "mape": mape,
        "result": []
    }

    # Populate the "result" list based on y_pred and y_test
    for pred, test,tstamp in zip(y_pred, y_test,dates):
        result_entry = {
            "point_timestamp": tstamp, # Replace with the actual date
            "point_value": test,
            "yhat": pred
        }
        desired_json["result"].append(result_entry)

    return desired_json

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read the data directly from the request body
        file_content = request.get_data(as_text=True)
        # Parse the file content as JSON
        input_data = json.loads(file_content)

        df = pd.DataFrame(input_data)
        result_json = generate_predictions(df)

        return jsonify(result_json)
    except Exception as e:
        logging.exception('An error occurred during prediction')
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
