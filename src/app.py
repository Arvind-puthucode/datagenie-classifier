from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
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
    #print(p)
    x=pd.DataFrame(p,index=[0])
    x=obj.rename_ts_columns(x)
    print(x.head(),x.describe())
    le = preprocessing.LabelEncoder()
    for i in range(len(x.columns)-1):
        x.iloc[:,i] = le.fit_transform(x.iloc[:,i])
        
        
    classifier = joblib.load('trained_classifier.joblib')
    
    best_model=(classifier.predict(x))[0]
    bm_v=best_model
    # Create an object of the best_model class
    module = importlib.import_module(f'models.{best_model}')
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
"""
@app.route('/')
def home():
    return render_template('index.html')
"""

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
