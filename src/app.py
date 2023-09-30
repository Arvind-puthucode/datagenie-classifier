from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from parameter import Parameters
from sklearn import preprocessing
import importlib

app = Flask(__name__)

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
    x.to_csv('whyerror.csv')
   # print(x.head(),x.describe())
    le = preprocessing.LabelEncoder()
    for i in range(len(x.columns)-1):
        x.iloc[:,i] = le.fit_transform(x.iloc[:,i])
        
        
    classifier = joblib.load('trained_classifier.joblib')
    
    best_model=(classifier.predict(x))[0]
    
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
    """# Use the best model's predictions
    predictions = all_models_predictions[best_model]

    # Add the best model to the DataFrame
    data['best_model'] = best_model

    # Generate the desired response
    response = {
        'model': best_model,
        'mape_error': '0.05',  # Replace with the actual MAPE error
        'result': data.to_dict(orient='records')
    }
    """
    return result_json

@app.route('/predict', methods=['POST'])
def predict():
    # Get the request parameters
    date_from = request.args.get('date_from')
    date_to = request.args.get('date_to')
    period = int(request.args.get('period', 0))

    # Example input data (list of dictionaries)
    input_data = request.get_json()

    # Preprocess input data
    df=preprocess_input(input_data)

    # Generate predictions and prepare the response
    response = generate_predictions(df)

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
