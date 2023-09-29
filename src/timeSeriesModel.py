import os
import importlib
import pandas as pd
import concurrent.futures

parent_dir=""
colab_run=False
if colab_run==True:
    parent_dir="/content/timeseries/timeseries-classifier/"

class TimeSeriesModel:
    def __init__(self, data: pd.Series):
        self.data = data

    def create_all_models(self):
        model_classes = self.load_model_classes()

        # Submit the tasks to the executor
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_model = {
                executor.submit(model(self.data.copy()).create_model): model_name
                for model_name, model in model_classes.items()
            }

            # Wait for the tasks to complete and retrieve the results
            errors = {}
            for future in concurrent.futures.as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    mape = future.result()
                    if isinstance(mape, pd.Series):
                        mape = float(mape)
                    errors[model_name] = mape
                except Exception as e:
                    print(f'{model_name} model failed with error: {e}')
        
        print(f'The errors: {errors}')

        # Get the name of the model with the lowest error
        best_model = min(errors, key=lambda k: errors[k])
        return best_model, errors[best_model]

    def load_model_classes(self):
        model_classes = {}
        models_folder = 'src/models'  # Change this to your models folder
        for file_name in os.listdir(parent_dir+models_folder):
           # print(os.listdir(models_folder))
            if file_name.endswith('.py') and file_name != '__init__.py':
                model_name = file_name[:-3]  # Remove the .py extension
                module = importlib.import_module(f'models.{model_name}')
                model_class_name = model_name + 'Model'
                if model_name == 'prophetModel':
                    model_class_name = 'prophetModel'

                model_class = getattr(module, model_class_name)
                model_classes[model_name] = model_class

        return model_classes

if __name__ == "__main__":
    data = pd.read_csv(f"{parent_dir}data/daily/sample_1.csv", index_col="point_timestamp")
    t1 = TimeSeriesModel(data)
    print(t1.create_all_models())
