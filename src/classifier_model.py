import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn import preprocessing
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight

import numpy as np

class TimeSeriesClassifier:
    
    def __init__(self, data_folder,filename,classifier_name):
        self.classifiername=classifier_name
        file_path = os.path.join(data_folder, filename)
        if os.path.exists(file_path):
            self.data = pd.read_csv(file_path)
        else:
            raise FileNotFoundError(f"File not found: {file_path}")
    def visualize_tree(self, clf, feature_names, class_names):
        # Visualize the decision tree
        plt.figure(figsize=(15, 10))
        tree.plot_tree(clf, filled=True, feature_names=feature_names, class_names=class_names)
        plt.show()
    def train_classifier(self):
        # Prepare features (X) and target (y)

        le = preprocessing.LabelEncoder()
        for i in range(len(self.data.columns)-1):
            self.data.iloc[:,i] = le.fit_transform(self.data.iloc[:,i])
        
        X = self.data.drop('best_model', axis=1)  # Features
        y = self.data['best_model']  # Target
        scaler = StandardScaler()
        X, y = shuffle(X, y, random_state=42)

        X= pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        # Lower the significance of 'ts' features
        for col in X.columns:
            if col.startswith('ts'):
                X[col] *= 0.1  # Adjust the multiplier as needed

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(len(X_train),len(X_test))
        # Train the classifier with the best parameters
        clf = RandomForestClassifier(
            n_estimators=10,
            max_depth=4,
            class_weight='balanced',
            random_state=42
        )
        clf.fit(X_train, y_train)
        # Predict on the test set
        y_pred = clf.predict(X_test)

        # Calculate accuracy
        print(f'ypred:{y_pred}\n ytest:{y_test}')
        accuracy = accuracy_score(y_test, y_pred)
        print('Accuracy:', accuracy)

        # Save the trained classifier
        feature_names = X.columns.tolist()  # Assuming X is your features dataframe
        class_names = [str(cls) for cls in clf.classes_]  # Convert class labels to strings
        
        self.visualize_tree(clf.estimators_[0], feature_names,class_names)

        joblib.dump(clf, self.classifiername)
        print('model_saved')

    
if __name__ == "__main__":
    data_folder = "data"  # Update this with your actual data folder
    classifier_filename = "trained_classifier.joblib"  # Update with desired classifier file name

    # Assuming train_params.csv contains the training data with 'best_model' as the target
    classifier = TimeSeriesClassifier(data_folder,"train_params.csv",classifier_filename)
    trained_classifier = classifier.train_classifier()
    