import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder

class ClassifierModel:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)  # Load the parameters data from CSV
        self.non_ts_features = ['Trend', 'Seasonality', 'Autocorrelation', 'Stationarity', 'Heteroscedasticity', 'Residual Patterns']
        self.ts_features = [col for col in self.data.columns if col.startswith('tsfeature')]  # Select columns that start with 'tsfeature'

    def preprocess_data(self):
        # Identify non-numeric columns and encode them
        non_numeric_columns = self.data.select_dtypes(exclude=['number']).columns
        for col in non_numeric_columns:
            if self.data[col].nunique() == 2:  # Binary categorical column
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col])
            else:  # Categorical column with more than two categories
                self.data = pd.concat([self.data, pd.get_dummies(self.data[col], prefix=col)], axis=1)
                self.data.drop(col, axis=1, inplace=True)

        # Normalize features (excluding best_model column)
        scaler = MinMaxScaler()
        self.data[self.data.columns[1:-1]] = scaler.fit_transform(self.data[self.data.columns[1:-1]])

    def train_classifier(self):
        # Prepare the training data (features) and labels (best_model)
        X = self.data[self.non_ts_features + self.ts_features]
        y = self.data['best_model']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        classifier.fit(X_train, y_train)

        # Test the classifier
        y_pred = classifier.predict(X_test)

        # Evaluate the classifier
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    model = ClassifierModel("data/train_params.csv")  # Adjust the path
    model.preprocess_data()
    model.train_classifier()
