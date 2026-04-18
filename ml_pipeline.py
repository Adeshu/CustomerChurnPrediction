import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load dataset
def load_data(file_path):
    return pd.read_csv(file_path)

# Preprocess data
def preprocess_data(df):
    # Handling missing values
    imputer = SimpleImputer(strategy='mean')
    df[['feature1', 'feature2']] = imputer.fit_transform(df[['feature1', 'feature2']])
    
    # Encoding categorical variables
    df = pd.get_dummies(df, columns=['categorical_feature'], drop_first=True)
    
    # Scaling features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df.drop('target', axis=1))
    
    return pd.DataFrame(scaled_features, columns=df.columns[:-1]), df['target']

# Train model
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluate model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Classification Report:\n", classification_report(y_test, predictions))
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))

# Main function
if __name__ == "__main__":
    # Load data
    df = load_data('path/to/your/dataset.csv')
    
    # Preprocess data
    X, y = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_test, y_test)