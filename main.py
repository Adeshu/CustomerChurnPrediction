import argparse
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Setting up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Customer Churn Prediction Training Pipeline')
    parser.add_argument('--data', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size proportion')
    return parser.parse_args()

def load_data(file_path):
    logging.info('Loading data from %s', file_path)
    data = pd.read_csv(file_path)
    logging.info('Data loaded successfully with shape: %s', data.shape)
    return data

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    logging.info('Training model...')
    model.fit(X_train, y_train)
    logging.info('Model training completed.')
    return model

def evaluate_model(model, X_test, y_test):
    logging.info('Evaluating model...')
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    logging.info('Model accuracy: %.2f%%', accuracy * 100)
    logging.info('Classification report:
%s', report)

if __name__ == '__main__':
    args = parse_arguments()
    data = load_data(args.data)
    
    # Assuming the last column is the target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)
    
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
