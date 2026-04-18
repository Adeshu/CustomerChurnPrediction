import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChurnPredictor:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_names = None
        
    def load_data(self, file_path):
        logging.info('Loading data from %s', file_path)
        data = pd.read_csv(file_path)
        logging.info('Data loaded successfully with shape: %s', data.shape)
        return data
    
    def preprocess_data(self, df, fit=False):
        """Preprocess data for modeling"""
        df = df.copy()
        
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
        
        if 'Churn' in df.columns:
            y = df['Churn'].map({'Yes': 1, 'No': 0})
            X = df.drop('Churn', axis=1)
        else:
            y = None
            X = df
        
        # Encode categorical variables properly
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if fit:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    X[col] = le.transform(X[col].astype(str))
        
        if fit:
            self.feature_names = X.columns.tolist()
        
        return X, y
    
    def train(self, X_train, y_train, model_name='Random Forest'):
        logging.info('Training %s model...', model_name)
        
        if model_name == 'Random Forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        
        self.model.fit(X_train, y_train)
        logging.info('Model training completed.')
        return self.model
    
    def evaluate(self, model, X_test, y_test):
        logging.info('Evaluating model...')
        predictions = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        
        logging.info('Model accuracy: %.4f', accuracy)
        logging.info('Precision: %.4f', precision)
        logging.info('Recall: %.4f', recall)
        logging.info('F1-Score: %.4f', f1)
        logging.info('Classification report:\n%s', classification_report(y_test, predictions))
        
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def save(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        
        joblib.dump(self.model, os.path.join(save_path, 'churn_model.pkl'))
        joblib.dump(self.label_encoders, os.path.join(save_path, 'label_encoders.pkl'))
        joblib.dump(self.feature_names, os.path.join(save_path, 'feature_names.pkl'))
        
        logging.info('Model saved to %s', save_path)
    
    def load(self, model_path):
        self.model = joblib.load(os.path.join(model_path, 'churn_model.pkl'))
        self.label_encoders = joblib.load(os.path.join(model_path, 'label_encoders.pkl'))
        self.feature_names = joblib.load(os.path.join(model_path, 'feature_names.pkl'))
        logging.info('Model and encoders loaded successfully')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Customer Churn Prediction Training Pipeline')
    parser.add_argument('--data', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size proportion')
    parser.add_argument('--model', type=str, default='Random Forest', help='Model to train')
    parser.add_argument('--save', type=str, default='models/', help='Path to save model')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    
    print("\n" + "="*70)
    print("CUSTOMER CHURN PREDICTION - TRAINING PIPELINE")
    print("="*70 + "\n")
    
    try:
        predictor = ChurnPredictor()
        
        data = predictor.load_data(args.data)
        X, y = predictor.preprocess_data(data, fit=True)
        logging.info('Features shape: %s', X.shape)
        logging.info('Churn distribution - Yes: %d, No: %d', y.sum(), len(y) - y.sum())
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42, stratify=y)
        logging.info('Train set: %d, Test set: %d', X_train.shape[0], X_test.shape[0])
        
        model = predictor.train(X_train, y_train, args.model)
        metrics = predictor.evaluate(model, X_test, y_test)
        
        if args.save:
            predictor.save(args.save)
        
        print("\n" + "="*70)
        print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70 + "\n")
        
    except Exception as e:
        logging.error('Error: %s', str(e), exc_info=True)
        print(f"\n✗ Error: {str(e)}\n")