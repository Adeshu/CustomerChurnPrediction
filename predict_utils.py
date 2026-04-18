# predict_utils.py

import joblib
import numpy as np
import pandas as pd

def load_model(model_path):
    """Load a trained model from the specified path."""
    return joblib.load(model_path)

def predict_single_customer_churn(model, customer_data):
    """Predict churn for a single customer."""
    customer_df = pd.DataFrame([customer_data])
    prediction = model.predict(customer_df)
    return prediction[0]

def batch_predict_customers_churn(model, customers_data):
    """Predict churn for multiple customers."""
    customers_df = pd.DataFrame(customers_data)
    predictions = model.predict(customers_df)
    return predictions.tolist()

def example_usage():
    """Example usage of the prediction utilities."""
    # Load the model
    model_path = 'path_to_model/model.joblib'  # replace with actual path
    model = load_model(model_path)
    
    # Predict single customer churn
    single_customer = {'feature1': value1, 'feature2': value2, ...}  # replace with actual features
    prediction = predict_single_customer_churn(model, single_customer)
    print(f'Churn prediction for single customer: {prediction}')
    
    # Batch predict
    customers = [
        {'feature1': value1, 'feature2': value2, ...},  # replace with actual features
        {'feature1': value1, 'feature2': value2, ...},
        # Add more customers
    ]
    batch_predictions = batch_predict_customers_churn(model, customers)
    print(f'Churn predictions for batch: {batch_predictions}')
