import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Title
st.title("Customer Churn Prediction")

# Upload file
st.header("Upload Customer Data")
uploaded_file = st.file_uploader("Choose a file", type='csv')

if uploaded_file is not None:
    # Read the data
    data = pd.read_csv(uploaded_file)
    st.write(data.head())
    
    # Preprocessing
    # Assumed preprocessing steps
    X = data.drop('Churn', axis=1)
    y = data['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Prediction
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Display results
    st.write(f'Accuracy: {accuracy * 100:.2f}%')

# Footer
st.write("Developed by Adeshu")
