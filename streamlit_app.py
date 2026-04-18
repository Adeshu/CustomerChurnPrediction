import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="Churn Predictor", layout="wide")

st.title("🔮 Customer Churn Prediction")
st.markdown("---")

# Load model with new caching syntax
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/churn_model.pkl')
        return model
    except:
        return None

def preprocess_prediction_data(df):
    """Preprocess data for prediction (no Churn column)"""
    # Drop non-predictive columns
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    # Handle categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = pd.Categorical(df[col]).codes
    
    return df

model = load_model()

if model is None:
    st.error("❌ Model not found. Please train the model first.")
    st.stop()
else:
    st.success("✅ Model loaded successfully!")

# Sidebar for input
st.sidebar.header("Customer Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ['Male', 'Female'])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ['Yes', 'No'])
    dependents = st.selectbox("Dependents", ['Yes', 'No'])
    tenure = st.number_input("Tenure (months)", 0, 72, 24)

with col2:
    phone_service = st.selectbox("Phone Service", ['Yes', 'No'])
    multiple_lines = st.selectbox("Multiple Lines", ['Yes', 'No', 'No phone service'])
    internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox("Online Security", ['Yes', 'No', 'No internet service'])
    online_backup = st.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])

device_protection = st.selectbox("Device Protection", ['Yes', 'No', 'No internet service'])
tech_support = st.selectbox("Tech Support", ['Yes', 'No', 'No internet service'])
streaming_tv = st.selectbox("Streaming TV", ['Yes', 'No', 'No internet service'])
streaming_movies = st.selectbox("Streaming Movies", ['Yes', 'No', 'No internet service'])
contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
paperless_billing = st.selectbox("Paperless Billing", ['Yes', 'No'])
payment_method = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'])
monthly_charges = st.number_input("Monthly Charges ($)", 20.0, 120.0, 65.5)
total_charges = st.number_input("Total Charges ($)", 100.0, 10000.0, 1570.0)

# Create DataFrame with input
customer_data = pd.DataFrame({
    'gender': [gender],
    'SeniorCitizen': [senior_citizen],
    'Partner': [partner],
    'Dependents': [dependents],
    'tenure': [tenure],
    'PhoneService': [phone_service],
    'MultipleLines': [multiple_lines],
    'InternetService': [internet_service],
    'OnlineSecurity': [online_security],
    'OnlineBackup': [online_backup],
    'DeviceProtection': [device_protection],
    'TechSupport': [tech_support],
    'StreamingTV': [streaming_tv],
    'StreamingMovies': [streaming_movies],
    'Contract': [contract],
    'PaperlessBilling': [paperless_billing],
    'PaymentMethod': [payment_method],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges]
})

if st.button("🔮 Predict Churn", use_container_width=True):
    try:
        # Preprocess (without Churn column)
        X = preprocess_prediction_data(customer_data.copy())
        
        # Predict
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.error("⚠️ HIGH RISK - Customer likely to churn")
            else:
                st.success("✅ LOW RISK - Customer likely to stay")
        
        with col2:
            churn_prob = probability[1] * 100
            st.metric("Churn Probability", f"{churn_prob:.2f}%")
        
        st.markdown("---")
        
        # Show customer summary
        st.subheader("Customer Summary")
        summary_cols = st.columns(3)
        
        with summary_cols[0]:
            st.metric("Tenure", f"{tenure} months")
        with summary_cols[1]:
            st.metric("Monthly Charges", f"${monthly_charges:.2f}")
        with summary_cols[2]:
            st.metric("Total Charges", f"${total_charges:.2f}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.write(f"Debug: {e}")