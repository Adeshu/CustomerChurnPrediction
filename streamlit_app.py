import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Custom CSS styling
st.markdown('<style>body{background-color: #f5f5f5;} .metric-card{border: 1px solid #d1d1d1; border-radius: 12px; padding: 10px; margin: 10px 0;}</style>', unsafe_allow_html=True)

# Title
st.title('Customer Churn Prediction')

# Sidebar navigation
st.sidebar.title('Navigation')
options = st.sidebar.radio('Select a page:', ['Dashboard', 'Single Prediction', 'Batch Upload', 'Analytics'])

# Load dataset
@st.cache
def load_data():
    # This is just a placeholder. Load your actual dataset here.
    data = pd.DataFrame({
        'CustomerID': range(1, 101),
        'Churn': np.random.choice([0, 1], size=100),
        'Tenure': np.random.randint(1, 73, size=100),
        'MonthlyCharges': np.random.uniform(20.0, 120.0, size=100)
    })
    return data

data = load_data()

# Dashboard page
if options == 'Dashboard':
    st.header('Dashboard')
    fig = px.histogram(data, x='Tenure', color='Churn', title='Customer Churn by Tenure')
    st.plotly_chart(fig)
    churn_rate = data['Churn'].mean() * 100
    st.markdown(f'<div class="metric-card"><h3>Churn Rate</h3><h1>{churn_rate:.2f}%</h1></div>', unsafe_allow_html=True)

# Single prediction page
elif options == 'Single Prediction':
    st.header('Single Prediction')
    tenure = st.number_input('Customer Tenure (months)', min_value=1, max_value=72)
    monthly_charges = st.number_input('Monthly Charges', min_value=20.0, max_value=120.0)
    if st.button('Predict'):
        # Placeholder for predictive model
        prediction = 'Yes' if np.random.rand() > 0.5 else 'No'
        st.write('Churn Prediction:', prediction)

# Batch upload page
elif options == 'Batch Upload':
    st.header('Batch Upload')
    uploaded_file = st.file_uploader('Upload CSV file', type=['csv'])
    if uploaded_file:
        batch_data = pd.read_csv(uploaded_file)
        # Process predictions (placeholder)
        st.write(batch_data)

# Analytics page
elif options == 'Analytics':
    st.header('Analytics')
    # Placeholder for analytics content
    st.write('Analytics content goes here...')

# Run the app
if __name__ == '__main__':
    st.write('Streamlit app is running...')
