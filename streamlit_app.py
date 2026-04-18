from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from main import ChurnPredictor

st.set_page_config(page_title='Customer Churn Prediction', layout='wide')

DEFAULT_MODEL_PATH = 'models/churn_model.pkl'

DEFAULT_INPUT = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'No',
    'Dependents': 'Yes',
    'tenure': 24,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 70.0,
    'TotalCharges': 1680.0,
}


@st.cache_resource
def load_predictor(model_path: str):
    if not Path(model_path).exists():
        return None
    return ChurnPredictor.load(model_path)


@st.cache_data
def load_metrics(metrics_path: str):
    path = Path(metrics_path)
    if not path.exists():
        return None
    return pd.read_json(path, typ='series')


def customer_input_form() -> pd.DataFrame:
    st.sidebar.header('Customer Details')

    input_payload = DEFAULT_INPUT.copy()

    input_payload['gender'] = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    input_payload['SeniorCitizen'] = st.sidebar.selectbox('Senior Citizen', [0, 1])
    input_payload['Partner'] = st.sidebar.selectbox('Partner', ['Yes', 'No'])
    input_payload['Dependents'] = st.sidebar.selectbox('Dependents', ['Yes', 'No'])
    input_payload['tenure'] = st.sidebar.slider('Tenure (months)', min_value=0, max_value=72, value=24)
    input_payload['PhoneService'] = st.sidebar.selectbox('Phone Service', ['Yes', 'No'])
    input_payload['MultipleLines'] = st.sidebar.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service'])
    input_payload['InternetService'] = st.sidebar.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    input_payload['OnlineSecurity'] = st.sidebar.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
    input_payload['OnlineBackup'] = st.sidebar.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
    input_payload['DeviceProtection'] = st.sidebar.selectbox('Device Protection', ['Yes', 'No', 'No internet service'])
    input_payload['TechSupport'] = st.sidebar.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])
    input_payload['StreamingTV'] = st.sidebar.selectbox('Streaming TV', ['Yes', 'No', 'No internet service'])
    input_payload['StreamingMovies'] = st.sidebar.selectbox('Streaming Movies', ['Yes', 'No', 'No internet service'])
    input_payload['Contract'] = st.sidebar.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    input_payload['PaperlessBilling'] = st.sidebar.selectbox('Paperless Billing', ['Yes', 'No'])
    input_payload['PaymentMethod'] = st.sidebar.selectbox(
        'Payment Method',
        ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
    )
    input_payload['MonthlyCharges'] = st.sidebar.slider('Monthly Charges ($)', min_value=18.0, max_value=130.0, value=70.0)
    input_payload['TotalCharges'] = st.sidebar.slider('Total Charges ($)', min_value=0.0, max_value=9000.0, value=1680.0)

    return pd.DataFrame([input_payload])


def risk_label(probability: float) -> str:
    if probability >= 0.7:
        return 'High Risk'
    if probability >= 0.4:
        return 'Medium Risk'
    return 'Low Risk'


def main():
    st.title('🔮 Customer Churn Prediction')
    st.write('Production inference app using the trained model artifact and feature encoders.')

    model_path = st.text_input('Model artifact path', value=DEFAULT_MODEL_PATH)
    predictor = load_predictor(model_path)

    if predictor is None:
        st.error(f'Model artifact not found at `{model_path}`. Train first with `python main.py --data <csv>`.')
        st.stop()

    st.success('Model loaded successfully.')

    cols = st.columns(2)
    with cols[0]:
        metrics = load_metrics('models/metrics.json')
        if metrics is not None:
            st.metric('Validation Accuracy', f"{float(metrics['accuracy']):.3f}")
            st.metric('Validation ROC-AUC', f"{float(metrics['roc_auc']):.3f}")

    with cols[1]:
        if st.button('Show Feature List'):
            st.write(predictor.feature_columns)

    customer_df = customer_input_form()

    if st.button('Predict Churn', use_container_width=True):
        result = predictor.predict(customer_df).iloc[0]
        prob = float(result['churn_probability'])
        prediction = str(result['prediction'])

        rcol1, rcol2, rcol3 = st.columns(3)
        rcol1.metric('Prediction', prediction)
        rcol2.metric('Churn Probability', f'{prob * 100:.2f}%')
        rcol3.metric('Risk Assessment', risk_label(prob))

        fig = px.bar(
            x=['Stay', 'Churn'],
            y=[1 - prob, prob],
            labels={'x': 'Outcome', 'y': 'Probability'},
            title='Prediction Probability Distribution',
            color=['Stay', 'Churn'],
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader('Input Payload')
        st.dataframe(customer_df, use_container_width=True)


if __name__ == '__main__':
    main()
