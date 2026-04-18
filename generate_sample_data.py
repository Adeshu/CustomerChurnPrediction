import argparse
from pathlib import Path

import numpy as np
import pandas as pd


RNG = np.random.default_rng(42)


def generate(n_rows: int) -> pd.DataFrame:
    tenure = RNG.integers(0, 73, size=n_rows)
    monthly = np.round(RNG.uniform(18.0, 120.0, size=n_rows), 2)
    total = np.round(monthly * np.maximum(tenure, 1) * RNG.uniform(0.6, 1.4, size=n_rows), 2)

    contract = RNG.choice(['Month-to-month', 'One year', 'Two year'], size=n_rows, p=[0.55, 0.25, 0.20])
    internet = RNG.choice(['DSL', 'Fiber optic', 'No'], size=n_rows, p=[0.35, 0.45, 0.20])
    payment = RNG.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
        size=n_rows,
    )

    churn_score = (
        (contract == 'Month-to-month').astype(float) * 0.5
        + (internet == 'Fiber optic').astype(float) * 0.25
        + (payment == 'Electronic check').astype(float) * 0.2
        + (monthly > 85).astype(float) * 0.15
        + (tenure < 12).astype(float) * 0.25
        - (contract == 'Two year').astype(float) * 0.35
    )
    churn_prob = 1 / (1 + np.exp(-(churn_score - 0.35)))
    churn = np.where(RNG.random(n_rows) < churn_prob, 'Yes', 'No')

    df = pd.DataFrame(
        {
            'customerID': [f'CUST-{i:06d}' for i in range(1, n_rows + 1)],
            'gender': RNG.choice(['Male', 'Female'], size=n_rows),
            'SeniorCitizen': RNG.choice([0, 1], size=n_rows, p=[0.84, 0.16]),
            'Partner': RNG.choice(['Yes', 'No'], size=n_rows),
            'Dependents': RNG.choice(['Yes', 'No'], size=n_rows, p=[0.28, 0.72]),
            'tenure': tenure,
            'PhoneService': RNG.choice(['Yes', 'No'], size=n_rows, p=[0.9, 0.1]),
            'MultipleLines': RNG.choice(['Yes', 'No', 'No phone service'], size=n_rows, p=[0.42, 0.48, 0.10]),
            'InternetService': internet,
            'OnlineSecurity': RNG.choice(['Yes', 'No', 'No internet service'], size=n_rows, p=[0.30, 0.50, 0.20]),
            'OnlineBackup': RNG.choice(['Yes', 'No', 'No internet service'], size=n_rows, p=[0.35, 0.45, 0.20]),
            'DeviceProtection': RNG.choice(['Yes', 'No', 'No internet service'], size=n_rows, p=[0.35, 0.45, 0.20]),
            'TechSupport': RNG.choice(['Yes', 'No', 'No internet service'], size=n_rows, p=[0.30, 0.50, 0.20]),
            'StreamingTV': RNG.choice(['Yes', 'No', 'No internet service'], size=n_rows, p=[0.38, 0.42, 0.20]),
            'StreamingMovies': RNG.choice(['Yes', 'No', 'No internet service'], size=n_rows, p=[0.37, 0.43, 0.20]),
            'Contract': contract,
            'PaperlessBilling': RNG.choice(['Yes', 'No'], size=n_rows, p=[0.6, 0.4]),
            'PaymentMethod': payment,
            'MonthlyCharges': monthly,
            'TotalCharges': total,
            'Churn': churn,
        }
    )
    return df


def parse_args():
    parser = argparse.ArgumentParser(description='Generate sample customer churn dataset')
    parser.add_argument('--rows', type=int, default=1000, help='Number of rows to generate')
    parser.add_argument('--output', type=str, default='data/customer_churn.csv', help='Output CSV path')
    return parser.parse_args()


def main():
    args = parse_args()
    data = generate(args.rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False)
    print(f'Generated {len(data)} rows at: {output}')


if __name__ == '__main__':
    main()
