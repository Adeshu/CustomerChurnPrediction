import argparse
from pathlib import Path

import pandas as pd

from main import ChurnPredictor

DEFAULT_MODEL_PATH = 'models/churn_model.pkl'


def parse_args():
    parser = argparse.ArgumentParser(description='Run churn predictions from a saved model artifact')
    parser.add_argument('--input', type=str, required=True, help='Input CSV for batch prediction')
    parser.add_argument('--output', type=str, default='models/predictions.csv', help='Output CSV path')
    return parser.parse_args()


def main():
    args = parse_args()
    if not Path(DEFAULT_MODEL_PATH).exists():
        raise FileNotFoundError(
            f'Model artifact not found at {DEFAULT_MODEL_PATH}. Train first with: '
            'python main.py --data data/customer_churn.csv --model \"Random Forest\"'
        )
    predictor = ChurnPredictor.load(DEFAULT_MODEL_PATH)
    input_df = pd.read_csv(args.input)

    predictions = predictor.predict(input_df)
    output_df = pd.concat([input_df, predictions], axis=1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f'Saved predictions: {output_path}')


if __name__ == '__main__':
    main()
