import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)


class ChurnPredictor:
    """Train/predict churn with persisted feature + label encoders."""

    def __init__(self, model_type: str = 'Random Forest', random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = self._build_model(model_type)
        self.feature_encoders: Dict[str, LabelEncoder] = {}
        self.target_encoder: Optional[LabelEncoder] = None
        self.feature_columns: List[str] = []
        self.categorical_columns: List[str] = []

    @staticmethod
    def _build_model(model_type: str):
        model_type_key = model_type.strip().lower()
        if model_type_key in {'random forest', 'random_forest', 'rf'}:
            return RandomForestClassifier(n_estimators=300, random_state=42)
        if model_type_key in {'logistic regression', 'logistic_regression', 'lr'}:
            return LogisticRegression(max_iter=1000, random_state=42)
        raise ValueError(f'Unsupported model type: {model_type}')

    @staticmethod
    def _normalize_total_charges(df: pd.DataFrame) -> pd.DataFrame:
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0.0)
        return df

    @staticmethod
    def _resolve_target_column(df: pd.DataFrame) -> str:
        if 'Churn' in df.columns:
            return 'Churn'
        lowered = {c.lower(): c for c in df.columns}
        if 'churn' in lowered:
            return lowered['churn']
        raise ValueError("'Churn' column not found in dataset")

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'customerID' in df.columns:
            df = df.drop(columns=['customerID'])
        return self._normalize_total_charges(df)

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        df = self._prepare_features(df)
        target_col = self._resolve_target_column(df)

        y_raw = df[target_col].astype(str)
        X = df.drop(columns=[target_col])
        self.feature_columns = X.columns.tolist()

        self.categorical_columns = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        for col in self.categorical_columns:
            encoder = LabelEncoder()
            values = X[col].fillna('Unknown').astype(str)
            classes = sorted(values.unique().tolist())
            if '__unknown__' not in classes:
                classes.append('__unknown__')
            encoder.fit(classes)
            self.feature_encoders[col] = encoder
            X[col] = values.map(lambda v: v if v in encoder.classes_ else '__unknown__')
            X[col] = encoder.transform(X[col])

        self.target_encoder = LabelEncoder()
        self.target_encoder.fit(['No', 'Yes'])
        y = pd.Series(
            self.target_encoder.transform(y_raw.map(lambda v: 'Yes' if str(v).strip().lower() in {'1', 'yes', 'true'} else 'No')),
            index=df.index,
        )

        return X, y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_columns:
            raise RuntimeError('Predictor has not been fitted or loaded')

        df = self._prepare_features(df)
        X = df.copy()

        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0

        X = X[self.feature_columns]

        for col in self.categorical_columns:
            encoder = self.feature_encoders[col]
            values = X[col].fillna('Unknown').astype(str)
            values = values.map(lambda v: v if v in encoder.classes_ else '__unknown__')
            X[col] = encoder.transform(values)

        return X

    def train(self, df: pd.DataFrame, test_size: float = 0.2):
        X, y = self.fit_transform(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y,
        )

        LOGGER.info('Training %s model...', self.model_type)
        self.model.fit(X_train, y_train)
        metrics = self.evaluate(X_test, y_test)
        return metrics

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, object]:
        predictions = self.model.predict(X_test)
        probabilities = self.model.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': float(accuracy_score(y_test, predictions)),
            'roc_auc': float(roc_auc_score(y_test, probabilities)),
            'confusion_matrix': confusion_matrix(y_test, predictions).tolist(),
            'classification_report': classification_report(y_test, predictions, target_names=self.target_encoder.classes_, output_dict=True),
        }

        LOGGER.info('Accuracy: %.4f', metrics['accuracy'])
        LOGGER.info('ROC AUC: %.4f', metrics['roc_auc'])
        LOGGER.info('Confusion matrix: %s', metrics['confusion_matrix'])
        return metrics

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self.transform(df)
        pred_idx = self.model.predict(X)
        pred_prob = self.model.predict_proba(X)[:, 1]
        labels = self.target_encoder.inverse_transform(pred_idx)
        return pd.DataFrame({'prediction': labels, 'churn_probability': pred_prob})

    def save(self, model_path: str):
        artifact = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_columns': self.feature_columns,
            'categorical_columns': self.categorical_columns,
            'feature_encoders': self.feature_encoders,
            'target_encoder': self.target_encoder,
            'random_state': self.random_state,
        }
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, model_path)
        LOGGER.info('Saved model artifact: %s', model_path)

    @classmethod
    def load(cls, model_path: str) -> 'ChurnPredictor':
        artifact = joblib.load(model_path)
        predictor = cls(model_type=artifact['model_type'], random_state=artifact.get('random_state', 42))
        predictor.model = artifact['model']
        predictor.feature_columns = artifact['feature_columns']
        predictor.categorical_columns = artifact['categorical_columns']
        predictor.feature_encoders = artifact['feature_encoders']
        predictor.target_encoder = artifact['target_encoder']
        return predictor


def parse_arguments():
    parser = argparse.ArgumentParser(description='Customer Churn Prediction Training Pipeline')
    parser.add_argument('--data', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test split size')
    parser.add_argument('--model', type=str, default='Random Forest', choices=['Random Forest', 'Logistic Regression'], help='Model algorithm')
    parser.add_argument('--model-out', type=str, default='models/churn_model.pkl', help='Output model artifact path')
    parser.add_argument('--metrics-out', type=str, default='models/metrics.json', help='Output metrics path')
    return parser.parse_args()


def main():
    args = parse_arguments()
    data = pd.read_csv(args.data)

    predictor = ChurnPredictor(model_type=args.model)
    metrics = predictor.train(data, test_size=args.test_size)
    predictor.save(args.model_out)

    metrics_out = Path(args.metrics_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    LOGGER.info('Saved evaluation metrics: %s', metrics_out)


if __name__ == '__main__':
    main()
