import tempfile
import unittest
from pathlib import Path

import numpy as np

from generate_sample_data import generate
from main import ChurnPredictor


class TestChurnPredictor(unittest.TestCase):
    def setUp(self):
        self.data = generate(300)

    def test_train_and_predict(self):
        predictor = ChurnPredictor(model_type='Random Forest')
        metrics = predictor.train(self.data, test_size=0.25)

        self.assertIn('accuracy', metrics)
        self.assertIn('roc_auc', metrics)

        result = predictor.predict(self.data.drop(columns=['Churn']).head(5))
        self.assertEqual(len(result), 5)
        self.assertIn('prediction', result.columns)
        self.assertIn('churn_probability', result.columns)

    def test_save_and_load_preserves_predictions(self):
        predictor = ChurnPredictor(model_type='Logistic Regression')
        predictor.train(self.data, test_size=0.2)

        sample = self.data.drop(columns=['Churn']).head(10)
        before = predictor.predict(sample)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / 'churn_model.pkl'
            predictor.save(str(model_path))
            reloaded = ChurnPredictor.load(str(model_path))
            after = reloaded.predict(sample)

        self.assertListEqual(before['prediction'].tolist(), after['prediction'].tolist())
        np.testing.assert_allclose(before['churn_probability'].to_numpy(), after['churn_probability'].to_numpy())


if __name__ == '__main__':
    unittest.main()
