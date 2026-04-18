# Configuration Parameters for Customer Churn Prediction

CONFIG = {
    'data_paths': {
        'training_data': 'data/train.csv',
        'testing_data': 'data/test.csv'
    },
    'model_names': {
        'logistic_regression': 'LogisticRegression',
        'random_forest': 'RandomForestClassifier'
    },
    'features': {
        'categorical': ['gender', 'geography', 'has_cr_card', 'is_active_member'],
        'numerical': ['age', 'credit_score', 'balance', 'estimated_salary']
    },
    'hyperparameters': {
        'logistic_regression': {'solver': 'liblinear', 'max_iter': 100},
        'random_forest': {'n_estimators': 100, 'max_depth': 10}
    },
    'evaluation_metrics': ['accuracy', 'precision', 'recall', 'f1_score']
}