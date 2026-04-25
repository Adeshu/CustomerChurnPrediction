# CustomerChurnPrediction

## Quickstart

```bash
python generate_sample_data.py --rows 2000 --output data/customer_churn.csv
python main.py --data data/customer_churn.csv --model "Random Forest"
python predict.py --input data/customer_churn.csv --output models/predictions.csv
streamlit run streamlit_app.py
```

## Notes

- Data files are ignored by default for privacy (`*.csv`).
- Use `generate_sample_data.py` to create local test datasets with the expected schema.
- Trained model artifact stores the model + feature encoders in `models/churn_model.pkl`.


python -m streamlit run streamlit_app.py