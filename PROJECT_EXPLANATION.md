# Customer Churn Prediction Project - Complete Explanation

## 📌 Project Overview

**Objective:** Predict whether a customer will churn (leave/stop using services) based on their characteristics and behavior.

**Real-world Use:** Telecom/SaaS companies can identify at-risk customers and take retention actions (discounts, better support, etc.).

---

## 🎯 How It Works - Simple Explanation

### Step 1: Historical Data
```
We have 2000 customers with their:
- Demographics (age, gender, dependents, etc.)
- Services used (internet, phone, TV, security, etc.)
- Billing info (monthly charges, contract type, etc.)
- RESULT: Did they churn? (Yes/No)
```

### Step 2: Machine Learning Model Learns
```
Model sees patterns:
- "Customers with month-to-month contracts churn more"
- "Customers with 2-year contracts rarely churn"
- "Fiber optic + high charges = higher churn risk"
- etc.
```

### Step 3: Predict New Customers
```
New Customer:
Input: gender=Female, tenure=24 months, contract=month-to-month, charges=$70
Output: 65% chance they will churn (Medium Risk)
```

---

## 📁 Project File Structure

```
CustomerChurnPrediction/
├── data/
│   └── customer_churn.csv          ← Training dataset
├── models/
│   ├── churn_model.pkl             ← Trained model + encoders
│   ├── metrics.json                ← Performance metrics
│   └── predictions.csv             ← Batch predictions
├── main.py                         ← Training script (ChurnPredictor class)
├── predict.py                      ← Batch prediction script
├── streamlit_app.py                ← Interactive web interface
├── ml_pipeline.py                  ← Data preprocessing pipeline
├── config.py                       ← Configuration settings
├── generate_sample_data.py         ← Create test dataset
├── requirements.txt                ← Python dependencies
└── README.md                       ← Quick start guide
```

---

## 🔧 Key Files Explained

### 1. **main.py** - The Brain (ChurnPredictor Class)

**What it does:** Handles ALL ML operations

```python
ChurnPredictor class:
├── __init__()              → Initialize model (Random Forest or Logistic Regression)
├── fit_transform()         → Prepare data (encoding, normalization)
├── train()                 → Train the model on data
├── evaluate()              → Calculate accuracy, ROC-AUC, confusion matrix
├── predict()               → Make predictions on new data
├── save()                  → Save trained model + encoders
└── load()                  → Load saved model for inference
```

**How it's used:**
```bash
python main.py --data data/customer_churn.csv --model "Random Forest"
```

**What happens:**
1. Loads CSV data
2. Cleans & prepares features
3. Encodes categorical variables (text → numbers)
4. Splits data: 80% training, 20% testing
5. Trains Random Forest model
6. Evaluates with test data
7. Saves complete artifact to `models/churn_model.pkl`

---

### 2. **predict.py** - Batch Predictions

**What it does:** Makes predictions for multiple customers at once from a CSV

**How it's used:**
```bash
python predict.py --input data/customer_churn.csv --output models/predictions.csv
```

**What happens:**
1. Loads saved model from `models/churn_model.pkl`
2. Reads input CSV with customer data
3. Applies same preprocessing/encoding used during training
4. Generates predictions for each row
5. Saves results: customer_data + prediction + probability
6. Output CSV example:
```csv
gender,tenure,MonthlyCharges,...,prediction,churn_probability
Female,24,70,...,No,0.35
Male,12,105,...,Yes,0.78
```

---

### 3. **streamlit_app.py** - Interactive Web UI

**What it does:** Web app for single customer prediction with nice visuals

**Features:**
- Input form in sidebar (sliders, dropdowns)
- Real-time prediction
- Probability visualization (bar chart)
- Risk classification (Low/Medium/High)
- Shows model accuracy & ROC-AUC

**How it's used:**
```bash
streamlit run streamlit_app.py
```

Opens in browser: `http://localhost:8501`

**Workflow:**
1. User selects customer details (sidebar)
2. Clicks "Predict Churn" button
3. Model predicts instantly
4. Shows:
   - Prediction (Yes/No)
   - Probability (e.g., 65%)
   - Risk Level (Low/Medium/High)
   - Bar chart visualization
   - Input payload

---

### 4. **ml_pipeline.py** - Data Preprocessing

**What it does:** Generic preprocessing pipeline (template for data cleaning)

**Includes:**
- Load data from CSV
- Handle missing values (imputation)
- Encode categorical variables (one-hot encoding)
- Scale features (StandardScaler)
- Split train/test data

**Note:** main.py has its own custom preprocessing (LabelEncoding instead of one-hot)

---

### 5. **config.py** - Configuration Settings

**What it does:** Centralized configuration for the project

**Contains:**
```python
- Data paths (training/testing data locations)
- Model choices (Logistic Regression, Random Forest)
- Feature definitions (which are categorical, which are numerical)
- Hyperparameters (model settings)
- Evaluation metrics to track
```

**Used for:** Easy configuration without editing code

---

### 6. **generate_sample_data.py** - Create Test Data

**What it does:** Generates realistic customer data for testing

**How it's used:**
```bash
python generate_sample_data.py --rows 2000 --output data/customer_churn.csv
```

**Creates:** 2000 synthetic customer records with realistic churn labels

**Why needed:** Privacy - don't commit real customer data to GitHub

---

## 🤖 Machine Learning Models Explained

### **Model 1: Random Forest Classifier** (PRIMARY)

**What it is:** Ensemble of decision trees

**How it works:**
```
1. Creates 300 decision trees
2. Each tree learns from random subset of data
3. Each tree makes a prediction
4. Final prediction = majority vote of all trees
5. More robust than single tree
```

**Decision Tree Example:**
```
        Is tenure > 12 months?
            /          \
          Yes           No
         /              \
    Low churn      Is contract month-to-month?
    (stay)              /              \
                       Yes            No
                      /                \
                High churn      Medium churn
                (churn)         (might churn)
```

**Advantages:**
- ✅ Handles both numerical and categorical data
- ✅ Captures non-linear relationships
- ✅ Robust to outliers
- ✅ Feature importance available
- ✅ Good for imbalanced data

**Disadvantages:**
- ❌ Slower prediction (300 trees to check)
- ❌ More memory usage
- ❌ Harder to explain to business

**Usage in main.py:**
```python
model = RandomForestClassifier(
    n_estimators=300,      # 300 decision trees
    random_state=42        # Reproducibility
)
model.fit(X_train, y_train)
```

---

### **Model 2: Logistic Regression** (ALTERNATIVE)

**What it is:** Linear probability model

**How it works:**
```
Input Features → Linear Combination → Sigmoid Function → Probability (0-1)

Example:
Score = 0.5*tenure - 0.3*MonthlyCharges + 0.2*ContractType + bias
Probability = 1 / (1 + e^(-Score))    ← Sigmoid function
```

**Sigmoid Function Visualization:**
```
Probability
    1.0 |           ___....___
    0.75|       ___/          \___
    0.5 |      /                  \
    0.25|    _/                     \_
    0.0 |___________________________
         -∞      0      Score      +∞
```

**Advantages:**
- ✅ Fast training & prediction
- ✅ Interpretable results (see feature importance)
- ✅ Lower computational cost
- ✅ Good baseline model
- ✅ Outputs probability directly

**Disadvantages:**
- ❌ Assumes linear relationship (may miss complex patterns)
- ❌ Requires feature scaling
- ❌ Struggles with high-dimensional data

**Usage in main.py:**
```python
model = LogisticRegression(
    max_iter=1000,         # Max iterations to converge
    random_state=42        # Reproducibility
)
model.fit(X_train, y_train)
```

---

## 🔄 Feature Engineering & Encoding

### **Problem:** Models need numbers, not text

### **Solution: Label Encoding**

Each categorical feature mapped to integers:

```python
gender: Male → 0, Female → 1
Partner: No → 0, Yes → 1
Contract: Month-to-month → 0, One year → 1, Two year → 2
InternetService: DSL → 0, Fiber → 1, No → 2
```

### **Target Encoding:**
```python
Churn: No → 0 (stayed), Yes → 1 (churned)
```

### **Unknown Category Handling:**
```python
During training: Saw values {DSL, Fiber, No}
During prediction: Get new value "Cable" → Map to '__unknown__'
This prevents crashes on new data
```

### **Encoding Flow in main.py:**
```
Raw Data                  Fit on Training Data      Fit Transform
┌──────────────────┐      ┌─────────────┐          ┌──────────────┐
│ gender: Female   │ ────→│ LabelEncoder│ ────────→│ gender: 1    │
│ tenure: 24       │      └─────────────┘          │ tenure: 24   │
│ Partner: Yes     │                               │ Partner: 1   │
│ Churn: No        │                               │ Churn: 0     │
└──────────────────┘                               └──────────────┘

Stored in:
self.feature_encoders['gender']       ← For prediction
self.feature_encoders['Partner']      ← For prediction
self.target_encoder                   ← For decoding result
```

---

## 📊 Data Splitting & Training

### **Train-Test Split:**
```
Original Data (2000 samples)
        ↓
    80% / 20%
    ↙        ↘
Training   Testing
1600       400
samples    samples
   ↓         ↓
[Train      [Only used for
model]      evaluation]
```

**Why 80/20?**
- Train on enough data for learning
- Test on held-out data to measure real performance
- 80/20 is industry standard

**Stratified Split:**
```python
# Ensures both sets have similar churn ratio
train_test_split(X, y, test_size=0.2, stratify=y)

Before split:  Churn ratio = 26% (520 churned, 1480 stayed)
After split:   Train ratio = 26%, Test ratio = 26% ✓
```

---

## 📈 Model Evaluation Metrics

After training, model is evaluated on test data:

### **1. Accuracy**
```
Definition: % of correct predictions
Formula: (TP + TN) / (TP + TN + FP + FN)

Example:
Predicted 385 customers correctly out of 400
Accuracy = 385/400 = 96.25%

Interpretation: 96% correct but NOT best metric for imbalanced data
```

### **2. ROC-AUC Score** (Best for imbalanced data)
```
Definition: Area Under Receiver Operating Characteristic Curve
Range: 0.0 to 1.0
- 1.0 = Perfect classifier
- 0.5 = Random guessing
- 0.0 = Totally wrong

Example: ROC-AUC = 0.92 = Model is 92% good at ranking churners

Why it's better:
- Doesn't just look at accuracy
- Considers true positives AND false positives
- Works well when churn% is low (imbalanced)
```

### **3. Confusion Matrix**
```
Shows 4 outcomes:

                Predicted
             Churn    Stay
Actual ┌─────────────────────┐
Churn  │  TP (77) │  FN (23)│  ← Actually churned
       ├──────────┼──────────┤
Stay   │  FP (12) │  TN(288)│  ← Actually stayed
       └─────────────────────┘

TP = True Positive (predicted churn, actually churned) ✓
TN = True Negative (predicted stay, actually stayed) ✓
FP = False Positive (predicted churn, actually stayed) ✗
FN = False Negative (predicted stay, actually churned) ✗ (worst)
```

### **4. Classification Report**
```
Includes per-class metrics:

              Precision  Recall  F1-Score
Churn (Yes)     0.87     0.77     0.82
Stay (No)       0.96     0.98     0.97

Precision: Of predicted churners, how many actually churned? (87%)
Recall: Of actual churners, how many did we catch? (77%)
F1-Score: Balance between Precision & Recall
```

---

## 🔮 Prediction Process (Inference)

### **Step-by-Step Flow:**

```
1. USER INPUT (Streamlit App)
   └─→ Gender: Female
   └─→ Tenure: 24 months
   └─→ Contract: month-to-month
   └─→ Monthly Charges: $70

2. LOAD MODEL & ENCODERS
   └─→ Load from models/churn_model.pkl
   └─→ Get feature encoders (gender, contract, etc.)
   └─→ Get target encoder (for final output)

3. ENCODE FEATURES (Transform)
   └─→ gender: Female → 1
   └─→ contract: month-to-month → 0
   └─→ All 19 features encoded

4. PREPARE FEATURE VECTOR
   └─→ [1, 24, 0, 70, ...] ← 19 features ready
   └─→ Ensure same order as training

5. MODEL PREDICTION (Random Forest)
   └─→ All 300 trees vote
   └─→ Average probability
   └─→ Result: 0.65 (65% churn probability)

6. DECODE & CLASSIFY
   └─→ If probability > 0.5 → Predict "Yes" (will churn)
   └─→ If probability >= 0.7 → "High Risk" (action needed!)
   └─→ If probability >= 0.4 → "Medium Risk" (monitor)
   └─→ Otherwise → "Low Risk" (stable)

7. OUTPUT
   ┌─────────────────────────────┐
   │ Prediction: Yes (will churn)│
   │ Probability: 65%            │
   │ Risk Level: Medium Risk     │
   │ Chart: ████ Churn 65%      │
   │        ██ Stay 35%         │
   └─────────────────────────────┘
```

---

## 🏗️ Complete Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                   CUSTOMER CHURN PREDICTION SYSTEM                  │
└─────────────────────────────────────────────────────────────────────┘

PHASE 1: DATA GENERATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
generate_sample_data.py
    ↓
2000 synthetic customers with demographics, services, churn labels
    ↓
data/customer_churn.csv

PHASE 2: MODEL TRAINING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
python main.py --data data/customer_churn.csv --model "Random Forest"
    ��
main.py (ChurnPredictor class)
    ├─ Load CSV
    ├─ fit_transform(): Prepare & encode features
    ├─ train(): Split data (80/20) and train Random Forest
    ├─ evaluate(): Test on held-out data
    └─ save(): Save artifact with model + encoders
    ↓
models/churn_model.pkl (290 KB artifact)
models/metrics.json (Accuracy: 96%, ROC-AUC: 0.92)

PHASE 3: BATCH PREDICTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
python predict.py --input data/customer_churn.csv --output predictions.csv
    ↓
predict.py
    ├─ Load trained model
    ├─ Read new customer data
    ├─ transform(): Apply same encoders
    ├─ predict(): Generate probability for each customer
    └─ Save results
    ↓
models/predictions.csv (with churn probabilities)

PHASE 4: INTERACTIVE PREDICTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
streamlit run streamlit_app.py
    ↓
Browser: http://localhost:8501
    ├─ Input form (sidebar)
    │  ├─ Gender selector
    │  ├─ Tenure slider
    │  ├─ Contract type selector
    │  ├─ Monthly charges slider
    │  └─ ... 19 total features
    │
    ├─ "Predict Churn" button
    │
    └─ Results display
       ├─ Prediction: Yes/No
       ├─ Probability: 65%
       ├─ Risk Level: Medium Risk
       └─ Bar chart visualization
```

---

## 💾 Model Artifact - What Gets Saved?

When you train a model, everything needed for prediction is saved:

```python
artifact = {
    'model': RandomForestClassifier(...),           ← Trained model object
    'model_type': 'Random Forest',                  ← Which model
    'feature_columns': [                            ← Feature order
        'gender', 'SeniorCitizen', 'Partner',
        'tenure', 'MonthlyCharges', ...
    ],
    'categorical_columns': [                        ← Which ones to encode
        'gender', 'Partner', 'InternetService', ...
    ],
    'feature_encoders': {                           ← Encoders for encoding
        'gender': LabelEncoder(...),
        'Partner': LabelEncoder(...),
        'InternetService': LabelEncoder(...),
        ...
    },
    'target_encoder': LabelEncoder(...),            ← For decoding prediction
    'random_state': 42                              ← Reproducibility
}

Saved as: models/churn_model.pkl (290 KB binary file using joblib)
```

**Why save all this?**
- Model: Makes predictions
- Encoders: Transform text to numbers (same way as training)
- Feature order: Ensure correct feature mapping
- Random state: Reproducible results

---

## 🎯 Risk Classification Thresholds

Based on churn probability:

```python
HIGH_RISK_THRESHOLD = 0.7    (70%)
MEDIUM_RISK_THRESHOLD = 0.4  (40%)

Probability           Risk Level       Action
  < 40%         →   Low Risk      → No action needed
40% - 70%       →   Medium Risk   → Monitor, prepare retention
  > 70%         →   High Risk     → Immediate action! Offer incentive
```

**Example scenarios:**
```
Customer A: 35% churn probability → Low Risk → Stable customer
Customer B: 55% churn probability → Medium Risk → Might need discount
Customer C: 78% churn probability → High Risk → Call with special offer!
```

---

## 🚀 Complete Usage Example

```bash
# Step 1: Setup environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Step 2: Create sample data
python generate_sample_data.py --rows 2000 --output data/customer_churn.csv

# Step 3: Train model (creates models/churn_model.pkl)
python main.py --data data/customer_churn.csv --model "Random Forest"

# Step 4: Batch predict (creates models/predictions.csv)
python predict.py --input data/customer_churn.csv --output models/predictions.csv

# Step 5: Launch web app
streamlit run streamlit_app.py
# Open: http://localhost:8501
# Use sidebar to input customer details
# Click "Predict Churn" to see prediction
```

---

## 🔧 Hyperparameters Explained

### **Random Forest:**
```python
n_estimators=300        ← Number of trees (more = better but slower)
random_state=42         ← Seed for reproducibility
max_depth=None          ← Max tree depth (deeper = overfit risk)
min_samples_split=2     ← Min samples to split a node
```

### **Logistic Regression:**
```python
max_iter=1000           ← Max iterations to converge
random_state=42         ← Seed for reproducibility
solver='lbfgs'          ← Optimization algorithm
C=1.0                   ← Inverse regularization strength
```

---

## 📚 Key Concepts Summary

| Concept | Definition |
|---------|-----------|
| **Churn** | Customer leaving/stopping service |
| **Label Encoding** | Convert categories to integers |
| **Training Data** | 1600 customers used to teach model |
| **Test Data** | 400 customers used to verify model |
| **Accuracy** | % correct predictions |
| **ROC-AUC** | Model's ability to rank churners correctly |
| **True Positive** | Correctly predicted churner |
| **False Negative** | Missed a churner (worst case) |
| **Probability** | Confidence in prediction (0-1) |
| **Feature** | Input variable (tenure, charges, etc.) |
| **Model Artifact** | Saved model + encoders |

---

## ✅ Success Metrics

Current model performance:
- **Accuracy:** 96.25% (correct predictions)
- **ROC-AUC:** 0.92 (excellent ranking ability)
- **Recall (Churn):** 77% (catch 77% of churners)
- **Precision (Churn):** 87% (87% of predicted churners actually churn)

---

## 🎓 Learning Path

**Beginner:** Understand prediction basics
1. Run `python generate_sample_data.py` to create data
2. Run `python main.py` to train model
3. Run `streamlit run streamlit_app.py` and test predictions

**Intermediate:** Understand models & metrics
1. Read about Random Forest vs Logistic Regression
2. Understand confusion matrix and evaluation metrics
3. Try different hyperparameters in main.py

**Advanced:** Improve model
1. Add new features
2. Try different model architectures
3. Implement cross-validation
4. Deploy to production

