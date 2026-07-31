# Disease Prediction from Medical Data
**CodeAlpha ML Internship — Task 4**

Predicts the risk of three diseases — Heart Disease, Diabetes, and Breast Cancer — using classical ML models and a soft-voting ensemble.

---

## What This Project Does

- Loads three medical datasets (Heart Disease, Diabetes, Breast Cancer)
- Applies feature engineering specific to each disease
- Trains 4 models per disease: SVM, Logistic Regression, Random Forest, Gradient Boosting
- Combines the best three into a **Soft Voting Ensemble** for better accuracy
- Evaluates with Accuracy, F1-Score, ROC-AUC, Confusion Matrix
- Generates a full visual dashboard and saves all trained models
- Includes a **patient risk inference function** (Low / Moderate / High risk)

---

## Project Structure

```
disease_prediction/
├── disease_prediction_complete.py   ← Main script (run this)
├── requirements.txt
└── README.md
```

After running, these are generated:
```
├── disease_prediction_dashboard.png  ← Full visual dashboard
└── disease_models.pkl                ← All saved ensemble models
```

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the project
python disease_prediction_complete.py
```

All three datasets are either auto-downloaded (Breast Cancer via sklearn) or synthetically generated to match UCI distributions — no manual downloads needed.

---

## Datasets

| Dataset | Samples | Features | Source |
|---|---|---|---|
| Heart Disease | 303 | 13 + 4 engineered | Cleveland UCI (synthetic replica) |
| Diabetes | 768 | 8 + 4 engineered | Pima Indians UCI (synthetic replica) |
| Breast Cancer | 569 | 30 | sklearn built-in (real UCI data) |

---

## Models Per Disease

| Model | Type |
|---|---|
| SVM | Support Vector Machine (RBF kernel) |
| Logistic Regression | Linear classifier |
| Random Forest | 200-tree ensemble |
| Gradient Boosting | 120-estimator boosting |
| ⭐ Ensemble | Soft Voting (LR + RF + GB) |

---

## Feature Engineering

**Heart Disease** — 4 extra features:
- `age_risk` — flag for age > 55
- `hr_reserve` — heart rate reserve (220 − age − max HR)
- `bp_category` — blood pressure category (normal/elevated/stage1/stage2)
- `chol_risk` — high cholesterol flag (> 240 mg/dL)

**Diabetes** — 4 extra features:
- `glucose_cat` — glucose category (normal/prediabetic/diabetic)
- `bmi_category` — BMI category (underweight/normal/overweight/obese)
- `age_over_40` — age risk flag
- `insulin_resist` — elevated insulin flag (> 200)

**Breast Cancer** — uses all 30 original features (already rich).

---

## Dashboard Output

The generated `disease_prediction_dashboard.png` contains (per disease):
1. ROC Curves — all 5 models overlaid with AUC scores
2. AUC comparison bar chart
3. Ensemble confusion matrix
4. Summary table — best model per disease with all metrics

---

## Patient Risk Inference

After training, you can predict risk for a new patient:

```python
import joblib
from disease_prediction_complete import predict_patient_risk

models = joblib.load("disease_models.pkl")
scaler, model, feat_cols = models["Heart Disease"]

patient = {
    "age": 58, "sex": 1, "cp": 0, "trestbps": 145,
    "chol": 270, "fbs": 1, "restecg": 1, "thalach": 130,
    "exang": 1, "oldpeak": 2.8, "slope": 1, "ca": 2, "thal": 3,
    "age_risk": 1, "hr_reserve": 32, "bp_category": 2, "chol_risk": 1,
}

prob = predict_patient_risk(scaler, model, feat_cols, patient, "Heart Disease")
```

Output:
```
Patient Assessment — Heart Disease
Risk Level         : 🔴 HIGH RISK
Disease Probability: 78.43%
```

---

## GitHub Repository Name
`CodeAlpha_DiseasePrediction`
