# California Housing Price Prediction Pipeline

An end-to-end Machine Learning pipeline built with Scikit-Learn to estimate median housing prices across California districts.

---

## 📌 Project Overview
This project processes block-group level housing data from the California Housing dataset to forecast median home values. The pipeline implements stratified sampling, automated preprocessing (imputation, custom ratio attribute combination, standard scaling), hyperparameter tuning via `GridSearchCV`, and model serialization using `joblib`.

---

## 📊 Model Evaluation & Baseline Comparison

| Model | Evaluation Strategy | Test / CV RMSE | Key Findings |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | 10-Fold Cross-Validation | ~$68,600 | High underfitting baseline. |
| **Decision Tree** | 10-Fold Cross-Validation | ~$70,500 | Severe overfitting on training data. |
| **Random Forest (Tuned)** | Single Pass Test Evaluation | **$47,535.59** | **Final Model**: Reduced error by over $21,000 against baseline. |

---

## 🛡️ Production Strategy & Prediction Intervals

Machine learning models output statistical averages rather than exact prices. To account for the test error margin of **~$47,535**, predictions are served using **95% Prediction Intervals**:

$$\text{Margin of Error} = 1.96 \times \text{RMSE} \approx \$93,170$$

* **Sample Point Estimate:** $\$280,000$
* **Served Valuation Range:** $\$186,830 - \$373,170$

This interval provides end-users with a realistic price window based on statistical confidence.

---

## 📁 Repository Structure

```text
.
├── MY AI MODEL.ipynb               # Primary notebook (EDA, pipeline, tuning, evaluation)
├── california_housing_model.pkl    # Serialized pre-trained Random Forest model
├── full_pipeline.pkl               # Serialized Scikit-Learn preprocessing pipeline
└── README.md                       # Project documentation
