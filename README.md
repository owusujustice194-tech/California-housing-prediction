# California Housing Price Prediction Model 🏡

An end-to-end Machine Learning pipeline built with `scikit-learn` to predict median house values in California block groups. 

The project covers automated data ingestion, stratified sampling, pipeline-based feature engineering, cross-validation model selection, hyperparameter optimization via `GridSearchCV`, and model deployment using `joblib`.

---

## 📊 Model Performance & Benchmarks

Multiple regression architectures were evaluated using 10-fold cross-validation before selecting and fine-tuning the final Random Forest model.

| Model / Stage | 10-Fold CV Mean RMSE ($) | Final Test RMSE ($) | Key Notes |
| :--- | :--- | :--- | :--- |
| **Linear / Single Tree Baselines** | ~$69,392 – $72,072[cite: 12] | — | High error due to severe underfitting / overfitting[cite: 12] |
| **Random Forest (Default)** | ~$50,636[cite: 12] | — | Significant performance jump over simple baselines[cite: 12] |
| **Tuned Random Forest (`GridSearchCV`)** | **~$49,421**[cite: 13] | **$47,312.93**[cite: 15] | Optimal parameters: `max_features=6`, `n_estimators=30`[cite: 13] |

### 🎯 Statistical Evaluation
* **Final Test RMSE:** **$47,312.93**[cite: 15]
* **95% Confidence Interval for Generalization Error:** **[$45,350.26, $49,197.36]**[cite: 16]

---

## 🔑 Top Feature Importances

Using `grid_search.best_estimator_.feature_importances_`, the key drivers of property value predictions were identified:

1. **`median_income`**: 34.31%[cite: 14]
2. **`INLAND` (Ocean Proximity)**: 15.75%[cite: 14]
3. **`population_per_household`**: 10.37%[cite: 14]
4. **`bedrooms_per_room`**: 8.33%[cite: 14]
5. **`longitude`**: 7.88%[cite: 14]
6. **`latitude`**: 7.30%[cite: 14]

---

## 🛠️ End-to-End Workflow

1. **Data Acquisition**: Automated downloading and extracting of the California housing dataset using `urllib.request` and `tarfile`.
2. **Stratified Sampling**: Binned `median_income` to perform `StratifiedShuffleSplit`, preventing sampling bias between train and test sets.
3. **Data Transformation Pipeline**:
   * Imputed missing values with `SimpleImputer(strategy="median")`.
   * Created custom combined attributes (`rooms_per_household`, `bedrooms_per_room`, `population_per_household`).
   * Scaled numerical features via `StandardScaler`.
   * One-hot encoded categorical variables (`ocean_proximity`) via `OneHotEncoder`.
4. **Hyperparameter Tuning**: Ran `GridSearchCV` over multi-parameter trees to optimize model generalization[cite: 13].
5. **Model Export**: Saved the fitted preprocessing pipeline and best estimator using `joblib`[cite: 15].

---

## 📁 Repository Structure

```text
California-housing-prediction/
├── MY AI MODEL.ipynb             # Full Jupyter Notebook implementation
├── california_housing_model.pkl  # Trained Random Forest model
├── full_pipeline.pkl             # Fitted preprocessing pipeline
└── README.md                     # Project documentation
