# Customer Churn Prediction

## Project Overview
This project predicts customer churn using machine learning models trained on a telecommunications dataset. The goal is to identify customers who are likely to churn so that business teams can take action before they leave.

## Files
- `1.ipynb` - initial exploratory data analysis, preprocessing, encoding, scaling, and basic model training.
- `2.ipynb` - refined end-to-end pipeline with train/test split, model search, model evaluation, SHAP explainability, and persistence.
- `Dataset/` - contains the training and testing CSV files.
- `models/` - stores serialized model and scaler artifacts.

## Dataset
The dataset includes customer attributes such as:
- `Gender`
- `Subscription Type`
- `Contract Length`
- `Age`
- `Tenure`
- `Usage Frequency`
- `Support Calls`
- `Payment Delay`
- `Total Spend`
- `Last Interaction`
- `Churn`

The pipeline currently drops missing rows, encodes categorical variables, and uses stratified train/test splitting.

## Modeling
The project experiments with multiple models:
- Logistic Regression
- Random Forest with `RandomizedSearchCV`
- XGBoost with class imbalance handling

The current best model is selected via `RandomizedSearchCV`, and `roc_auc` is used as the main scoring metric.

## SHAP Explainability
SHAP is integrated to explain predictions from the best model. It provides:
- global feature importance
- local explanations for individual customers

This makes the model actionable by showing which features drive churn risk.

## Saved Artifacts
The notebook saves the following artifacts under `models/`:
- `churn_model.pkl` - trained model object
- `scaler.pkl` - fitted `StandardScaler`

## Next Steps
The project roadmap includes:
1. **FastAPI** - expose the saved model as a REST API for integration.
2. **Streamlit** - build a user-friendly interface for business users to enter customer details and see churn predictions with explanations.
3. **Docker** - containerize the API and UI so the app runs consistently anywhere.
4. **Deployment** - deploy the app to a hosting provider such as Render for a live demo.

## How to Run
1. Activate the virtual environment:
```bash
cd "C:/Users/hp/Desktop/ntg/Customer chur prediction"
.\venv\Scripts\Activate.ps1
```
2. Install dependencies if needed:
```bash
pip install -r requirements.txt
```
3. Open and run `2.ipynb` to retrain models and generate SHAP plots.

## Notes
- Ensure the notebook kernel uses the same Python environment where packages like `shap`, `xgboost`, and `scikit-learn` are installed.
- The current accuracy is modest, so future improvement should focus on feature engineering, hyperparameter tuning, and model explainability.
