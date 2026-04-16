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
3. **Docker** - containerize the API so the app runs consistently anywhere.
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
3. Run the FastAPI backend:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080
```
4. Run the Streamlit UI in a separate terminal:
```bash
streamlit run streamlit_app.py
```

## Docker
Build the container locally:
```bash
docker build -t churn-predictor .
```
Run the container:
```bash
docker run -p 8080:8080 churn-predictor
```
The API will be available at `http://localhost:8080`.

## Deployment
For deployment on Render or another container host, use the provided `Dockerfile`. The container exposes port `8080` and runs the API by default.

## Notes
- The FastAPI endpoint accepts business-friendly string values for `Gender`, `Subscription_Type`, and `Contract_Length`.
- The Streamlit app can be used by non-technical stakeholders to enter customer details and see churn probability plus SHAP feature explanations.
- The model was trained with label encoding for categorical data, so the API and UI use the same mappings.
