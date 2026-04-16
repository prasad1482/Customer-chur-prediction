import os
import streamlit as st
import requests
import pandas as pd
from datetime import datetime

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

# Page config
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Styling
st.markdown(
    """
    <style>
    .main { max-width: 800px; }
    .stMetric { text-align: center; }
    .high-risk { color: #d32f2f; font-weight: bold; }
    .medium-risk { color: #f57c00; font-weight: bold; }
    .low-risk { color: #388e3c; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 Customer Churn Predictor")
st.markdown("---")

# API endpoint
API_URL = "http://127.0.0.1:8000/predict"

# Create form in columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    gender = st.selectbox("Gender", ["Female", "Male"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=18)
    support_calls = st.number_input("Support Calls", min_value=0, max_value=10, value=3)

with col2:
    payment_delay = st.number_input("Payment Delay (days)", min_value=0, max_value=60, value=10)
    subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
    contract_length = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
    total_spend = st.number_input("Total Spend ($)", min_value=0, max_value=5000, value=645)

col3, col4 = st.columns(2)
with col3:
    last_interaction = st.number_input("Last Interaction (days)", min_value=0, max_value=365, value=12)
with col4:
    usage_frequency = st.number_input("Usage Frequency (per week)", min_value=0, max_value=20, value=8)

# Predict button
if st.button("🔮 Predict Churn Risk", use_container_width=True, type="primary"):
    try:
        # Prepare payload
        payload = {
            "Age": float(age),
            "Gender": gender,
            "Tenure": float(tenure),
            "Support_Calls": float(support_calls),
            "Payment_Delay": float(payment_delay),
            "Subscription_Type": subscription_type,
            "Contract_Length": contract_length,
            "Total_Spend": float(total_spend),
            "Last_Interaction": float(last_interaction),
            "Usage_Frequency": float(usage_frequency),
        }

        # Call FastAPI
        response = requests.post(API_URL, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            
            st.markdown("---")
            st.markdown("### 📈 Prediction Results")
            
            # Display risk level with color
            risk_level = result["risk_level"]
            churn_prob = result["churn_probability"]
            will_churn = result["will_churn"]
            
            # Color-coded risk display
            if risk_level == "High":
                st.error(f"🔴 **Risk Level: HIGH** — {churn_prob*100:.1f}% churn probability")
            elif risk_level == "Medium":
                st.warning(f"🟡 **Risk Level: MEDIUM** — {churn_prob*100:.1f}% churn probability")
            else:
                st.success(f"🟢 **Risk Level: LOW** — {churn_prob*100:.1f}% churn probability")
            
            st.markdown("---")
            
            # Display metrics
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("Churn Probability", f"{churn_prob*100:.1f}%")
            with col_m2:
                st.metric("Will Churn?", "Yes ⚠️" if will_churn else "No ✅")
            with col_m3:
                st.metric("Risk Category", risk_level)
            
            st.markdown("---")
            
            # Display top reasons
            st.markdown("### 💡 Why?")
            reasons = result["top_reasons"]
            for i, reason in enumerate(reasons, 1):
                st.info(f"**{i}. {reason}**")
            
            st.markdown("---")
            st.markdown(f"*Prediction made on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}*")
        
        else:
            st.error(f"API Error: {response.status_code}")
            st.write(response.text)
    
    except requests.exceptions.ConnectionError:
        st.error("❌ **Cannot connect to API**. Make sure FastAPI is running on http://127.0.0.1:8000")
        st.info("Run this in terminal: `uvicorn app.main:app --reload`")
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Sidebar info
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown(
        """
        This app predicts **customer churn risk** using a Random Forest model trained on customer behavior data.
        
        **Input Parameters:**
        - **Age**: Customer's age
        - **Gender**: Customer's gender
        - **Tenure**: Months as customer
        - **Support Calls**: Times contacted support
        - **Payment Delay**: Days late on payments
        - **Subscription**: Plan type
        - **Contract**: Contract length
        - **Total Spend**: Total money spent
        - **Last Interaction**: Days since last contact
        - **Usage Frequency**: Weekly usage
        
        **Risk Levels:**
        - 🟢 **Low**: < 40% probability
        - 🟡 **Medium**: 40-70% probability
        - 🔴 **High**: > 70% probability
        """
    )
    
    st.markdown("---")
    st.markdown("🚀 **Built with FastAPI + Streamlit**")
