import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(
    page_title="AutoLux | Car Price AI",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #07111f 0%, #101d35 55%, #102b3b 100%);
    color: #edf6ff;
}
[data-testid="stSidebar"] {
    background: #0b1425;
    border-right: 1px solid #263852;
}
h1, h2, h3, p, label {
    color: #edf6ff !important;
}
.hero {
    padding: 28px;
    border-radius: 22px;
    background: linear-gradient(120deg, #123e59, #172554, #312e81);
    border: 1px solid #3b82a0;
    margin-bottom: 25px;
}
.hero h1 {
    font-size: 38px;
    margin-bottom: 5px;
}
.hero p {
    color: #c6e7f5 !important;
    font-size: 16px;
}
.metric-card {
    background: linear-gradient(145deg, #14253d, #101a2b);
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #2d4967;
    min-height: 135px;
}
.metric-label {
    color: #9db5d1;
    font-size: 14px;
}
.metric-value {
    font-size: 27px;
    font-weight: 800;
    color: #67e8f9;
    margin-top: 8px;
}
.section-card {
    background: rgba(17, 31, 51, 0.85);
    padding: 22px;
    border: 1px solid #2b415e;
    border-radius: 18px;
    margin-bottom: 18px;
}
div.stButton > button {
    width: 100%;
    border: 0;
    border-radius: 12px;
    padding: 13px;
    color: #07111f;
    font-weight: 800;
    background: linear-gradient(90deg, #67e8f9, #60a5fa);
    transition: 0.2s;
}
div.stButton > button:hover {
    background: linear-gradient(90deg, #a5f3fc, #93c5fd);
    transform: translateY(-2px);
}
div[data-testid="stMetric"] {
    background: #14253d;
    border: 1px solid #2d4967;
    padding: 16px;
    border-radius: 14px;
}
hr {
    border-color: #2b415e;
}
</style>
""", unsafe_allow_html=True)
# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🚘 AutoLux AI")
    st.caption("USED CAR INTELLIGENCE")
    st.markdown("---")

    st.markdown("### 🧭 Navigation")
    st.write("🏠 Price Predictor")
    st.write("📊 Dataset Overview")
    st.write("🤖 Model Insights")

    st.markdown("---")
    st.markdown("### ⚡ Engine")
    st.success("Random Forest + Linear Regression")
    st.caption("Predictions are estimates, not guaranteed selling prices.")

#load saved objects
encoder=joblib.load("encoder.pkl")
liner_model=joblib.load("liner_model.pkl")
rf_model=joblib.load("random_forest_model.pkl")

st.title("Used Car Price Prediction")
st.write("Enter the car details to estimate its price")

st.header("Car Details")

year=st.number_input(
    "manufacturing year",
    min_value=1990,
    max_value=2026,
    value=2020
)


brand=st.text_input("brand","Honda")
model_name=st.text_input("model","city")

distance=st.number_input(
    "Distance Travelled (km)",
    min_value=0,
    value=30000
)

fuel_type=st.selectbox(
    "Fule type",
    ["Petrol","Diesel","CNG","Electric","LPG"]
)

city=st.text_input("City","coimbatore")

st.write("your inputs are ready")



from scipy.sparse import hstack

if st.button("Predict Car Price"):

    input_cat = pd.DataFrame([{
        "brand": brand,
        "model_name": model_name,
        "fuel_type": fuel_type,
        "city": city
    }])

    # Step 1: Encode the categorical inputs
    encoded_input = encoder.transform(input_cat)

    # Step 2: Prepare numerical inputs
    input_num = [[year, distance]]

    # Step 3: Combine numerical and encoded inputs
    final_input = hstack([input_num, encoded_input])

    # Step 4: Predict
    linear_price = liner_model.predict(final_input)[0]
    rf_price = rf_model.predict(final_input)[0]

    st.subheader("Predicted Used Car Price")
    st.metric("Random Forest Prediction", f"₹{rf_price:,.0f}")
    st.metric("Linear Regression Prediction", f"₹{linear_price:,.0f}")
