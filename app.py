import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, and expected columns
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = [
    'age',
    'is_female',
    'bmi',
    'children',
    'is_smoker',
    'region_northwest',
    'region_southeast',
    'region_southwest',
    'bmi_category_Obese',
    'bmi_category_Overweight',
    'bmi_category_Underweight'
]

st.set_page_config(page_title="Insurance Charges Predictor", layout="centered")

st.title("💰 Insurance Charges Prediction App")

# Sidebar Inputs
with st.sidebar:
    st.header("📋 Input Patient Details")
    age = st.slider("Age", 18, 100, 30)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.slider("BMI", 10.0, 50.0, 25.0)
    children = st.slider("Number of Children", 0, 5, 0)
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# --- Feature Engineering ---
def preprocess(age, sex, bmi, children, smoker, region):
    data = {}
    data["age"] = age
    data["is_female"] = 1 if sex == "female" else 0
    data["bmi"] = bmi
    data["children"] = children
    data["is_smoker"] = 1 if smoker == "yes" else 0

    # One-hot for region
    for reg in ["region_northwest", "region_southeast", "region_southwest"]:
        data[reg] = 1 if region == reg.split("_")[1] else 0

    # BMI Category
    data["bmi_category_Underweight"] = 1 if bmi < 18.5 else 0
    data["bmi_category_Overweight"] = 1 if 25 <= bmi < 30 else 0
    data["bmi_category_Obese"] = 1 if bmi >= 30 else 0

    return pd.DataFrame([data])

# Predict on Button Click
if st.button("🔍 Predict Charges"):
    try:
        # Preprocess inputs
        input_df = preprocess(age, sex, bmi, children, smoker, region)

        # Align with expected columns
        input_df = input_df.reindex(columns=expected_columns, fill_value=0)

        # Transform using trained scaler
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0]
        st.subheader("✅ Predicted Insurance Charges:")
        st.success(f"${prediction:,.2f}")

        # Debug info (optional)
        st.write("🔎 **Processed Input:**", input_df)

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
