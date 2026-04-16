import streamlit as st
import requests

# Page config
st.set_page_config(
    page_title="Cardiac Shield AI",
    page_icon="🫀",
    layout="wide"
)

# -----------------------------
# Simple Header
# -----------------------------
st.title("🫀 Cardiac Shield AI")
st.subheader("Heart Attack Risk Prediction")

# -----------------------------
# Input Fields
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", [0,1,2,3])
    trestbps = st.number_input("Resting BP", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)

with col2:
    fbs = st.selectbox("Fasting Blood Sugar >120", [0,1])
    restecg = st.selectbox("Rest ECG", [0,1,2])
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Angina", [0,1])
    oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)

slope = st.selectbox("Slope", [0,1,2])
ca = st.selectbox("CA", [0,1,2,3])
thal = st.selectbox("Thal", [1,2,3])

# -----------------------------
# Convert values
# -----------------------------
sex = 1 if sex == "Male" else 0

features = [
    age, sex, cp, trestbps, chol,
    fbs, restecg, thalach,
    exang, oldpeak, slope, ca, thal
]

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🔍 Predict"):

    try:
        with st.spinner("Analyzing..."):

            url = "https://heart-api-s9cu.onrender.com/predict"

            response = requests.post(
                url,
                json={"data": features},
                timeout=10
            )

            response.raise_for_status()
            result = response.json()

            # Extract values
            risk = result.get("risk", "Unknown")
            probability = result.get("probability", 0.0)
            advice = result.get("advice", "No advice")

            prob_pct = probability * 100 if probability <= 1 else probability

            # -----------------------------
            # Display Result
            # -----------------------------
            st.success("Prediction Complete")

            if "high" in risk.lower():
                st.error(f"🚨 Risk Level: {risk}")
            elif "medium" in risk.lower():
                st.warning(f"⚠️ Risk Level: {risk}")
            else:
                st.success(f"✅ Risk Level: {risk}")

            st.write(f"**Probability:** {prob_pct:.2f}%")
            st.write(f"**Advice:** {advice}")

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API")

    except requests.exceptions.Timeout:
        st.error("Request timeout")

    except requests.exceptions.HTTPError as e:
        st.error(f"Server error: {e}")

    except Exception as e:
        st.error(f"Error: {e}")
