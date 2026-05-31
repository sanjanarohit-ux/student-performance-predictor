import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg,#0f172a,#1e293b);
}

.big-title {
    text-align:center;
    font-size:45px;
    font-weight:bold;
    color:white;
}

.sub-title {
    text-align:center;
    font-size:18px;
    color:#cbd5e1;
}

.stButton>button {
    width:100%;
    background: linear-gradient(90deg,#3b82f6,#8b5cf6);
    color:white;
    border:none;
    border-radius:12px;
    height:50px;
    font-size:18px;
    font-weight:bold;
}

.result-box {
    padding:20px;
    border-radius:15px;
    text-align:center;
    background:#1e293b;
    color:white;
    font-size:25px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL
# ==========================================
model = joblib.load("knn_model.pkl")

# ==========================================
# HEADER
# ==========================================
st.markdown('<p class="big-title">🎓 Student Performance Predictor</p>',
            unsafe_allow_html=True)

st.markdown(
    '<p class="sub-title">Predict Student Academic Performance using Machine Learning</p>',
    unsafe_allow_html=True)

st.divider()

# ==========================================
# INPUTS
# ==========================================
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["M", "F"])
    raisedhands = st.slider("Raised Hands", 0, 100, 50)
    visited = st.slider("Visited Resources", 0, 100, 50)
    announcements = st.slider("Announcements View", 0, 100, 50)
    discussion = st.slider("Discussion Participation", 0, 100, 50)

with col2:
    absence = st.selectbox(
        "Student Absence Days",
        ["Under-7", "Above-7"]
    )

    parent_survey = st.selectbox(
        "Parent Answering Survey",
        ["Yes", "No"]
    )

    relation = st.selectbox(
        "Parent Relation",
        ["Father", "Mum"]
    )

# ==========================================
# CREATE FEATURE VECTOR
# ==========================================
if st.button("🔍 Predict Performance"):

    features = np.zeros((1,72))

    # numerical features
    features[0][0] = raisedhands
    features[0][1] = visited
    features[0][2] = announcements
    features[0][3] = discussion

    try:
        prediction = model.predict(features)[0]

        label_map = {
            "L": "🔴 Low Performance",
            "M": "🟡 Medium Performance",
            "H": "🟢 High Performance"
        }

        result = label_map.get(prediction, prediction)

        st.markdown(
            f"""
            <div class='result-box'>
            Predicted Performance <br><br>
            {result}
            </div>
            """,
            unsafe_allow_html=True
        )

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0]

            st.subheader("Prediction Confidence")

            conf_df = pd.DataFrame({
                "Class":["Low","Medium","High"],
                "Probability": probs
            })

            st.bar_chart(
                conf_df.set_index("Class")
            )

    except Exception as e:
        st.error(f"Prediction Error: {e}")

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit & Machine Learning")