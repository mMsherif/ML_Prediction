import streamlit as st
import numpy as np
from joblib import load

# =========================
# Load Models & Scaler
# =========================
scaler = load("scaler.pkl")

# Regression models
ridge = load("ridge_final_score.pkl")
lasso = load("lasso_final_score.pkl")
gradient = load("gbr_final_score.pkl")

# Classification models
knn = load("knn_pass_fail.pkl")
svm = load("svm_pass_fail.pkl")

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# =========================
# CSS Styling
# =========================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #eef2ff, #f8fafc);
}
.main {
    background-color: white;
    padding: 2.5rem;
    border-radius: 18px;
}
h1, h2, h3 {
    text-align: center;
    color: #1f2937;
}
.card {
    background: white;
    padding: 1.5rem;
    border-radius: 16px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.08);
    margin-bottom: 1.5rem;
}
.metric {
    font-size: 28px;
    font-weight: bold;
    text-align: center;
}
.pass {
    color: #16a34a;
    font-size: 26px;
    font-weight: bold;
    text-align: center;
}
.fail {
    color: #dc2626;
    font-size: 26px;
    font-weight: bold;
    text-align: center;
}
.small {
    text-align: center;
    color: #6b7280;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Title
# =========================
st.markdown("<h1>🎓 Student Performance Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p class='small'>Predict final score & pass/fail status using ML models</p>", unsafe_allow_html=True)

# =========================
# Inputs
# =========================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📝 Student Academic Inputs")

col1, col2 = st.columns(2)
with col1:
    midterm = st.slider("Midterm Score", 0, 100, 70)
    quiz_avg = st.slider("Average Quiz Score", 0, 100, 65)
with col2:
    assignment_rate = st.slider("Assignment Submission Rate (%)", 0, 100, 80)
    attendance = st.slider("Lecture Attendance Rate (%)", 0, 100, 85)

st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Model Selection
# =========================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("⚙️ Model Selection")

col3, col4 = st.columns(2)
with col3:
    reg_model_name = st.selectbox(
        "Final Score Model",
        ["Gradient Boosting", "Ridge", "Lasso"]
    )
with col4:
    clf_model_name = st.selectbox(
        "Pass / Fail Model",
        ["SVM", "KNN"]
    )

st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Prediction Button
# =========================
if st.button("🚀 Predict Performance", use_container_width=True):

    X = np.array([[midterm, quiz_avg, assignment_rate, attendance]])
    X_scaled = scaler.transform(X)

    # -------- Regression --------
    reg_model = {
        "Gradient Boosting": gradient,
        "Ridge": ridge,
        "Lasso": lasso
    }[reg_model_name]
    final_score = reg_model.predict(X_scaled)[0]

    # -------- Classification --------
    if clf_model_name == "SVM":
        prob = svm.predict_proba(X_scaled)[0][1]
        pass_fail = "Pass" if prob >= 0.65 else "Fail"
    else:  # KNN
        prob = knn.predict_proba(X_scaled)[0][1]
        pass_fail = "Pass" if prob >= 0.5 else "Fail"

    # =========================
    # Results
    # =========================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Prediction Results")

    st.markdown(
        f"<div class='metric'>Final Score: {final_score:.1f} / 100</div>",
        unsafe_allow_html=True
    )
    st.progress(min(int(final_score), 100))
    st.write("")
    st.markdown(f"<div class='small'>Pass Probability: {prob:.2%}</div>", unsafe_allow_html=True)

    if pass_fail == "Pass":
        st.markdown("<p class='pass'>✅ PASS</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='fail'>❌ FAIL</p>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
