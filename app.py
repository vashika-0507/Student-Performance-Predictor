"""
Student Performance Predictor - web app
Enter a student's details and the trained Linear Regression model predicts their math score.
Run with:  python -m streamlit run app.py
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")


@st.cache_resource
def load_model():
    # model trained in student_performance.py (background factors + reading and writing scores)
    return joblib.load(Path(__file__).parent / "model_full.pkl")


model = load_model()

st.title("🎓 Student Performance Predictor")
st.write(
    "Predict a student's **math score** from their background and their reading and "
    "writing scores. The model is a Linear Regression trained on the Kaggle "
    "*Students Performance in Exams* dataset (R² = 0.88)."
)

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["female", "male"])
    race = st.selectbox("Race/ethnicity group", ["group A", "group B", "group C", "group D", "group E"])
    parent_edu = st.selectbox(
        "Parental level of education",
        ["some high school", "high school", "some college",
         "associate's degree", "bachelor's degree", "master's degree"],
    )
    lunch = st.selectbox("Lunch type", ["standard", "free/reduced"])

with col2:
    prep = st.selectbox("Test preparation course", ["none", "completed"])
    reading = st.slider("Reading score", 0, 100, 70)
    writing = st.slider("Writing score", 0, 100, 70)

if st.button("Predict math score", type="primary"):
    # column names must match the ones used when the model was trained
    student = pd.DataFrame([{
        "gender": gender,
        "race_ethnicity": race,
        "parental_level_of_education": parent_edu,
        "lunch": lunch,
        "test_preparation_course": prep,
        "reading_score": reading,
        "writing_score": writing,
    }])
    prediction = float(model.predict(student)[0])
    prediction = max(0.0, min(100.0, prediction))   # scores stay between 0 and 100

    st.metric("Predicted math score", f"{prediction:.0f} / 100")
    st.caption("On test data the model's predictions were off by about 4 points on average (MAE 4.2).")

st.divider()
st.caption(
    "Learning project only. The predictions show patterns in one small dataset (1,000 students) "
    "and must not be used to make real decisions about students."
)
