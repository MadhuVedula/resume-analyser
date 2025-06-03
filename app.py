
import streamlit as st
from resume_parser import extract_resume_text
import joblib

# Load model and vectorizer
model = joblib.load("model/resume_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

st.title("Resume Analyzer")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])

if uploaded_file:
    resume_text = extract_resume_text(uploaded_file)
    st.subheader("Extracted Resume Text:")
    st.write(resume_text[:1000])  # preview

    features = vectorizer.transform([resume_text])
    prediction = model.predict(features)
    st.success(f"Predicted Category: **{prediction[0]}**")
import streamlit as st
from resume_parser import extract_resume_text
import joblib

# Load model and vectorizer
model = joblib.load("model/resume_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

st.title("Resume Analyzer")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])

if uploaded_file:
    resume_text = extract_resume_text(uploaded_file)
    st.subheader("Extracted Resume Text:")
    st.write(resume_text[:1000])  # preview

    features = vectorizer.transform([resume_text])
    prediction = model.predict(features)
    st.success(f"Predicted Category: **{prediction[0]}**")
