import joblib

model = joblib.load("model/ai_vs_human_model.joblib")
vectorizer = joblib.load("model/ai_vs_human_vectorizer.joblib")

def predict_resume_origin(resume_text):
    X = vectorizer.transform([resume_text])
    return model.predict(X)[0]

