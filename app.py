import streamlit as st
import joblib
import PyPDF2
import docx2txt
from resume_score import score_resume_against_jd
from suggestion_engine import get_suggestions
from ats_score import ats_score
from tech_extractor import extract_tech_entities
from extracurricular_extractor import extract_extracurriculars
from ai_classifier import classify_resume_origin  # New AI vs Human classifier

# ---------------- Text Extraction ----------------
def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return docx2txt.process(uploaded_file)
    elif uploaded_file.type == "text/plain":
        return uploaded_file.getvalue().decode("utf-8")
    else:
        return None

# ---------------- Load Models ----------------
model = joblib.load("model/rf_model.joblib")
vectorizer = joblib.load("model/vectorizer.joblib")

# ---------------- UI Setup ----------------
st.set_page_config(page_title="🧠 Resume Analyzer", layout="wide")
st.markdown("""
<style>
.badge {
    display: inline-block;
    padding: 0.35em 0.7em;
    margin: 0.2em 0.2em;
    font-size: 85%;
    font-weight: 600;
    color: white;
    border-radius: 0.35em;
}
.badge-tool { background-color: #1f77b4; }
.badge-lang { background-color: #ff7f0e; }
.badge-platform { background-color: #2ca02c; }
.badge-ner { background-color: #d62728; }
.badge-extra { background-color: #9467bd; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Resume Analyzer")

with st.sidebar:
    st.image("logo.png", width=120)
    st.markdown("## About")
    st.write("Analyze resumes for job fit, ATS score, AI/human origin, and skill extraction.")
    st.markdown("---")
    st.markdown("### Contact")
    st.write("Author: Madhughna Vedula")
    st.write("Can approach through linkedin and may suggest changes")
    st.write("[GitHub](https://github.com/MadhuVedula?)")
    st.write("[LinkedIn](https://www.linkedin.com/in/madhughna-vedula-417b8225b/)")

# ---------------- Input Fields ----------------
uploaded_file = st.file_uploader("📄 Upload your resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
resume_text = ""
if uploaded_file is not None:
    resume_text = extract_text_from_file(uploaded_file)
    if not resume_text.strip():
        st.error("Could not extract text from the uploaded file.")

jd_text = st.text_area("💼 Paste the job description here:")

if st.button("🚀 Analyze"):
    if not resume_text or not jd_text.strip():
        st.warning("Please upload a resume and paste the job description.")
    else:
        # ---------------- Core Processing ----------------
        X = vectorizer.transform([resume_text])
        prediction = model.predict(X)[0] if hasattr(model, "predict") else "Unknown"
        jd_score = score_resume_against_jd(resume_text, jd_text)
        ats = ats_score(resume_text)
        ai_or_human = classify_resume_origin(resume_text)
        suggestions = get_suggestions(resume_text)
        tech_entities = extract_tech_entities(resume_text)
        extracurriculars = extract_extracurriculars(resume_text)

        # ---------------- Output Section ----------------
        st.markdown("## 📊 Results Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("📁 Resume Type", ai_or_human)
        col2.metric("🎯 JD Match Score", f"{jd_score:.2f} %")
        col3.metric("📈 ATS Score", f"{ats:.2f} %")

        st.markdown("### 💡 Suggestions")
        if suggestions:
            with st.expander("View Suggestions"):
                for s in suggestions:
                    st.write("- " + s)
        else:
            st.success("No suggestions! Your resume looks good.")

        # ---------------- Technologies ----------------
        st.markdown("### 🛠️ Tech Stack & Skills")
        keywords = tech_entities.get("keyword_entities", {})
        ner_entities = tech_entities.get("ner_entities", [])

        def render_badges(items, badge_class):
            return " ".join(f"<span class='badge {badge_class}'>{item}</span>" for item in items)

        if keywords.get("programming_languages"):
            st.markdown("**Programming Languages:**")
            st.markdown(render_badges(keywords["programming_languages"], "badge-lang"), unsafe_allow_html=True)
        if keywords.get("tools_frameworks"):
            st.markdown("**Tools & Frameworks:**")
            st.markdown(render_badges(keywords["tools_frameworks"], "badge-tool"), unsafe_allow_html=True)
        if keywords.get("platforms"):
            st.markdown("**Platforms:**")
            st.markdown(render_badges(keywords["platforms"], "badge-platform"), unsafe_allow_html=True)
        if ner_entities:
            st.markdown("**NER Entities:**")
            st.markdown(render_badges(ner_entities, "badge-ner"), unsafe_allow_html=True)

        # ---------------- Extracurriculars ----------------
        if extracurriculars:
            st.markdown("### 🧑‍🎓 Extracurricular Activities / Volunteering")
            st.markdown(render_badges(extracurriculars, "badge-extra"), unsafe_allow_html=True)
