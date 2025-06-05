def ats_score(resume_text):
    score = 0
    if any(word in resume_text.lower() for word in ["education", "skills", "experience", "projects"]):
        score += 25
    if "contact" in resume_text.lower() and "email" in resume_text.lower():
        score += 25
    if len(resume_text.split()) > 250:
        score += 25
    if "certification" in resume_text.lower():
        score += 25
    return score
