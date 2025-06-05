import spacy
import re

# Load spaCy English model (run: python -m spacy download en_core_web_sm)
import subprocess
import sys

# Ensure the model is downloaded at runtime
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

PROGRAMMING_LANGUAGES = [
    "Python", "Java", "C++", "JavaScript", "TypeScript", "Ruby", "Go", "Rust",
    "Scala", "Kotlin", "Swift", "PHP", "SQL", "R", "MATLAB"
]

TOOLS_AND_FRAMEWORKS = [
    "Django", "Flask", "React", "Angular", "Vue", "Spring", "Node.js",
    "Express", "Docker", "Kubernetes", "TensorFlow", "PyTorch", "AWS",
    "Azure", "GCP", "Git", "Jenkins", "Travis", "CircleCI", "Ansible",
    "Terraform", "REST", "GraphQL", "MongoDB", "MySQL", "PostgreSQL", "Redis"
]

PLATFORMS = [
    "Linux", "Windows", "macOS", "iOS", "Android"
]

def extract_keywords(text):
    keywords_found = {
        "programming_languages": [],
        "tools_frameworks": [],
        "platforms": []
    }
    text_lower = text.lower()

    for lang in PROGRAMMING_LANGUAGES:
        if lang.lower() in text_lower:
            keywords_found["programming_languages"].append(lang)

    for tool in TOOLS_AND_FRAMEWORKS:
        if tool.lower() in text_lower:
            keywords_found["tools_frameworks"].append(tool)

    for plat in PLATFORMS:
        if plat.lower() in text_lower:
            keywords_found["platforms"].append(plat)

    return keywords_found

def extract_named_entities(text):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        if ent.label_ in ("ORG", "PRODUCT", "WORK_OF_ART", "LANGUAGE", "NORP"):
            entities.append(ent.text)
    return list(set(entities))

def extract_tech_entities(text):
    """
    Combine keyword extraction and NER-based extraction.
    Returns dict with keyword categories and list of named entities.
    """
    keywords = extract_keywords(text)
    ner_entities = extract_named_entities(text)
    return {
        "keyword_entities": keywords,
        "ner_entities": ner_entities
    }
