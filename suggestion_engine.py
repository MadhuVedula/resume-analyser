import re
from tech_extractor import extract_tech_entities

COMMON_MISSPELLINGS = {
    "teh": "the",
    "recieve": "receive",
    "adress": "address",
    "langauge": "language",
    "flaskk": "flask",
    "pyhton": "python",
    "javscript": "javascript",
    "enviroment": "environment",
    "proficienty": "proficiency",
    "mangement": "management",
    "dependecy": "dependency",
    "intermidiate": "intermediate"
}

def get_suggestions(resume_text):
    tech_data = extract_tech_entities(resume_text)

    tech_terms = set(tech_data["ner_entities"])
    for category in tech_data["keyword_entities"]:
        tech_terms.update(tech_data["keyword_entities"][category])
    tech_terms = set(term.lower() for term in tech_terms)

    suggestions = []
    words = re.findall(r'\b\w+\b', resume_text)

    for word in words:
        word_lower = word.lower()
        if word_lower in tech_terms:
            continue
        if word_lower in COMMON_MISSPELLINGS:
            correct = COMMON_MISSPELLINGS[word_lower]
            suggestions.append(f"Consider correcting '{word}' to '{correct}'.")

    return suggestions
