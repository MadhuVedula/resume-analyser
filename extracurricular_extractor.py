import re

def extract_extracurriculars(text):
    activities = []

    patterns = [
        r'\bvolunteer(?:ed|ing)?(?: at)?\b',
        r'\b(clean[\s\-]?up drive|blood donation|awareness campaign|community service)\b',
        r'\b(hackathon(?:s)?|coding contest(?:s)?|programming competition)\b',
        r'\b(event(?:s)?|fest(?:s)?|workshop(?:s)?|seminar(?:s)?|conference|webinar|talk|symposium)\b',
        r'\b(marathon|social cause|environmental activity|ngo|campaign|student club|ambassador|leadership)\b'
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # Normalize capitalization and add
            activities.extend(set([m.strip().title() for m in matches]))

    return list(set(activities))
