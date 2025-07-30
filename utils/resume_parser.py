import PyPDF2
import docx
import spacy
import re
from spacy.matcher import Matcher
from .skills import SKILLS_LIST

nlp = spacy.load("en_core_web_sm")

def extract_text_from_file(uploaded_file):
    """Extracts raw text from PDF or DOCX files."""
    file_extension = uploaded_file.name.split('.')[-1].lower()
    text = ""
    if file_extension == 'pdf':
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            return f"Error parsing PDF: {e}"
    elif file_extension == 'docx':
        try:
            doc = docx.Document(uploaded_file)
            for para in doc.paragraphs:
                text += para.text + "\n"
        except Exception as e:
            return f"Error parsing DOCX: {e}"
    else:
        return "Unsupported file type."
    return text

def extract_name(text):
    """Extracts the name using spaCy's Named Entity Recognition (NER)."""
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return None

def extract_email(text):
    """Extracts the email using a regular expression."""
    match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    return match.group(0) if match else None

def extract_skills(text):
    """Extracts skills using spaCy's Matcher."""
    matcher = Matcher(nlp.vocab)

    for skill in SKILLS_LIST:
        pattern = [{"LOWER": word} for word in skill.lower().split()]
        matcher.add(skill, [pattern])
    
    doc = nlp(text)
    matches = matcher(doc)
    
    found_skills = set()
    for match_id, start, end in matches:
        span = doc[start:end]
        found_skills.add(span.text.title()) # Use title case for nice formatting
    
    return list(found_skills)


def process_resume(uploaded_file):
    """
    The main function to process an uploaded resume.
    It extracts text and then structured data from the text.
    """
    raw_text = extract_text_from_file(uploaded_file)
    if "Error" in raw_text or "Unsupported" in raw_text:
        return {"error": raw_text}

    name = extract_name(raw_text)
    email = extract_email(raw_text)
    skills = extract_skills(raw_text)
    
    return {
        "name": name,
        "email": email,
        "skills": skills,
        "raw_text": raw_text
    }
