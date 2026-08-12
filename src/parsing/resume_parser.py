"""Module 1 — Resume parser.

Extracts structured profile data (Name, Skills, Experience, Education, Target Role)
from resume documents (PDF, DOCX, TXT, MD) using Google Gemini structured JSON output,
with an automatic local regex/taxonomy fallback when Gemini API rate limits (429) are hit.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, Any, Union, List
import google.generativeai as genai
from dotenv import load_dotenv

from src import config
from src.parsing import loader

# Load environment variables from a local .env file if available
load_dotenv()

# Technical skills taxonomy for local resume parsing fallback
LOCAL_SKILLS_TAXONOMY = [
    "Python", "Java", "C++", "C#", "C", "JavaScript", "TypeScript", "SQL", "PL/SQL", "R",
    "HTML", "CSS", "Go", "Golang", "Scala", "Bash", "Shell", "PowerShell", "PHP", "Ruby",
    "Data Analysis", "Data Science", "Machine Learning", "Deep Learning", "Artificial Intelligence",
    "Pandas", "NumPy", "Scikit-Learn", "PyTorch", "TensorFlow", "Keras", "NLTK", "OpenCV",
    "Tableau", "PowerBI", "Power BI", "Excel", "ETL", "Big Data", "Hadoop", "Spark", "Hive",
    "Airflow", "Kafka", "Data Warehousing", "Snowflake", "Databricks",
    "React", "Angular", "Vue", "Node.js", "Express", "Flask", "FastAPI", "Django", "Spring Boot",
    "REST API", "GraphQL", "Microservices", "HTML5", "CSS3", "Bootstrap", "Tailwind",
    "AWS", "Azure", "GCP", "Google Cloud", "Docker", "Kubernetes", "CI/CD", "Git", "GitHub",
    "GitLab", "Linux", "Unix", "Terraform", "Ansible", "Jenkins", "JIRA", "Agile", "Scrum",
    "MySQL", "PostgreSQL", "MongoDB", "Redis", "Oracle", "SQLite", "DynamoDB", "NoSQL",
    "Communication", "Problem Solving", "Leadership", "Teamwork"
]


def _get_configured_model() -> genai.GenerativeModel:
    """Helper function to verify API key configuration and return a Gemini GenerativeModel instance."""
    api_key = os.getenv(config.API_KEY_ENV)
    if not api_key:
        raise ValueError(
            f"Gemini API key not found. Please set the '{config.API_KEY_ENV}' "
            f"environment variable in your .env file."
        )

    genai.configure(api_key=api_key)

    # Configure Gemini to enforce JSON structured output and deterministic extraction
    generation_config = {
        "response_mime_type": "application/json",
        "temperature": 0.1,
    }

    return genai.GenerativeModel(
        model_name=config.CHAT_MODEL,
        generation_config=generation_config
    )


def _parse_resume_locally(resume_text: str) -> Dict[str, Any]:
    """Pure local fallback parser using regex heuristics and skill taxonomy matching.

    Args:
        resume_text (str): Raw resume text string.

    Returns:
        Dict[str, Any]: Structured dictionary matching the 5-key profile schema.
    """
    if not resume_text or not resume_text.strip():
        return {
            "name": "Unknown",
            "skills": [],
            "experience": [],
            "education": [],
            "target_role": "Not specified",
        }

    lines = [line.strip() for line in resume_text.splitlines() if line.strip()]

    # 1. Candidate Name Extraction (top non-contact header line)
    candidate_name = "Unknown"
    for line in lines[:5]:
        if not re.search(r'(@|http|www|\+|\d{5,}|resume|curriculum|profile|address|email)', line, re.IGNORECASE):
            if len(line.split()) <= 5 and re.match(r'^[A-Za-z\s\.\'-]+$', line):
                candidate_name = line.strip()
                break

    # 2. Skill Taxonomy Detection
    extracted_skills: List[str] = []
    text_lower = resume_text.lower()
    for skill in LOCAL_SKILLS_TAXONOMY:
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, text_lower):
            extracted_skills.append(skill)

    # 3. Section Segmentation (Experience & Education)
    experience_bullets: List[str] = []
    education_items: List[str] = []
    current_section = None

    for line in lines:
        upper_line = line.upper()

        if any(h in upper_line for h in ["EXPERIENCE", "WORK HISTORY", "EMPLOYMENT", "PROJECTS", "WORK EXPERIENCE"]):
            current_section = "EXP"
            continue
        elif any(h in upper_line for h in ["EDUCATION", "ACADEMIC", "QUALIFICATIONS", "CERTIFICATIONS", "DEGREES"]):
            current_section = "EDU"
            continue
        elif any(h in upper_line for h in ["SKILLS", "TECHNICAL SKILLS", "SUMMARY", "OBJECTIVE", "DECLARATION"]):
            current_section = "OTHER"
            continue

        if current_section == "EXP":
            if line.startswith(('-', '•', '*', '>', '–', '—')) or len(line) > 20:
                clean_bullet = line.lstrip('-•*>–— ').strip()
                if clean_bullet and len(clean_bullet) > 10:
                    experience_bullets.append(clean_bullet)
        elif current_section == "EDU":
            if any(k in upper_line for k in ["BACHELOR", "MASTER", "DEGREE", "B.TECH", "B.E.", "B.S.", "M.S.", "M.TECH", "COLLEGE", "UNIVERSITY", "DIPLOMA"]):
                education_items.append(line.strip())

    # Fallbacks if sections were not explicitly tagged
    if not experience_bullets:
        for line in lines:
            if line.startswith(('-', '•', '*', '>')) and len(line) > 15:
                experience_bullets.append(line.lstrip('-•*> ').strip())
        if not experience_bullets and len(lines) > 2:
            experience_bullets = [l for l in lines[1:6] if len(l) > 15]

    if not education_items:
        for line in lines:
            if any(k in line.upper() for k in ["BACHELOR", "MASTER", "DEGREE", "B.TECH", "B.E.", "B.S.", "M.S.", "UNIVERSITY"]):
                education_items.append(line.strip())

    # 4. Target Role Inference (Evidence-weighted matching)
    target_role = "Not specified"

    # Step A: Check top lines for an explicit job title heading
    for line in lines[:8]:
        line_lower = line.lower()
        if any(role in line_lower for role in ["software engineer", "software developer", "data analyst", "data scientist", "machine learning engineer", "full stack developer", "frontend developer", "backend developer", "system architect"]) and not any(e in line_lower for e in ["b.tech", "degree", "bachelor", "master", "education"]):
            target_role = line.strip()
            break

    # Step B: Weighted evidence matching if no explicit header title is present
    if target_role == "Not specified":
        se_score = 0
        da_score = 0
        ml_score = 0

        # Degree & Education evidence
        edu_text_lower = " ".join(education_items).lower() if education_items else text_lower[:500]
        if any(term in edu_text_lower for term in ["computer science", "computer engineering", "software engineering", "cse"]):
            se_score += 3
        if any(term in edu_text_lower for term in ["statistics", "analytics", "business intelligence", "economics"]):
            da_score += 3

        # Skill & Technology evidence
        skills_set = set(extracted_skills)
        se_skills = {"Java", "C++", "C#", "C", "JavaScript", "TypeScript", "Go", "Ruby", "PHP", "React", "Angular", "Vue", "Node.js", "Express", "Spring Boot", "Django", "FastAPI", "REST API", "GraphQL", "Microservices", "HTML", "CSS", "HTML5", "CSS3"}
        da_skills = {"Tableau", "PowerBI", "Power BI", "Excel", "ETL", "Data Warehousing", "Snowflake", "Databricks", "Data Analysis", "SAS"}
        ml_skills = {"Machine Learning", "Deep Learning", "PyTorch", "TensorFlow", "Keras", "Scikit-Learn", "NLTK", "OpenCV", "Data Science", "Artificial Intelligence"}

        se_score += len(se_skills.intersection(skills_set)) * 2
        da_score += len(da_skills.intersection(skills_set)) * 2
        ml_score += len(ml_skills.intersection(skills_set)) * 2

        # Shared skills (Python, SQL, R)
        if "Python" in skills_set:
            se_score += 1
            da_score += 1
            ml_score += 1
        if "SQL" in skills_set:
            se_score += 1
            da_score += 1

        # Project & Experience text keywords
        exp_text_lower = " ".join(experience_bullets).lower() if experience_bullets else text_lower
        if any(term in exp_text_lower for term in ["web", "app", "application", "booking", "system", "frontend", "backend", "full stack"]):
            se_score += 2
        if any(term in exp_text_lower for term in ["dashboard", "visualization", "reporting", "insight"]):
            da_score += 2
        if any(term in exp_text_lower for term in ["prediction model", "model training", "neural network", "classification"]):
            ml_score += 2

        # Determine highest scoring role
        max_score = max(se_score, da_score, ml_score)
        if max_score > 0:
            if max_score == se_score:
                target_role = "Software Engineer"
            elif max_score == ml_score:
                target_role = "Data Scientist / ML Engineer"
            elif max_score == da_score:
                target_role = "Data Analyst"
        elif extracted_skills:
            target_role = f"{extracted_skills[0]} Specialist"

    return {
        "name": candidate_name,
        "skills": extracted_skills,
        "experience": experience_bullets[:10],
        "education": education_items[:5],
        "target_role": target_role
    }


def parse_resume_text(resume_text: str) -> Dict[str, Any]:
    """Parse raw resume text into a structured candidate profile dictionary using Gemini API,
    falling back to local parsing if a 429 rate limit is encountered.

    Args:
        resume_text (str): The plain text content extracted from a resume.

    Returns:
        Dict[str, Any]: Structured candidate profile containing:
            - name (str)
            - skills (List[str])
            - experience (List[str])
            - education (List[str])
            - target_role (str)
    """
    if not resume_text or not resume_text.strip():
        # Handle empty text input gracefully with default empty values
        return {
            "name": "Unknown",
            "skills": [],
            "experience": [],
            "education": [],
            "target_role": "Not specified",
        }

    # Initialize configured Gemini model
    model = _get_configured_model()

    # Structured prompt guiding Gemini to return JSON adhering strictly to our schema
    prompt = f"""
You are an expert HR and resume parsing AI assistant.
Analyze the following resume text and extract key profile details into a valid JSON object.

The output MUST be a valid JSON object containing EXACTLY these keys:
- "name": (string) Candidate's full name, or "Unknown" if not present.
- "skills": (list of strings) List of technical skills, programming languages, tools, and domain expertise.
- "experience": (list of strings) Summary bullet points of job roles, titles, responsibilities, or projects.
- "education": (list of strings) Degrees, certifications, majors, or educational institutions.
- "target_role": (string) Inferred primary target job role or current career focus based on the resume.

Resume Content:
---
{resume_text}
---
"""

    try:
        # Call Gemini model
        response = model.generate_content(prompt)

        # Parse output JSON text
        raw_json_str = response.text.strip()
        data = json.loads(raw_json_str)

        # Safely assemble profile dict with safe fallback defaults for each field
        profile = {
            "name": str(data.get("name", "Unknown")),
            "skills": data.get("skills", []) if isinstance(data.get("skills"), list) else [],
            "experience": data.get("experience", []) if isinstance(data.get("experience"), list) else [],
            "education": data.get("education", []) if isinstance(data.get("education"), list) else [],
            "target_role": str(data.get("target_role", "Not specified")),
        }
        return profile

    except Exception as error:
        err_msg = str(error)
        # Check if error is due to Gemini API rate limit or 429 quota exhaustion
        if "429" in err_msg or "ResourceExhausted" in err_msg or "quota" in err_msg or "Quota" in err_msg:
            print(f"Notice: Gemini API rate limit hit ({error}). Activating local resume parser fallback...")
            return _parse_resume_locally(resume_text)

        # Graceful handling for non-quota parsing errors
        print(f"Warning: Failed to parse Gemini response as structured JSON ({error}).")
        return {
            "name": "Unknown",
            "skills": [],
            "experience": [],
            "education": [],
            "target_role": "Not specified",
            "error": str(error)
        }


def parse_resume(file_path_or_text: Union[str, Path]) -> Dict[str, Any]:
    """Load a resume file (.pdf, .docx, .txt, .md) or raw text and parse it into structured JSON.

    Args:
        file_path_or_text (Union[str, Path]): File path to resume document OR raw resume text string.

    Returns:
        Dict[str, Any]: Structured resume profile dictionary.
    """
    path = Path(file_path_or_text) if isinstance(file_path_or_text, str) else file_path_or_text

    # Check if the parameter refers to an existing document file on disk
    if isinstance(path, Path) and path.exists() and path.is_file():
        # Use loader module to extract plain text from PDF/DOCX/TXT
        resume_text = loader.load_text(path)
    else:
        # Treat parameter as raw string text
        resume_text = str(file_path_or_text)

    return parse_resume_text(resume_text)
