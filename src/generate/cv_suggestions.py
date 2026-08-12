"""Module 3 — CV improvement generator.

Generates tailored CV suggestions (Missing Skills, Weak Bullet Points, and Rewritten Summary)
for a candidate's resume profile against a target job using Google Gemini API.
"""

import json
import os
import time
from typing import Dict, Any, List
import google.generativeai as genai
from dotenv import load_dotenv

from src import config
from src.generate import prompts

try:
    from google.api_core.exceptions import ResourceExhausted
except ImportError:
    ResourceExhausted = Exception

# Load environment variables from a local .env file if available
load_dotenv()


def _get_configured_model() -> genai.GenerativeModel:
    """Helper function to verify API key configuration and return a Gemini GenerativeModel instance."""
    api_key = os.getenv(config.API_KEY_ENV)
    if not api_key:
        raise ValueError(
            f"Gemini API key not found. Please set the '{config.API_KEY_ENV}' "
            f"environment variable in your .env file."
        )

    genai.configure(api_key=api_key)

    # Enforce JSON output for Gemini responses
    generation_config = {
        "response_mime_type": "application/json",
        "temperature": 0.2,
    }

    return genai.GenerativeModel(
        model_name=config.CHAT_MODEL,
        generation_config=generation_config
    )


def generate_cv_suggestions(
    profile: Dict[str, Any],
    target_job: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate structured CV improvement suggestions for a candidate profile against a target job.

    Args:
        profile (Dict[str, Any]): Parsed candidate profile dictionary (from Step 1).
        target_job (Dict[str, Any]): Target job details dictionary (from Step 2).

    Returns:
        Dict[str, Any]: Structured dictionary containing:
            - "missing_skills": List[str]
            - "weak_bullet_points": List[Dict[str, str]] (with original_bullet, reason_weak, improved_bullet)
            - "rewritten_summary": str
    """
    if not profile or not isinstance(profile, dict):
        raise ValueError("Candidate profile must be a non-empty dictionary.")

    if not target_job or not isinstance(target_job, dict):
        raise ValueError("Target job must be a non-empty dictionary.")

    # Extract candidate details safely
    cand_name = str(profile.get("name", "Candidate"))
    cand_role = str(profile.get("target_role", "Not specified"))
    cand_skills = ", ".join(profile.get("skills", [])) if isinstance(profile.get("skills"), list) else str(profile.get("skills", ""))
    cand_exp = "\n- ".join(profile.get("experience", [])) if isinstance(profile.get("experience"), list) else str(profile.get("experience", ""))
    cand_edu = ", ".join(profile.get("education", [])) if isinstance(profile.get("education"), list) else str(profile.get("education", ""))

    # Extract target job details safely
    job_title = str(target_job.get("jobtitle", target_job.get("title", "N/A")))
    job_company = str(target_job.get("company", "N/A"))
    job_skills = str(target_job.get("skills", "N/A"))
    job_desc = str(target_job.get("snippet", target_job.get("jobdescription", "N/A")))

    # Format structured prompt template
    prompt = prompts.CV_SUGGESTIONS_PROMPT.format(
        candidate_name=cand_name,
        candidate_target_role=cand_role,
        candidate_skills=cand_skills,
        candidate_experience=cand_exp,
        candidate_education=cand_edu,
        job_title=job_title,
        job_company=job_company,
        job_skills=job_skills,
        job_description=job_desc
    )

    model = _get_configured_model()

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            raw_json_str = response.text.strip()
            data = json.loads(raw_json_str)

            # Extract and validate missing_skills
            raw_missing = data.get("missing_skills", [])
            missing_skills = [str(s) for s in raw_missing] if isinstance(raw_missing, list) else []

            # Extract and validate weak_bullet_points
            raw_weak = data.get("weak_bullet_points", [])
            weak_bullets = []
            if isinstance(raw_weak, list):
                for item in raw_weak:
                    if isinstance(item, dict):
                        weak_bullets.append({
                            "original_bullet": str(item.get("original_bullet", "N/A")),
                            "reason_weak": str(item.get("reason_weak", "N/A")),
                            "improved_bullet": str(item.get("improved_bullet", "N/A")),
                        })

            # Extract rewritten summary
            rewritten_summary = str(data.get("rewritten_summary", ""))

            return {
                "missing_skills": missing_skills,
                "weak_bullet_points": weak_bullets,
                "rewritten_summary": rewritten_summary
            }

        except (ResourceExhausted, Exception) as error:
            err_msg = str(error)
            if ("429" in err_msg or "ResourceExhausted" in err_msg or "quota" in err_msg) and attempt < max_retries - 1:
                sleep_time = (2 ** (attempt + 1)) + 2
                time.sleep(sleep_time)
            elif isinstance(error, json.JSONDecodeError) and attempt < max_retries - 1:
                time.sleep(1)
            else:
                print(f"Error generating CV suggestions: {error}")
                return {
                    "missing_skills": [],
                    "weak_bullet_points": [],
                    "rewritten_summary": "Unable to generate CV suggestions due to processing error.",
                    "error": str(error)
                }

    return {
        "missing_skills": [],
        "weak_bullet_points": [],
        "rewritten_summary": "Failed after max retries."
    }
