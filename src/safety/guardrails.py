"""Module 5 — Guardrails Layer.

Implements input validation, safety filtering, prompt injection detection,
and career-domain scope protection that runs BEFORE calling LLMs.

Pattern: validate_question(user_input) -> (is_valid: bool, feedback_message: str)
"""

import re
from typing import Tuple, List

# Minimum and maximum character length limits
MIN_INPUT_LENGTH = 5
MAX_INPUT_LENGTH = 1000

# Known prompt injection & system override patterns
PROMPT_INJECTION_PATTERNS: List[str] = [
    r"ignore\s+(all\s+)?(previous|above)\s+instructions",
    r"disregard\s+(all\s+)?instructions",
    r"forget\s+(all\s+)?previous\s+prompts",
    r"system\s+prompt",
    r"you\s+are\s+now\s+DAN",
    r"override\s+safety",
    r"jailbreak",
    r"bypass\s+restrictions",
    r"act\s+as\s+an\s+unrestricted",
    r"developer\s+mode",
]

# Unsafe / harmful content keywords
UNSAFE_KEYWORDS: List[str] = [
    "hack", "malware", "exploit", "virus", "ddos",
    "bomb", "weapon", "illegal", "suicide", "hate speech",
    "pirate", "bypass password", "steal credentials"
]

# Relevant career, job, interview, resume domain keywords
CAREER_DOMAIN_KEYWORDS: List[str] = [
    "career", "job", "work", "resume", "cv", "interview", "skill",
    "role", "salary", "hiring", "applicant", "roadmap", "analyst",
    "developer", "engineer", "experience", "education", "project",
    "learn", "course", "portfolio", "bullet point", "qualification",
    "industry", "transition", "apply", "company", "recruit", "sql",
    "python", "data", "software", "management", "code", "tech",
    "mentor", "advice", "guidance", "tip", "prep", "prepare", "preparation",
    "offer", "hire", "hired", "promotion", "intern", "internship", "java",
    "javascript", "react", "c++", "ai", "cloud", "aws", "azure", "git",
    "github", "frontend", "backend", "fullstack", "testing", "qa", "design"
]

# Explicitly off-topic general trivia / non-career patterns
OFF_TOPIC_PATTERNS: List[str] = [
    r"capital\s+of",
    r"\bwho\s+is\b",
    r"\bwho\s+was\b",
    r"\bwho\s+are\b",
    r"who\s+won\s+the",
    r"recipe\s+for",
    r"weather\s+in",
    r"movie\s+rating",
    r"score\s+of\s+the\s+game",
    r"tell\s+me\s+a\s+joke"
]


def validate_question(text: str) -> Tuple[bool, str]:
    """Validate user input before making an LLM call.

    Args:
        text (str): The raw text query submitted by the user.

    Returns:
        Tuple[bool, str]:
            - is_valid (bool): True if the query passes all guardrails, False otherwise.
            - message (str): Approval confirmation or specific rejection reason.
    """
    if text is None:
        return False, "Input cannot be empty. Please ask a valid career question."

    cleaned_text = text.strip()

    # 1. Empty / Whitespace check
    if not cleaned_text:
        return False, "Input cannot be empty. Please ask a valid career question."

    # 2. Length validation
    if len(cleaned_text) < MIN_INPUT_LENGTH:
        return False, f"Question is too short (minimum {MIN_INPUT_LENGTH} characters required)."

    if len(cleaned_text) > MAX_INPUT_LENGTH:
        return False, f"Question is too long (maximum {MAX_INPUT_LENGTH} characters allowed)."

    lower_text = cleaned_text.lower()

    # 3. Prompt Injection & Jailbreak check
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, lower_text):
            return False, "Input rejected: Prompt injection or system override phrase detected."

    # 4. Unsafe / Harmful Content check
    for keyword in UNSAFE_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", lower_text):
            return False, f"Input rejected: Unsafe or inappropriate content detected ('{keyword}')."

    # Check domain keyword presence
    has_domain_keyword = any(kw in lower_text for kw in CAREER_DOMAIN_KEYWORDS)

    # 5. Off-Topic / General Trivia pattern check
    for pattern in OFF_TOPIC_PATTERNS:
        if re.search(pattern, lower_text) and not has_domain_keyword:
            return False, "Input rejected: Question is off-topic. Please ask a career, job, resume, or interview-related question."

    # 6. Domain relevance check (ensure input relates to careers/jobs)
    if not has_domain_keyword:
        return False, "Input rejected: Question does not appear to be related to careers, jobs, resumes, or professional skills."

    return True, "Input validation passed."
