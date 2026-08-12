# SmartHire GenAI — Resume Matching & AI Career Mentor

SmartHire GenAI is an AI-powered career assistance platform that helps job seekers analyze resumes, discover relevant job opportunities, optimize CV bullet points, and consult an AI Career Mentor.

The application combines Generative AI (Google Gemini 2.5 Flash Lite), local semantic search (SentenceTransformers `all-MiniLM-L6-v2`), FAISS vector databases, Retrieval-Augmented Generation (RAG), input safety guardrails, and an interactive Streamlit web application.

---

## Overview

SmartHire GenAI provides a 4-step workflow for candidate career optimization:

1. **📊 Candidate Profile Extraction**: Resume parsing with structured profile extraction (Name, Skills, Experience, Education, Target Role) via Gemini API with an automatic local fallback parser.
2. **🎯 Semantic Job Matcher**: Sub-second vector similarity search using local `all-MiniLM-L6-v2` embeddings and FAISS indices over job postings.
3. **⚡ CV Optimizer**: AI-driven CV enhancement providing missing skill gaps, bullet point improvements (before/after comparison), and tailored professional summaries.
4. **💬 AI Career Mentor**: RAG-powered career Q&A chatbot grounded in curated career roadmaps and resume writing guides, equipped with input guardrails.

---

## Key Features & Component Details

### 1. Resume Parser (`src/parsing/`)
* Extracts structured candidate profiles containing `name`, `skills`, `experience`, `education`, and `target_role`.
* **Hybrid Parser Architecture**:
  * Primary: Google Gemini 2.5 Flash Lite structured JSON mode.
  * Resilience Fallback (`_parse_resume_locally`): Pure local regex and skill taxonomy parser that automatically intercepts 429 rate limits or network issues to return valid profile objects.

### 2. Semantic Job Matcher (`src/search/`)
* Converts candidate profile data into 384-dimensional vector embeddings using local `SentenceTransformer("all-MiniLM-L6-v2")`.
* Performs sub-second similarity matching against a FAISS L2 Euclidean index (`vectorstore/jobs_faiss`).
* Displays human-readable relevance match badges (🟢 High Match, 🔵 Good Match, 🟡 Moderate Match) and enables direct candidate target job selection.

### 3. CV Optimizer (`src/generate/`)
* Compares candidate profile against selected target job descriptions.
* Generates tailored improvement recommendations:
  * **Skill Gap Analysis**: Identifies key missing technical skills.
  * **Bullet Point Diff Analysis**: Highlights weak bullet points and provides action-verb-oriented rewrites.
  * **Professional Summary**: Rewrites candidate summaries tailored to the target role.

### 4. RAG AI Career Mentor (`src/mentor/`)
* Answers career development, resume writing, and interview prep questions.
* **Retrieval-Augmented Generation (RAG)**: Retrieves top relevant documentation chunks from FAISS note indices (`vectorstore/notes_faiss`) built over curated career notes (`data/career_notes/`).
* **Source Attribution**: Cites specific reference documents used to formulate answers.

### 5. Input Safety Guardrails (`src/safety/`)
* Pre-execution validation layer running before LLM calls (`src/safety/guardrails.py`).
* **Safety Filters**:
  * Character length bounds validation.
  * Prompt injection & system override detection.
  * Harmful / inappropriate keyword filtering.
  * Off-topic question filter & career domain relevance validation.

### 6. Evaluation & Reports (`reports/`, `src/evaluate.py`)
* Evaluation framework and performance metrics documented in `reports/answer_quality.md`.
* `src/evaluate.py` serves as the evaluation module interface stub for automated benchmark extensions.

---

## Architecture & Data Flow

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                           User Interface                                │
│                     (Streamlit Web Portal - app/streamlit_app.py)       │
└───────┬─────────────────┬───────────────────┬───────────────────┬───────┘
        │                 │                   │                   │
        ▼                 ▼                   ▼                   ▼
┌───────────────┐ ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ Candidate     │ │ Semantic Job  │   │ CV Optimizer  │   │ AI Career     │
│ Profile       │ │ Matcher       │   │               │   │ Mentor        │
└───────┬───────┘ └───────┬───────┘   └───────┬───────┘   └───────┬───────┘
        │                 │                   │                   │
        ▼                 ▼                   ▼                   ▼
┌───────────────┐ ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ Gemini /      │ │ Local Model   │   │ Gemini 2.5    │   │ Local RAG     │
│ Local Parser  │ │ MiniLM + FAISS│   │ Flash Lite    │   │ + Guardrails  │
└───────────────┘ └───────────────┘   └───────────────┘   └───────────────┘
```

---

## Technology Stack

* **Frontend**: Streamlit
* **LLM Engine**: Google Gemini API (`gemini-2.5-flash-lite`)
* **Embedding Model**: `sentence-transformers` (`all-MiniLM-L6-v2`, 384 dimensions)
* **Vector Database**: FAISS (`faiss-cpu`)
* **Orchestration / RAG**: LangChain (`langchain-community`, `langchain-google-genai`)
* **Document Processing**: `pypdf`, `python-docx`, `pandas`
* **Environment Management**: `python-dotenv`

---

## Setup & Local Installation

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/GLahari-04/Smart_hire_Gen_AI.git
cd Smart_hire_Gen_AI

python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Keys
Create a `.env` file in the root directory (refer to `.env.example`):
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 4. Run Application
```bash
streamlit run app/streamlit_app.py
```

---

## Project Structure

```text
Smart_hire_Gen_AI/
├── app/
│   └── streamlit_app.py          # Streamlit Web Application
├── src/
│   ├── config.py                 # Centralized configuration & paths
│   ├── evaluate.py               # Evaluation module interface stub
│   ├── parsing/                  # Resume PDF/DOCX loaders & hybrid parser
│   ├── search/                   # Local embeddings & FAISS job search
│   ├── generate/                 # CV improvement generator & prompts
│   ├── mentor/                   # RAG career mentor chain
│   └── safety/                   # Pre-execution safety guardrails
├── data/
│   ├── jobs/                     # Job dataset (naukri_com-job_sample.csv)
│   ├── career_notes/             # Curated markdown guides for RAG
│   └── resumes/                  # Sample resumes for testing
├── vectorstore/
│   ├── jobs_faiss/               # FAISS 384-dim job postings vector index
│   └── notes_faiss/              # FAISS 384-dim career notes vector index
├── reports/
│   └── answer_quality.md         # Answer quality & retrieval evaluation report
├── .env.example                  # Environment configuration template
├── .gitignore                    # Git secrets & artifact exclusion rules
├── requirements.txt              # Production python package dependencies
└── README.md                     # Project documentation
```