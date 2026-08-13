# SmartHire GenAI — Resume Matching & AI Career Mentor

SmartHire GenAI is an end-to-end, AI-powered career assistance platform that helps job seekers extract resume intelligence, discover relevant job opportunities via vector similarity search, optimize CV bullet points, and consult an AI Career Mentor.

The application combines Generative AI (Google Gemini 2.5 Flash Lite), local zero-cost embeddings (SentenceTransformers `all-MiniLM-L6-v2`), FAISS vector databases, Hybrid Retrieval-Augmented Generation (RAG), input safety guardrails, automated evaluation benchmarking, and an interactive Streamlit web portal.

---

## 🚀 Live Demo & Web Application

The interactive SmartHire GenAI web portal is deployed and accessible online:

👉 **Live Deployed App**: [https://smarthiregenai-fnmvqvkgrjecvvqwm8zhzy.streamlit.app/](https://smarthiregenai-fnmvqvkgrjecvvqwm8zhzy.streamlit.app/)  
👉 **Local Development URL**: [http://localhost:8501](http://localhost:8501)

To launch the Streamlit application in your local environment:
```bash
streamlit run app/streamlit_app.py
```

---

## Overview & Workflow

SmartHire GenAI provides a 4-step workflow for candidate career optimization:

1. **📊 Candidate Profile Extraction**: Resume parsing with structured 5-key profile extraction (Name, Skills, Experience, Education, Target Role) via Gemini API with an automatic local fallback parser. Includes pre-loaded synthetic public PDF sample resumes.
2. **🎯 Semantic Job Matcher**: Sub-second vector similarity search using local 384-dimensional `all-MiniLM-L6-v2` embeddings and FAISS indices over job postings. Displays mathematically exact Cosine Semantic Similarity percentages and qualitative match badges.
3. **⚡ CV Optimizer**: AI-driven CV enhancement providing missing skill gap analysis, interactive skill coverage progress bars, before/after bullet point rewrites, tailored professional summaries, and downloadable Markdown reports (`.md`).
4. **💬 AI Career Mentor**: Hybrid RAG-powered career chatbot retrieving from BOTH curated career roadmaps and real job market postings, equipped with pre-execution safety guardrails and source citations.

---

## 🌟 Core Engineering Highlights

- 🔍 **Hybrid Dual-Retrieval RAG**: Simultaneously queries curated Markdown roadmaps (`notes_faiss`) and real job postings (`jobs_faiss`), attributing context blocks with exact `[Career Note]` and `[Job Posting]` labels.
- ⚡ **Zero-Cost Local FAISS Vector Search**: Performs sub-second semantic matching over 384-dimensional `all-MiniLM-L6-v2` embeddings, calculating exact Cosine Semantic Similarity percentages ($S_{\cos} = 1 - \frac{d^2}{2}$).
- 🛡️ **Hybrid Rate-Resilient Resume Parser**: Combines Google Gemini 2.5 Flash Lite structured JSON mode with an automatic local regex/taxonomy fallback parser (`_parse_resume_locally`) to handle 429 quota exhaustion gracefully.
- 🛡️ **Pre-Execution Input Guardrails**: Intercepts off-topic trivia, character bounds violations, and prompt injection attacks (`act as DAN`) prior to LLM invocation.
- 📈 **Automated Evaluation Benchmark Suite**: Self-contained module (`src/evaluate.py`) that evaluates Retrieval Precision @ 5 (100% Hit Rate), RAG grounding, and guardrail refusal, automatically compiling metrics to `reports/answer_quality.md`.
- 📥 **CV Optimization Summary Export**: Generates interactive skill coverage progress bars (`st.progress`) and downloadable Markdown summary reports (`CV_Optimization_Report_<Candidate_Name>.md`).

---

## 🖥️ Application Showcase & Module Walkthrough

SmartHire GenAI organizes candidate workflows across four specialized Streamlit portal tabs:

### 1. 📊 Candidate Intelligence Overview (Tab 1)
* Displays extracted 5-key candidate profile metrics: **Candidate Name**, **Target Role**, **Extracted Skills Count**, and **Accomplishments**.
* Renders color-coded skill chips (`<span class="skill-chip">`) and raw profile JSON expansion viewers.
* Includes 3 pre-loaded synthetic public PDF sample resumes ([`Alex Chen`](data/resumes/Sample_Software_Engineer_Resume.pdf), [`Jordan Taylor`](data/resumes/Sample_Data_Analyst_Resume.pdf), [`Morgan Vance`](data/resumes/Sample_ML_Engineer_Resume.pdf)) in `data/resumes/`.

### 2. 🎯 Semantic Job Matcher (Tab 2)
* Performs sub-second vector similarity matching against `vectorstore/jobs_faiss` using `SentenceTransformer("all-MiniLM-L6-v2")`.
* Renders color-coded match relevance badges:
  * `🟢 High Match` for $d < 1.1$ (Example: 81% Similarity at $d \approx 0.61$)
  * `🔵 Good Match` for $d < 1.25$ (Example: 63% Similarity at $d \approx 0.87$)
  * `🟡 Moderate Match` for $d \ge 1.25$ (Example: 34% Similarity at $d \approx 1.15$)
* Includes one-click **`🎯 Select for CV Analysis`** buttons to transfer target job context directly to Tab 3.

### 3. ⚡ CV Optimizer (Tab 3)
* **Interactive Skill Coverage Meter**: Visual progress bar (`st.progress`) comparing candidate skills against required job skills with green checkmark chips (`✓`) and orange warning chips (`⚠️`).
* **AI Bullet Point Rewrites**: Before/after bullet point diff analysis explaining why bullets are weak and providing action-verb rewrites.
* **Report Export**: `📥 Download CV Optimization Summary Report (.md)` button generating structured Markdown files (`CV_Optimization_Report_<Name>.md`).

### 4. 💬 AI Career Mentor (Tab 4)
* RAG-powered chatbot utilizing hybrid dual retrieval over career guides and job market postings.
* Features quick-prompt shortcuts (*"Data Analyst Skills Roadmap"*, *"Resume Writing Tips"*, *"Interview Preparation Guide"*).
* Displays cited reference sources directly below generated responses.

---

## Key Features & Component Details

### 1. Resume Parser (`src/parsing/`)
* Extracts structured candidate profiles containing `name`, `skills`, `experience`, `education`, and `target_role`.
* **Public Synthetic Demo Resumes (`data/resumes/`)**:
  * Includes three pre-built, realistic synthetic PDF sample resumes:
    - [`Sample_Software_Engineer_Resume.pdf`](data/resumes/Sample_Software_Engineer_Resume.pdf) *(Candidate: Alex Chen)*
    - [`Sample_Data_Analyst_Resume.pdf`](data/resumes/Sample_Data_Analyst_Resume.pdf) *(Candidate: Jordan Taylor)*
    - [`Sample_ML_Engineer_Resume.pdf`](data/resumes/Sample_ML_Engineer_Resume.pdf) *(Candidate: Morgan Vance)*
  * **Privacy Guarantee**: All candidate information in sample resumes is 100% synthetic. Personal resume files (e.g., `Lahari_Resume.pdf`) remain strictly private and excluded by `.gitignore`.
* **Hybrid Parser Architecture**:
  * Primary: Google Gemini 2.5 Flash Lite structured JSON mode.
  * Resilience Fallback (`_parse_resume_locally`): Pure local regex and evidence-weighted skill taxonomy parser that automatically intercepts 429 rate limits or network issues to return valid profile objects.

### 2. Semantic Job Matcher (`src/search/`)
* Converts candidate profile data into 384-dimensional vector embeddings using local `SentenceTransformer("all-MiniLM-L6-v2")`.
* Performs sub-second similarity matching against a FAISS L2 Euclidean index (`vectorstore/jobs_faiss`).
* **Mathematically Exact Cosine Semantic Similarity Percentage**:
  For unit-normalized vectors ($\|\vec{u}\| = \|\vec{v}\| = 1$), the Euclidean L2 distance $d = \|\vec{u} - \vec{v}\|_2$ and Cosine Similarity $S_{\cos}$ satisfy the exact mathematical identity:
  $$S_{\cos} = 1 - \frac{d^2}{2} \implies \text{Semantic Similarity (\%)} = \max\left(0, \min\left(100, \text{round}\left(100 \times \left(1 - \frac{d^2}{2}\right)\right)\right)\right)$$
* **Relevance Badges**: Displays human-readable match badges:
  * `🟢 High Match` for $d < 1.1$ (Example: 81% Similarity at $d \approx 0.61$)
  * `🔵 Good Match` for $d < 1.25$ (Example: 63% Similarity at $d \approx 0.87$)
  * `🟡 Moderate Match` for $d \ge 1.25$ (Example: 34% Similarity at $d \approx 1.15$)
* **Clarification**: The displayed percentage is a normalized **vector-text semantic similarity indicator** between candidate profile text and job postings; it is **NOT** a probability of getting hired or a job suitability prediction.

### 3. CV Optimizer (`src/generate/`)
* Compares candidate profile against selected target job descriptions.
* **Interactive Skill Coverage Visualization**: Displays visual progress bar (`st.progress`) and chip badges showing matched required skills (`✓`) vs missing required skills (`⚠️`).
* **Tailored Improvement Recommendations**:
  * **Skill Gap Analysis**: Identifies key missing technical skills.
  * **Bullet Point Diff Analysis**: Highlights weak bullet points and provides action-verb-oriented rewrites.
  * **Professional Summary**: Rewrites candidate summaries tailored to the target role.
* **Downloadable Report Export**: Includes a `📥 Download CV Optimization Summary Report (.md)` button generating structured Markdown files (`CV_Optimization_Report_<Candidate_Name>.md`).

### 4. Hybrid RAG AI Career Mentor (`src/mentor/`)
* Answers career development, roadmap, resume writing, interview prep, and job market demand questions.
* **Hybrid Dual Retrieval**: Retrieves top relevant chunks simultaneously from:
  1. **Career Notes Index (`vectorstore/notes_faiss`)**: Built over curated Markdown roadmaps (`data/career_notes/`):
     - `data_analyst_roadmap.md`
     - `resume_writing_tips.md`
     - `software_engineer_roadmap.md`
     - `interview_preparation_guide.md`
  2. **Job Market Index (`vectorstore/jobs_faiss`)**: Real job posting requirements and titles.
* **Context Labeling & Source Attribution**: Context blocks are labeled with `[Career Note: <filename>]` and `[Job Posting: <jobtitle> at <company>]`, with exact sources cited in mentor responses.

### 5. Input Safety Guardrails (`src/safety/`)
* Pre-execution validation layer running before LLM calls (`src/safety/guardrails.py`).
* **Safety Filters**:
  * Character length bounds validation.
  * Prompt injection & system override detection (e.g., `act as DAN`).
  * Harmful / inappropriate keyword filtering.
  * Off-topic question filter & career domain relevance validation.

### 6. Automated Evaluation Suite (`src/evaluate.py`, `reports/`)
* Complete automated benchmarking suite in `src/evaluate.py`:
  1. **Semantic Job Search Retrieval**: Precision @ 5 and Overall Hit Rate (100.0% benchmark score).
  2. **AI Career Mentor Answer Quality**: Evaluates answer correctness, RAG grounding, and source citations.
  3. **Prompt Comparison**: Documents output contrast between unstructured prompting vs structured JSON prompting.
  4. **Guardrail Refusal Check**: Verifies 100% refusal of off-topic trivia and prompt injections.
* Automatically compiles and saves execution metrics to [`reports/answer_quality.md`](reports/answer_quality.md).

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
│ Candidate     │ │ Semantic Job  │   │ CV Optimizer  │   │ Hybrid RAG    │
│ Profile       │ │ Matcher       │   │ & Report      │   │ Career Mentor │
│ (Synthetic/PDF) │ (1 - d²/2)    │   │ Download (.md)│   │ (Notes + Jobs)│
└───────┬───────┘ └───────┬───────┘   └───────┬───────┘   └───────┬───────┘
        │                 │                   │                   │
        ▼                 ▼                   ▼                   ▼
┌───────────────┐ ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ Gemini /      │ │ Local Model   │   │ Gemini 2.5    │   │ Local FAISS   │
│ Local Parser  │ │ MiniLM + FAISS│   │ Flash Lite    │   │ + Guardrails  │
└───────────────┘ └───────────────┘   └───────────────┘   └───────────────┘
```

---

## Technology Stack

* **Frontend**: Streamlit
* **LLM Engine**: Google Gemini API (`gemini-2.5-flash-lite`, `gemini-flash-latest`)
* **Embedding Model**: `sentence-transformers` (`all-MiniLM-L6-v2`, 384 dimensions)
* **Vector Database**: FAISS (`faiss-cpu`)
* **Orchestration / RAG**: LangChain (`langchain-community`, `langchain-core`)
* **Document Processing & PDF Generation**: `pypdf`, `reportlab`, `python-docx`, `pandas`
* **Environment & Testing**: `python-dotenv`, `py_compile`

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

### 4. Run Automated Evaluation Suite (Optional)
```bash
python src/evaluate.py
```

### 5. Run Streamlit Application
```bash
streamlit run app/streamlit_app.py
```

---

## Project Structure

```text
Smart_hire_Gen_AI/
├── app/
│   └── streamlit_app.py          # Streamlit Web Application (UI Portal)
├── src/
│   ├── config.py                 # Centralized configuration & paths
│   ├── evaluate.py               # Automated evaluation & benchmarking suite
│   ├── parsing/                  # Resume PDF/DOCX loaders & hybrid parser
│   ├── search/                   # Local embeddings & FAISS job search
│   ├── generate/                 # CV improvement generator & prompts
│   ├── mentor/                   # Hybrid RAG career mentor chain (Notes + Jobs)
│   └── safety/                   # Pre-execution safety guardrails
├── data/
│   ├── jobs/                     # Job dataset (naukri_com-job_sample.csv)
│   ├── career_notes/             # Markdown guides for RAG (Data Analyst, SE, Interview)
│   └── resumes/                  # Public synthetic PDF sample resumes
├── vectorstore/
│   ├── jobs_faiss/               # FAISS 384-dim job postings vector index
│   └── notes_faiss/              # FAISS 384-dim career notes vector index
├── reports/
│   └── answer_quality.md         # Automated evaluation benchmark report
├── .env.example                  # Environment configuration template
├── .gitignore                    # Git secrets & artifact exclusion rules
├── requirements.txt              # Production python package dependencies
└── README.md                     # Project documentation
```