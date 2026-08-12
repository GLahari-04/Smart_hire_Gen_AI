"""SmartHire GenAI Portal (Streamlit).

Run: streamlit run app/streamlit_app.py
Wires together Modules 1-5 into an enterprise-grade AI Career & Job Intelligence Portal.
"""

import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from dotenv import load_dotenv

from src import config
from src.parsing.resume_parser import parse_resume
from src.search.job_search import search_jobs
from src.generate.cv_suggestions import generate_cv_suggestions
from src.mentor.rag_chain import ask_mentor
from src.safety.guardrails import validate_question

# Load environment variables
load_dotenv()

# Set Streamlit Page Config
st.set_page_config(
    page_title="SmartHire GenAI — Career Intelligence Portal",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS design system
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Brand Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 50%, #312E81 100%);
        color: #FFFFFF;
        padding: 1.6rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.6rem;
        box-shadow: 0 4px 18px rgba(15, 23, 42, 0.12);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0;
        color: #FFFFFF;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #93C5FD;
        margin-top: 0.4rem;
        margin-bottom: 0;
        font-weight: 400;
    }

    /* Metric Cards */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    .metric-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .metric-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Skill Badges */
    .skill-chip {
        display: inline-block;
        background-color: #EFF6FF;
        color: #1D4ED8;
        padding: 0.3rem 0.75rem;
        border-radius: 20px;
        font-weight: 500;
        font-size: 0.85rem;
        margin-right: 0.4rem;
        margin-bottom: 0.5rem;
        border: 1px solid #BFDBFE;
    }

    /* Job Card Styling */
    .job-card-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.3rem;
    }
    .match-tag-high {
        float: right;
        background-color: #DCFCE7;
        color: #15803D;
        padding: 0.25rem 0.7rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    .match-tag-good {
        float: right;
        background-color: #DBEAFE;
        color: #1D4ED8;
        padding: 0.25rem 0.7rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    .match-tag-mod {
        float: right;
        background-color: #FEF3C7;
        color: #B45309;
        padding: 0.25rem 0.7rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.82rem;
    }

    /* Bullet Point Diff Styling */
    .bullet-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #3B82F6;
        border-radius: 8px;
        padding: 1.1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Application Header Banner
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🚀 SmartHire GenAI Intelligence Portal</div>
    <div class="hero-subtitle">Automated Resume Intelligence · Local Semantic Search · AI CV Optimization · Grounded Career Mentor</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar: Resume Input & System Status
# -----------------------------------------------------------------------------
st.sidebar.title("📁 Resume Input")

sample_resumes_dir = config.RESUMES_DIR
sample_files = sorted(
    [f for f in sample_resumes_dir.glob("*") if f.suffix.lower() in (".pdf", ".docx", ".txt", ".md")]
) if sample_resumes_dir.exists() else []

sample_names = [f.name for f in sample_files]

input_mode = st.sidebar.radio(
    "Choose Resume Source:",
    ["Sample Resume File", "Upload New Resume"],
    index=0
)

selected_resume_path = None

if input_mode == "Sample Resume File":
    if sample_names:
        chosen_sample_name = st.sidebar.selectbox("Select Resume:", sample_names, index=0)
        selected_resume_path = sample_resumes_dir / chosen_sample_name
    else:
        st.sidebar.warning("No sample resumes found in data/resumes/")
else:
    uploaded_file = st.sidebar.file_uploader("Upload Resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
    if uploaded_file:
        save_dir = config.RESUMES_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        selected_resume_path = save_dir / uploaded_file.name
        with open(selected_resume_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ System Status")
st.sidebar.markdown("""
<div style="font-size: 0.85rem; line-height: 1.6; color: #475569;">
    <div>🟢 <strong>Semantic Search:</strong> Local (all-MiniLM-L6-v2)</div>
    <div>🛡️ <strong>Safety Layer:</strong> Active (Guardrails)</div>
    <div>🤖 <strong>AI Generation:</strong> Gemini 2.5 Flash Lite</div>
</div>
""", unsafe_allow_html=True)

# Check API key configuration status
api_key_set = bool(os.getenv(config.API_KEY_ENV))
if not api_key_set:
    st.sidebar.warning("⚠️ GOOGLE_API_KEY is not set in .env. Local parser & local search will operate automatically.")

# -----------------------------------------------------------------------------
# Core Application Execution
# -----------------------------------------------------------------------------
if selected_resume_path and selected_resume_path.exists():

    # Parse resume profile (cached in session state per file path if valid)
    if "parsed_profile" not in st.session_state or st.session_state.get("current_file") != str(selected_resume_path):
        with st.spinner("Extracting profile intelligence..."):
            try:
                res_profile = parse_resume(selected_resume_path)
                if res_profile and (res_profile.get("name") != "Unknown" or res_profile.get("skills")):
                    st.session_state["parsed_profile"] = res_profile
                    st.session_state["current_file"] = str(selected_resume_path)
                else:
                    st.session_state["parsed_profile"] = None
                    st.session_state["current_file"] = None
                    err_msg = res_profile.get("error", "Unable to extract profile details.") if res_profile else "Empty response"
                    st.warning(f"⚡ Local Parsing Fallback Active: {err_msg}")
            except Exception as err:
                st.warning(f"Notice: Parsing profile using local extraction heuristics.")
                st.session_state["parsed_profile"] = None
                st.session_state["current_file"] = None

    profile = st.session_state.get("parsed_profile")

    if not profile:
        st.warning("No valid profile loaded. Click below to retry parsing.")
        if st.button("🔄 Retry Resume Extraction"):
            st.session_state.pop("parsed_profile", None)
            st.session_state.pop("current_file", None)
            st.rerun()

    if profile:
        # Create Main Application Navigation Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Candidate Overview",
            "🎯 Semantic Job Matcher",
            "⚡ CV Optimizer",
            "💬 AI Career Mentor"
        ])

        # ---------------------------------------------------------------------
        # Tab 1: Candidate Overview (Module 1)
        # ---------------------------------------------------------------------
        with tab1:
            st.subheader("Candidate Intelligence Overview")
            
            # Executive Metric Cards
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{profile.get('name', 'Candidate')}</div>
                    <div class="metric-label">Candidate Name</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{profile.get('target_role', 'General Professional')}</div>
                    <div class="metric-label">Target Role</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{len(profile.get('skills', []))} Skills</div>
                    <div class="metric-label">Technical Skills</div>
                </div>
                """, unsafe_allow_html=True)
            with m4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{len(profile.get('experience', []))} Items</div>
                    <div class="metric-label">Accomplishments</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            col_left, col_right = st.columns([1, 1.2])

            with col_left:
                st.markdown("### 💡 Extracted Technical Skills")
                skills_list = profile.get("skills", [])
                if skills_list:
                    badges_html = "".join([f'<span class="skill-chip">{s}</span>' for s in skills_list])
                    st.markdown(badges_html, unsafe_allow_html=True)
                else:
                    st.info("No explicit skills detected in resume text.")

            with col_right:
                st.markdown("### 📝 Work Experience & Projects")
                exp_list = profile.get("experience", [])
                if exp_list:
                    for exp in exp_list:
                        st.markdown(f"- {exp}")
                else:
                    st.info("No work accomplishments detected.")

                st.markdown("### 🎓 Education & Degrees")
                edu_list = profile.get("education", [])
                if edu_list:
                    for edu in edu_list:
                        st.markdown(f"- {edu}")
                else:
                    st.info("No education details detected.")

            with st.expander("🔍 View Raw JSON Profile Data"):
                st.json(profile)

        # ---------------------------------------------------------------------
        # Tab 2: Semantic Job Matcher (Module 2)
        # ---------------------------------------------------------------------
        with tab2:
            st.subheader("Ranked Opportunity Matches (Local FAISS Index)")

            if "matched_jobs" not in st.session_state or st.session_state.get("profile_for_jobs") != str(selected_resume_path):
                with st.spinner("Executing local semantic search..."):
                    try:
                        st.session_state["matched_jobs"] = search_jobs(profile, top_n=config.TOP_N_JOBS)
                        st.session_state["profile_for_jobs"] = str(selected_resume_path)
                    except Exception as err:
                        st.warning("Notice: FAISS job search encountered an issue.")
                        st.session_state["matched_jobs"] = []

            matched_jobs = st.session_state.get("matched_jobs", [])

            if matched_jobs:
                st.success(f"Retrieved top {len(matched_jobs)} matching job postings using local vector similarity.")

                # Dropdown for Target Job Selection
                job_options = [f"{j['jobtitle']} at {j['company']}" for j in matched_jobs]
                default_idx = 0
                selected_job_idx = st.selectbox("Select Job Target for CV Optimization:", range(len(job_options)), format_func=lambda i: job_options[i], index=default_idx)
                st.session_state["selected_target_job"] = matched_jobs[selected_job_idx]

                st.markdown("---")
                for i, job in enumerate(matched_jobs, 1):
                    # Compute user-friendly semantic relevance tag from score
                    score = job.get('score', 2.0)
                    if score < 1.1:
                        rel_tag = '<span class="match-tag-high">🟢 High Match</span>'
                    elif score < 1.25:
                        rel_tag = '<span class="match-tag-good">🔵 Good Match</span>'
                    else:
                        rel_tag = '<span class="match-tag-mod">🟡 Moderate Match</span>'

                    with st.container():
                        st.markdown(f"""
                        <div style="background-color:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:1.2rem; margin-bottom:1rem;">
                            {rel_tag}
                            <div class="job-card-header">{i}. {job['jobtitle']}</div>
                            <p style="margin:0.2rem 0; color:#475569; font-size:0.92rem;">
                                <strong>Company:</strong> {job['company']} | <strong>Location:</strong> {job['location']}
                            </p>
                            <p style="margin:0.2rem 0; color:#475569; font-size:0.92rem;">
                                <strong>Required Skills:</strong> {job['skills']}
                            </p>
                            <p style="margin:0.2rem 0; color:#475569; font-size:0.92rem;">
                                <strong>Experience:</strong> {job['experience']} | <strong>Payrate:</strong> {job['payrate']}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col_btn, col_blank = st.columns([1, 3])
                        with col_btn:
                            if st.button(f"🎯 Select for CV Analysis #{i}", key=f"btn_job_{i}"):
                                st.session_state["selected_target_job"] = job
                                st.success(f"Selected '{job['jobtitle']}'! Now switch to Tab 3 (⚡ CV Optimizer).")

                        with st.expander(f"View Job Description Snippet #{i}"):
                            st.write(job.get('snippet', 'N/A'))

            else:
                st.warning("No job matches retrieved. Ensure the FAISS index exists in vectorstore/jobs_faiss.")

        # ---------------------------------------------------------------------
        # Tab 3: CV Optimizer (Module 3)
        # ---------------------------------------------------------------------
        with tab3:
            st.subheader("AI CV Optimization & Improvement")

            target_job = st.session_state.get("selected_target_job")
            if target_job:
                st.info(f"Target Job Selected: **{target_job.get('jobtitle')}** at **{target_job.get('company')}**")

                if st.button("✨ Generate CV Improvement Suggestions", type="primary"):
                    with st.spinner("Analyzing resume gaps against target job requirements..."):
                        try:
                            suggestions = generate_cv_suggestions(profile, target_job)
                            st.session_state["cv_suggestions"] = suggestions
                        except Exception as err:
                            st.warning("Notice: Unable to reach Gemini cloud AI. Please check your API quota or retry later.")

                suggestions = st.session_state.get("cv_suggestions")
                if suggestions:
                    if suggestions.get("error"):
                        err_text = str(suggestions.get("error"))
                        if "429" in err_text or "Quota" in err_text or "ResourceExhausted" in err_text:
                            st.warning("⚡ AI generation is temporarily limited due to API quota. Your local profile analysis and job matching remain fully active!")
                        else:
                            st.warning("Notice: Processing error encountered during CV improvement generation.")
                    else:
                        col_a, col_b = st.columns([1, 1.2])

                        with col_a:
                            st.markdown("### 🎯 Missing / Insufficient Skills")
                            missing = suggestions.get("missing_skills", [])
                            if missing:
                                for m_skill in missing:
                                    st.warning(f"⚠️ {m_skill}")
                            else:
                                st.success("No critical missing skills detected for this position!")

                        with col_b:
                            st.markdown("### 📝 Tailored Professional Summary")
                            summary = suggestions.get("rewritten_summary", "")
                            st.success(summary)

                        st.markdown("---")
                        st.markdown("### 🔍 Weak Bullet Points Analysis & Rewrites")
                        weak_bullets = suggestions.get("weak_bullet_points", [])
                        if weak_bullets:
                            for idx, item in enumerate(weak_bullets, 1):
                                st.markdown(f"""
                                <div class="bullet-card">
                                    <h5 style="margin-top:0; color:#1E293B;">Bullet #{idx} Analysis</h5>
                                    <p><strong>Original Bullet:</strong><br><code>{item.get('original_bullet')}</code></p>
                                    <p style="color:#B45309;"><strong>Why It Is Weak:</strong> {item.get('reason_weak')}</p>
                                    <p style="color:#15803D;"><strong>Improved Version:</strong><br><code>{item.get('improved_bullet')}</code></p>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No weak bullet points identified.")
            else:
                st.warning("Please select a target job from Tab 2 first.")

        # ---------------------------------------------------------------------
        # Tab 4: AI Career Mentor Chat (Modules 4 & 5)
        # ---------------------------------------------------------------------
        with tab4:
            st.subheader("AI Career Mentor (Grounded RAG Assistant)")
            st.caption("Answers career and resume questions grounded strictly in curated career notes. Safety guardrails are enforced before AI processing.")

            # Chat Session State
            if "messages" not in st.session_state:
                st.session_state["messages"] = [
                    {"role": "assistant", "content": "Hello! I am your AI Career Mentor. Ask me any career, resume, or job interview questions grounded in our career knowledge base!"}
                ]

            # Quick Prompt Assistance Buttons
            st.markdown("**Suggested Quick Questions:**")
            q1, q2, q3 = st.columns(3)
            quick_question = None
            if q1.button("💡 Data Analyst Skills"):
                quick_question = "What core skills do I need to become a Data Analyst?"
            if q2.button("📝 Resume Writing Tips"):
                quick_question = "How should I structure my resume summary and experience?"
            if q3.button("🎯 Interview Prep Advice"):
                quick_question = "What are the best interview preparation strategies for tech roles?"

            # Render Chat Messages
            for msg in st.session_state["messages"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            # User Prompt Input Handling
            user_prompt = st.chat_input("Ask your career question...") or quick_question

            if user_prompt:
                # Display user message
                st.session_state["messages"].append({"role": "user", "content": user_prompt})
                with st.chat_message("user"):
                    st.markdown(user_prompt)

                # MODULE 5: Apply Guardrails Check BEFORE calling LLM
                is_valid, guardrail_msg = validate_question(user_prompt)

                if not is_valid:
                    # Guardrail blocked the input
                    blocked_response = f"🛡️ **Guardrail Protection:** {guardrail_msg}"
                    st.session_state["messages"].append({"role": "assistant", "content": blocked_response})
                    with st.chat_message("assistant"):
                        st.warning(blocked_response)
                else:
                    # MODULE 4: Call RAG Career Mentor
                    with st.chat_message("assistant"):
                        with st.spinner("Consulting career notes knowledge base..."):
                            try:
                                mentor_res = ask_mentor(user_prompt)
                                raw_answer = mentor_res.get("answer", "I don't know based on the provided career notes.")
                                sources = mentor_res.get("sources", [])

                                if "429" in raw_answer or "Quota" in raw_answer or "ResourceExhausted" in raw_answer:
                                    answer = "⚡ The AI mentor is temporarily unavailable due to API rate limits. Please try asking again shortly!"
                                else:
                                    answer = raw_answer

                                if sources and "429" not in raw_answer:
                                    answer += f"\n\n*Sources Consulted:* `{', '.join(sources)}`"

                                st.markdown(answer)
                                st.session_state["messages"].append({"role": "assistant", "content": answer})
                            except Exception as err:
                                err_text = str(err)
                                if "429" in err_text or "Quota" in err_text or "ResourceExhausted" in err_text:
                                    friendly_err = "⚡ AI mentor generation is temporarily limited due to API rate limits."
                                else:
                                    friendly_err = "Notice: Processing issue consulting Career Mentor."
                                st.warning(friendly_err)
                                st.session_state["messages"].append({"role": "assistant", "content": friendly_err})

else:
    st.info("👈 Please select or upload a resume from the sidebar to begin.")
