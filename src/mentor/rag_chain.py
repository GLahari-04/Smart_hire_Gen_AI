"""Module 4 — AI Career Mentor (RAG).

A RAG pipeline that answers career questions grounded in the career-notes
FAISS index (config.NOTES_INDEX_DIR): retrieves the top-K relevant chunks,
stuffs them into the mentor prompt, and generates grounded responses using Gemini.
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import google.generativeai as genai
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src import config
from src.generate import prompts
from src.parsing import loader
from src.search.job_search import GeminiEmbeddingsWrapper, load_faiss_index

try:
    from google.api_core.exceptions import ResourceExhausted
except ImportError:
    ResourceExhausted = Exception

# Load environment variables from a local .env file if available
load_dotenv()


def _get_configured_model() -> genai.GenerativeModel:
    """Helper function to verify API key configuration and return a Gemini model instance."""
    api_key = os.getenv(config.API_KEY_ENV)
    if not api_key:
        raise ValueError(
            f"Gemini API key not found. Please set the '{config.API_KEY_ENV}' "
            f"environment variable in your .env file."
        )

    genai.configure(api_key=api_key)

    # Low temperature for deterministic, strictly-grounded generation
    generation_config = {
        "temperature": 0.1,
    }

    return genai.GenerativeModel(
        model_name=config.CHAT_MODEL,
        generation_config=generation_config
    )


def build_and_save_notes_index(
    notes_dir: Optional[Path] = None,
    save_dir: Optional[Path] = None
) -> FAISS:
    """Load career notes documents, create embeddings, build FAISS index, and save locally.

    Args:
        notes_dir (Optional[Path]): Directory containing career notes (defaults to config.CAREER_NOTES_DIR).
        save_dir (Optional[Path]): Target save directory for FAISS index (defaults to config.NOTES_INDEX_DIR).

    Returns:
        FAISS: Built FAISS vectorstore instance.
    """
    target_notes_dir = notes_dir or config.CAREER_NOTES_DIR
    target_save_dir = save_dir or config.NOTES_INDEX_DIR
    target_save_dir.mkdir(parents=True, exist_ok=True)

    if not target_notes_dir.exists():
        raise FileNotFoundError(f"Career notes directory not found at '{target_notes_dir}'.")

    # Load and chunk all notes files in directory using loader module
    chunks_with_source = loader.load_folder(target_notes_dir)
    if not chunks_with_source:
        raise ValueError(f"No valid markdown/text files found in '{target_notes_dir}'.")

    # Convert chunks to LangChain Document objects with metadata
    docs = []
    for chunk_text, source_name in chunks_with_source:
        doc = Document(page_content=chunk_text, metadata={"source": source_name})
        docs.append(doc)

    print(f"Loaded {len(docs)} text chunks from career notes. Building FAISS vector index...")
    embeddings = GeminiEmbeddingsWrapper()
    vectorstore = FAISS.from_documents(docs, embeddings)

    vectorstore.save_local(str(target_save_dir))
    print(f"Career notes FAISS index saved successfully to '{target_save_dir}'.")
    return vectorstore


def load_notes_index(save_dir: Optional[Path] = None) -> FAISS:
    """Load existing career notes FAISS index from disk.

    Args:
        save_dir (Optional[Path]): Directory where FAISS index is saved.

    Returns:
        FAISS: Loaded FAISS vectorstore instance.
    """
    target_dir = save_dir or config.NOTES_INDEX_DIR
    if not target_dir.exists():
        # Automatically build index if it does not exist yet
        return build_and_save_notes_index(save_dir=target_dir)

    embeddings = GeminiEmbeddingsWrapper()
    return FAISS.load_local(
        str(target_dir),
        embeddings,
        allow_dangerous_deserialization=True
    )


def ask_mentor(
    question: str,
    top_k: int = config.TOP_K_NOTES,
    index_dir: Optional[Path] = None,
    top_k_jobs: int = 2
) -> Dict[str, Any]:
    """Query the AI Career Mentor using hybrid RAG over career notes and job postings.

    Args:
        question (str): The career or resume question from the candidate.
        top_k (int): Number of relevant note chunks to retrieve (defaults to config.TOP_K_NOTES).
        index_dir (Optional[Path]): Optional path to notes FAISS index directory.
        top_k_jobs (int): Number of job postings to retrieve from job market index (default: 2).

    Returns:
        Dict[str, Any]: Dictionary containing:
            - "question": str
            - "answer": str
            - "sources": List[str] (names of source files and job postings used)
            - "retrieved_chunks": List[str] (content of retrieved chunks)
    """
    if not question or not question.strip():
        return {
            "question": "",
            "answer": "Please ask a valid career question.",
            "sources": [],
            "retrieved_chunks": []
        }

    context_blocks = []
    sources = set()
    chunks = []

    # 1. Retrieve top K relevant chunks from Career Notes FAISS index
    try:
        notes_vectorstore = load_notes_index(index_dir)
        retrieved_notes = notes_vectorstore.similarity_search(question.strip(), k=top_k)
        for doc in retrieved_notes:
            source_file = doc.metadata.get("source", "Career Note")
            sources.add(source_file)
            chunks.append(doc.page_content)
            context_blocks.append(f"[Career Note: {source_file}]\n{doc.page_content}")
    except Exception as err:
        print(f"Notice: Career notes index lookup: {err}")

    # 2. Retrieve top K relevant job posting chunks from Jobs FAISS index
    try:
        jobs_vectorstore = load_faiss_index()
        retrieved_jobs = jobs_vectorstore.similarity_search(question.strip(), k=top_k_jobs)
        for doc in retrieved_jobs:
            job_title = doc.metadata.get("jobtitle", "Job Posting")
            company = doc.metadata.get("company", "Company")
            skills = doc.metadata.get("skills", "N/A")
            snippet = doc.page_content[:300] if len(doc.page_content) > 300 else doc.page_content
            
            job_label = f"Job: {job_title} at {company}"
            sources.add(job_label)
            
            job_text = f"Title: {job_title}\nCompany: {company}\nRequired Skills: {skills}\nDetails: {snippet}"
            chunks.append(job_text)
            context_blocks.append(f"[Job Posting: {job_title} at {company}]\n{job_text}")
    except Exception as err:
        print(f"Notice: Jobs index lookup: {err}")

    if not context_blocks:
        return {
            "question": question,
            "answer": "I don't know based on the provided career notes and job market context.",
            "sources": [],
            "retrieved_chunks": []
        }

    context_str = "\n\n".join(context_blocks)

    # 3. Format hybrid RAG system prompt
    prompt = prompts.MENTOR_SYSTEM_PROMPT.format(
        context=context_str,
        question=question.strip()
    )

    # 4. Generate answer using Gemini
    model = _get_configured_model()

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            answer_text = response.text.strip() if response and response.text else "I don't know based on the provided career notes and job market context."

            return {
                "question": question,
                "answer": answer_text,
                "sources": sorted(list(sources)),
                "retrieved_chunks": chunks
            }

        except (ResourceExhausted, Exception) as error:
            err_msg = str(error)
            if ("429" in err_msg or "ResourceExhausted" in err_msg or "quota" in err_msg) and attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
            else:
                print(f"Error generating mentor answer: {error}")
                return {
                    "question": question,
                    "answer": f"An error occurred while generating answer: {error}",
                    "sources": sorted(list(sources)),
                    "retrieved_chunks": chunks
                }

    return {
        "question": question,
        "answer": "Failed to generate answer after maximum retries.",
        "sources": [],
        "retrieved_chunks": []
    }
