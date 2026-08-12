"""Module 2 — Semantic job search.

Embeds job postings into a FAISS vector index (saved under config.JOBS_INDEX_DIR),
and provides top-N similarity search functionality for candidate profiles or text queries.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src import config
from src.search.embed import get_embedding, get_embeddings_batch


class GeminiEmbeddingsWrapper(Embeddings):
    """LangChain-compatible Embeddings wrapper utilizing src.search.embed module."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document strings using Gemini embeddings batch API."""
        return get_embeddings_batch(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single search query string using Gemini embedding API."""
        return get_embedding(text)


def prepare_job_documents(
    csv_path: Optional[Path] = None,
    max_jobs: Optional[int] = 1000
) -> List[Document]:
    """Load job postings from CSV and prepare LangChain Document objects with metadata.

    Args:
        csv_path (Optional[Path]): Path to job CSV file (defaults to config.JOBS_CSV).
        max_jobs (Optional[int]): Max number of job rows to process (default: 1000 for fast embedding).

    Returns:
        List[Document]: List of Document objects containing combined job text and metadata.
    """
    file_path = csv_path or config.JOBS_CSV
    if not file_path.exists():
        raise FileNotFoundError(
            f"Job CSV file not found at '{file_path}'. "
            f"Please run 'python download_data.py' first."
        )

    # Read dataset
    df = pd.read_csv(file_path)

    # Fill missing values in specified text columns
    for col in config.JOB_TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
        else:
            df[col] = ""

    # Slice dataframe if max_jobs is specified
    if max_jobs and len(df) > max_jobs:
        df = df.head(max_jobs)

    documents = []
    for idx, row in df.iterrows():
        title = row.get("jobtitle", "").strip()
        skills = row.get("skills", "").strip()
        desc = row.get("jobdescription", "").strip()

        # Combine text fields into a single rich text block for embedding
        combined_text = f"Job Title: {title}\nSkills: {skills}\nJob Description: {desc}".strip()

        if not combined_text or combined_text == "Job Title:\nSkills:\nJob Description:":
            continue

        # Metadata dictionary attached to each FAISS vector item
        metadata = {
            "jobtitle": str(row.get("jobtitle", "N/A")),
            "company": str(row.get("company", "N/A")),
            "skills": str(row.get("skills", "N/A")),
            "location": str(row.get("joblocation_address", "N/A")),
            "experience": str(row.get("experience", "N/A")),
            "payrate": str(row.get("payrate", "N/A")),
            "jobid": str(row.get("jobid", idx)),
        }

        doc = Document(page_content=combined_text, metadata=metadata)
        documents.append(doc)

    return documents


def build_and_save_faiss_index(
    csv_path: Optional[Path] = None,
    index_save_dir: Optional[Path] = None,
    max_jobs: Optional[int] = 1000
) -> FAISS:
    """Build FAISS vector index from job dataset and save it locally.

    Args:
        csv_path (Optional[Path]): Job CSV path.
        index_save_dir (Optional[Path]): Save directory for FAISS index (defaults to config.JOBS_INDEX_DIR).
        max_jobs (Optional[int]): Number of jobs to index.

    Returns:
        FAISS: Constructed FAISS vectorstore instance.
    """
    save_dir = index_save_dir or config.JOBS_INDEX_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading job records from CSV (max_jobs={max_jobs})...")
    docs = prepare_job_documents(csv_path=csv_path, max_jobs=max_jobs)
    print(f"Prepared {len(docs)} documents. Generating Gemini embeddings and building FAISS index...")

    embeddings = GeminiEmbeddingsWrapper()
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Save FAISS index locally
    vectorstore.save_local(str(save_dir))
    print(f"FAISS job index saved successfully to '{save_dir}'.")
    return vectorstore


def load_faiss_index(index_dir: Optional[Path] = None) -> FAISS:
    """Load an existing FAISS job index from disk.

    Args:
        index_dir (Optional[Path]): Saved FAISS index directory.

    Returns:
        FAISS: Loaded FAISS vectorstore instance.
    """
    target_dir = index_dir or config.JOBS_INDEX_DIR
    if not target_dir.exists():
        raise FileNotFoundError(
            f"FAISS index directory not found at '{target_dir}'. "
            f"Please build the index first using build_and_save_faiss_index() or notebook 02."
        )

    embeddings = GeminiEmbeddingsWrapper()
    return FAISS.load_local(
        str(target_dir),
        embeddings,
        allow_dangerous_deserialization=True
    )


def search_jobs(
    query_or_profile: Union[str, Dict[str, Any]],
    top_n: int = config.TOP_N_JOBS,
    index_dir: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """Perform semantic job search for a query string or parsed candidate profile dictionary.

    Args:
        query_or_profile (Union[str, Dict[str, Any]]): Text search query OR parsed profile dict.
        top_n (int): Number of top matching job results to return.
        index_dir (Optional[Path]): Directory path of FAISS index.

    Returns:
        List[Dict[str, Any]]: List of top matching jobs with similarity score and metadata.
    """
    # Load index from local disk
    vectorstore = load_faiss_index(index_dir)

    # Construct search query string from input
    if isinstance(query_or_profile, dict):
        role = query_or_profile.get("target_role", "")
        skills_raw = query_or_profile.get("skills", [])
        skills = ", ".join(skills_raw) if isinstance(skills_raw, list) else str(skills_raw)
        exp_raw = query_or_profile.get("experience", [])
        experience = " ".join(exp_raw) if isinstance(exp_raw, list) else str(exp_raw)

        search_query = f"Target Role: {role}\nSkills: {skills}\nExperience: {experience}".strip()
    else:
        search_query = str(query_or_profile).strip()

    if not search_query:
        return []

    # Run vector similarity search with distance score
    results_with_scores = vectorstore.similarity_search_with_score(search_query, k=top_n)

    matched_jobs = []
    for doc, score in results_with_scores:
        job_info = {
            "score": float(score),
            "jobtitle": doc.metadata.get("jobtitle", "N/A"),
            "company": doc.metadata.get("company", "N/A"),
            "skills": doc.metadata.get("skills", "N/A"),
            "location": doc.metadata.get("location", "N/A"),
            "experience": doc.metadata.get("experience", "N/A"),
            "payrate": doc.metadata.get("payrate", "N/A"),
            "snippet": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
        }
        matched_jobs.append(job_info)

    return matched_jobs
