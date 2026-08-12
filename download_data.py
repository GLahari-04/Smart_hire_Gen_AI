"""
Download the SmartHire GenAI job dataset from Kaggle into data/jobs/.

Run this ONCE to set up your job corpus:

    pip install kagglehub
    python download_data.py

The first run asks for your Kaggle credentials (username + API key).
Get them at: Kaggle -> profile -> Settings -> API -> Create New Token
(that downloads kaggle.json; your username and key are inside it).

After it finishes, data/jobs/ will contain:
    naukri_com-job_sample.csv       job postings you embed into FAISS

Sample resumes go in data/resumes/ (add your own PDFs/DOCX) and the mentor's
career notes go in data/career_notes/ (two starter notes are already there).
"""

import glob
import shutil
from pathlib import Path

import kagglehub

JOBS = Path(__file__).resolve().parent / "data" / "jobs"
JOBS.mkdir(parents=True, exist_ok=True)

# A single-CSV job dataset -> copied flat into data/jobs/.
JOB_DATASET = "PromptCloudHQ/jobs-on-naukricom"


def main():
    print(f"Downloading {JOB_DATASET} ...")
    cache_dir = kagglehub.dataset_download(JOB_DATASET)
    csvs = glob.glob(str(Path(cache_dir) / "**" / "*.csv"), recursive=True)
    if not csvs:
        print(f"  WARNING: no CSV found (files are in {cache_dir})")
        return
    for csv in csvs:
        shutil.copy(csv, JOBS / Path(csv).name)
        print(f"  copied -> {Path(csv).name}")

    print("\nDone. data/jobs/ now contains:")
    for f in sorted(JOBS.glob("*.csv")):
        print(" -", f.name)


if __name__ == "__main__":
    main()
