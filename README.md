# OpenQA RAG Pipeline

This repository contains the retrieval-augmented generation (RAG) pipeline and local vector database tooling built on ChromaDB and HotpotQA embeddings (`BAAI/bge-large-en-v1.5`).

---

## 1. Prerequisites

- **Python 3.10+**
- **Hugging Face Account** with access to the `NLP-CW/OpenQA` dataset repository
- *(Optional)* **CUDA-capable GPU** if you plan to generate new embeddings locally

---

## 2. Environment Setup

Clone the repository and create a virtual environment:

```bash
git clone <your-repo-url>
cd OpenQA

# Create and activate virtual environment
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## 3. Hugging Face Authentication

Because the pre-computed vector dataset is hosted privately on Hugging Face, you need to authenticate locally before downloading it.

1. Generate a **Read** access token from your [Hugging Face Token Settings](https://huggingface.co/settings/tokens).
2. Log in via your terminal:

```bash
hf auth login
```
*(Paste your `hf_...` token when prompted).*

---

## 4. Restore the Local Vector Database

Instead of re-embedding all 500,000+ chunks from scratch, pull the pre-computed Parquet shards from Hugging Face and populate your local ChromaDB instance:

```bash
python download_and_restore.py
```

This script will download the dataset from `NLP-CW/OpenQA` and create a local `./vectordb` folder containing the `hotpot` collection.

---

## 5. Updating & Uploading New Vectors (Maintainers Only)

If you add new chunks to your local `./vectordb` instance and want to sync the updated collection with the team:

1. Ensure you are logged in with a **Write** access token (`hf auth login`).
2. Run the batched export and upload script:

```bash
python upload.py
```

This will export your local Chroma collection into sharded Parquet files and push them to the `data/` directory of `NLP-CW/OpenQA`.