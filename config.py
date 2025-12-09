# config.py
import os
from pathlib import Path

# base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

INDEX_PATH = ARTIFACTS_DIR / "faiss_hnsw.index"
METADATA_PATH = ARTIFACTS_DIR / "metadata.pkl"

# models
# BGE encoder for embeddings
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"

# optional reranker (BGE cross-encoder)
USE_RERANKER = True
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

# llm model (change if you want)
LLM_MODEL_NAME = "gpt-4o-mini"

# chunking
CHUNK_SIZE = 800        # characters per chunk
CHUNK_OVERLAP = 200     # overlapping chars

# faiss HNSW params
FAISS_HNSW_M = 32
FAISS_HNSW_EF_CONSTRUCTION = 200
FAISS_HNSW_EF_SEARCH = 64

# env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def validateConfig():
    """Basic sanity checks before running anything."""
    if OPENAI_API_KEY is None:
        raise ValueError("OPENAI_API_KEY is not set in environment variables.")

    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not any(DATA_DIR.glob("**/*.txt")) and not any(DATA_DIR.glob("**/*.md")):
        print(f"[config] warning: no .txt or .md files found in {DATA_DIR}")
