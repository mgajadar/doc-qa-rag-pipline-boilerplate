# ingestion.py
import os
import pickle
from pathlib import Path
from typing import List, Dict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import (
    DATA_DIR,
    INDEX_PATH,
    METADATA_PATH,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    FAISS_HNSW_M,
    FAISS_HNSW_EF_CONSTRUCTION,
)


def loadDocuments(dataDir: Path = DATA_DIR) -> List[Dict]:
    """Load all .txt and .md docs from data/."""
    documents: List[Dict] = []

    for root, _, files in os.walk(dataDir):
        for fileName in files:
            if not fileName.lower().endswith((".txt", ".md")):
                continue

            filePath = Path(root) / fileName
            with open(filePath, "r", encoding="utf-8") as f:
                text = f.read()

            documents.append(
                {
                    "text": text,
                    "source": str(filePath.relative_to(dataDir)),
                }
            )

    return documents


def chunkText(text: str, chunkSize: int, chunkOverlap: int) -> List[str]:
    """Simple char-based sliding-window chunking."""
    chunks: List[str] = []
    start = 0
    textLength = len(text)

    while start < textLength:
        end = min(start + chunkSize, textLength)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == textLength:
            break

        start = end - chunkOverlap
        if start < 0:
            start = 0

    return chunks


def makeChunks(documents: List[Dict]) -> List[Dict]:
    """Create chunk-level records with source + chunk id."""
    chunkRecords: List[Dict] = []

    for doc in documents:
        chunks = chunkText(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        for idx, chunk in enumerate(chunks):
            chunkRecords.append(
                {
                    "text": chunk,
                    "source": doc["source"],
                    "chunkId": idx,
                }
            )

    return chunkRecords


def embedTexts(texts: List[str], modelName: str = EMBEDDING_MODEL_NAME):
    """Embed a list of texts with BGE, returning numpy float32 array + model."""
    print(f"[ingestion] loading embedding model: {modelName}")
    model = SentenceTransformer(modelName)

    # normalize_embeddings=True gives cosine-ready unit vectors
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    return embeddings, model


def buildFaissHnswIndex(embeddings: np.ndarray) -> faiss.IndexHNSWFlat:
    """Build an HNSW index (inner product) over the embeddings."""
    dim = embeddings.shape[1]
    print(f"[ingestion] building HNSW index (dim={dim}, M={FAISS_HNSW_M})")

    index = faiss.IndexHNSWFlat(dim, FAISS_HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = FAISS_HNSW_EF_CONSTRUCTION

    # embeddings already normalized; just add
    index.add(embeddings)

    return index


def runIngestionPipeline():
    print("[ingestion] loading documents...")
    documents = loadDocuments()

    if not documents:
        raise RuntimeError(f"No .txt or .md documents found under {DATA_DIR}")

    print(f"[ingestion] loaded {len(documents)} documents")

    print("[ingestion] chunking...")
    chunks = makeChunks(documents)
    print(f"[ingestion] created {len(chunks)} chunks")

    texts = [c["text"] for c in chunks]

    print("[ingestion] embedding chunks with BGE...")
    embeddings, _ = embedTexts(texts)

    print("[ingestion] building FAISS-HNSW index...")
    index = buildFaissHnswIndex(embeddings)

    print(f"[ingestion] saving index to {INDEX_PATH}")
    faiss.write_index(index, str(INDEX_PATH))

    print(f"[ingestion] saving metadata to {METADATA_PATH}")
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print("[ingestion] done.")
