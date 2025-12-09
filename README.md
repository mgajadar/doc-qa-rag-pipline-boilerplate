# RAG Document QA (BGE + FAISS-HNSW)

A small, self-contained retrieval-augmented generation (RAG) pipeline for
question answering over local documents.

- BGE embeddings (`BAAI/bge-base-en-v1.5`)
- FAISS-HNSW index
- Optional BGE reranker (`BAAI/bge-reranker-base`)
- OpenAI chat model for generation
- CLI for ingestion + QA
- Basic evaluation script with a toy F1 metric

## setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your_key_here"
