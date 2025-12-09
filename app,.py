# app.py
import argparse

from config import validateConfig
from ingestion import runIngestionPipeline
from rag_pipeline import RagPipeline


def runIngest():
    validateConfig()
    runIngestionPipeline()


def runAsk(query: str, topK: int):
    validateConfig()
    pipeline = RagPipeline.fromArtifacts()
    result = pipeline.answer(query, topK=topK)

    print("\n=== answer ===\n")
    print(result["answer"])
    print("\n=== sources ===\n")
    for c in result["context"]:
        scorePart = f"{c.get('score', 0.0):.3f}"
        rerankPart = c.get("rerankScore")
        if rerankPart is not None:
            print(
                f"- {c['source']} (chunk {c['chunkId']}, "
                f"vecScore={scorePart}, rerankScore={rerankPart:.3f})"
            )
        else:
            print(f"- {c['source']} (chunk {c['chunkId']}, vecScore={scorePart})")


def main():
    parser = argparse.ArgumentParser(description="RAG document QA pipeline (CLI)")
    subparsers = parser.add_subparsers(dest="command")

    ingestParser = subparsers.add_parser("ingest", help="ingest and index documents")

    askParser = subparsers.add_parser("ask", help="ask a question over docs")
    askParser.add_argument("--query", type=str, required=True, help="question to ask")
    askParser.add_argument(
        "--topK", type=int, default=5, help="number of chunks to retrieve"
    )

    args = parser.parse_args()

    if args.command == "ingest":
        runIngest()
    elif args.command == "ask":
        runAsk(args.query, args.topK)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
