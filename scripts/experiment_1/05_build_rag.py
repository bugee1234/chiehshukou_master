from __future__ import annotations

import argparse

from run_1a import stage04_rag


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp1-1A Stage 05: build RAG retrieval file")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--chunk-words", type=int, default=120)
    parser.add_argument("--overlap-words", type=int, default=30)
    parser.add_argument("--embed-model", type=str, default="text-embedding-3-small")
    args = parser.parse_args()
    stage04_rag(args.top_k, args.chunk_words, args.overlap_words, args.embed_model)


if __name__ == "__main__":
    main()
