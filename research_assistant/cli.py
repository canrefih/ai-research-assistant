import argparse

import requests
from dotenv import load_dotenv

from .pipeline import ResearchPipeline
from .web_search import TavilySearchProvider


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Citation-aware AI research assistant"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    index = sub.add_parser("index", help="Index Markdown/text documents")
    index.add_argument("directory", help="Directory containing .md/.txt files")
    index.add_argument(
        "--index-dir",
        default="data/index",
        help="Directory used to store the local index",
    )
    index.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence Transformer model used for semantic retrieval",
    )

    ask = sub.add_parser("ask", help="Ask a question against an existing index")
    ask.add_argument("question")
    ask.add_argument("--no-reranker", action="store_true")
    ask.add_argument("--top-k", type=int, default=8)
    ask.add_argument("--index-dir", default="data/index")
    ask.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence Transformer model used for semantic retrieval",
    )
    ask.add_argument(
        "--reranker-model",
        default="cross-encoder/ms-marco-MiniLM-L6-v2",
        help="CrossEncoder model used for reranking",
    )
    ask.add_argument(
        "--web",
        action="store_true",
        help="Include web search results in the evidence",
    )

    index_url = sub.add_parser(
        "index-url",
        help="Index a web page from a URL",
    )
    index_url.add_argument(
        "url",
        help="Web page URL to index",
    )
    index_url.add_argument(
        "--index-dir",
        default="data/index",
        help="Directory used to store the local index",
    )
    index_url.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence Transformer model used for semantic retrieval",
    )

    args = parser.parse_args()

    try:
        if args.command == "index":
            pipeline = ResearchPipeline(
                index_dir=args.index_dir,
                embedding_model=args.embedding_model,
            )
            count = pipeline.index(args.directory)
            print(f"Indexed {count} chunks into {args.index_dir}.")

        elif args.command == "index-url":
            pipeline = ResearchPipeline(
                index_dir=args.index_dir,
                embedding_model=args.embedding_model,
            )
            count = pipeline.index_url(args.url)
            print(f"Indexed {count} chunks from {args.url} into {args.index_dir}.")

        elif args.command == "ask":
            pipeline_kwargs = {
                "use_reranker": not args.no_reranker,
                "index_dir": args.index_dir,
                "embedding_model": args.embedding_model,
                "reranker_model": args.reranker_model,
            }

            if args.web:
                pipeline_kwargs["web_search_provider"] = TavilySearchProvider()

            pipeline = ResearchPipeline(**pipeline_kwargs)

            count = pipeline.load_index()
            print(f"Loaded {count} chunks from {args.index_dir}.")
            print(
                pipeline.ask(
                    args.question,
                    top_k=args.top_k,
                    use_web_search=args.web,
                )
            )

    except (
        ValueError,
        FileNotFoundError,
        RuntimeError,
        requests.RequestException,
    ) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()