import argparse

from dotenv import load_dotenv

from .pipeline import ResearchPipeline


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

    ask = sub.add_parser("ask", help="Ask a question against an existing index")
    ask.add_argument("question")
    ask.add_argument("--no-reranker", action="store_true")
    ask.add_argument("--top-k", type=int, default=8)
    ask.add_argument("--index-dir", default="data/index")

    args = parser.parse_args()

    if args.command == "index":
        pipeline = ResearchPipeline(index_dir=args.index_dir)
        count = pipeline.index(args.directory)
        print(f"Indexed {count} chunks into {args.index_dir}.")

    elif args.command == "ask":
        pipeline = ResearchPipeline(
            use_reranker=not args.no_reranker,
            index_dir=args.index_dir,
        )
        count = pipeline.load_index()
        print(f"Loaded {count} chunks from {args.index_dir}.")
        print(pipeline.ask(args.question, top_k=args.top_k))


if __name__ == "__main__":
    main()
