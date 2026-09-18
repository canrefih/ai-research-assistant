import argparse
from dotenv import load_dotenv
from .pipeline import ResearchPipeline

def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="AI Research Assistant")
    sub = parser.add_subparsers(dest="command", required=True)

    index = sub.add_parser("index")
    index.add_argument("directory")

    ask = sub.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--no-reranker", action="store_true")

    args = parser.parse_args()

    if args.command == "index":
        pipeline = ResearchPipeline()
        count = pipeline.index(args.directory)
        print(f"Indexed {count} chunks.")
    elif args.command == "ask":
        pipeline = ResearchPipeline(use_reranker=not args.no_reranker)
        pipeline.index("data/sample")
        print(pipeline.ask(args.question))

if __name__ == "__main__":
    main()
