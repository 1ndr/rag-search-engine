import argparse

from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    search_command,
    chunk_command
)

from lib.search_utils import (
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_CHUNK_SIZE
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_model_parser = subparsers.add_parser("verify", help="Verify that the embedded model is loaded")

    embed_text_parser = subparsers.add_parser("embed_text", help="embed the given text")
    embed_text_parser.add_argument("text", type=str, help="The text to be embeded")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify embeddings for the movie dataset")

    embed_query_parser = subparsers.add_parser("embed_query", help="Embed the given query")
    embed_query_parser.add_argument("query", type=str, help="query to be embedded")

    search_command_parser = subparsers.add_parser("search", help="Search for a given query using Semantic Search")
    search_command_parser.add_argument("query", type=str, help="Query to be searched for")
    search_command_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="optional limit to seraches")

    chunk_command_parser = subparsers.add_parser("chunk", help="Chunking text")
    chunk_command_parser.add_argument("text", type=str, help="Text to be chunked")
    chunk_command_parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="optional limit for chunk size")

    args = parser.parse_args()
    match args.command:
        case "verify":
            print("Verifying model used for Semantic Search")
            verify_model()

        case "embed_text":
            print(f'Embedding text: "{args.text}"')
            embed_text(args.text)

        case "verify_embeddings":
            print(f"Verifying embeddings...")
            verify_embeddings()

        case "embed_query":
            print(f"Embedding text for '{args.query}'")
            embed_query_text(args.query)

        case "search":
            print(f"Searching for '{args.query}' with the limit: {args.limit}")
            results = search_command(args.query, args.limit)

            for i, (score, res) in enumerate(results, 1):
                print(f"{i}. {res['title']} (score: {res['score']:.4f})\n{res['description']}")

        case "chunk":
            print(f"Preparing Chunking process for '{args.text}':")
            print(f"Chunking {len(args.text)} characters")
            chunks = chunk_command(args.text, args.chunk_size)
            for i, chunk in enumerate(chunks, 1):
                print(f"{i}. {" ".join(chunk)}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
