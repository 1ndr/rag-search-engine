import argparse

from lib.keyword_search import (
        search_command,
        build_command,
        tf_command,
        idf_command,
        tf_idf_command
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build the inverted index")
    
    tf_parser = subparsers.add_parser("tf", help="Returns the Term Frequency of a token in a doc")
    tf_parser.add_argument("doc_id", type=int, help="document id")
    tf_parser.add_argument("token", type=str, help="token to be searched for")

    idf_parser = subparsers.add_parser("idf", help="Returns the Inverse Document Frequency of a token")
    idf_parser.add_argument("term", type=str, help="term to be searched for")

    tf_idf_parser = subparsers.add_parser("tfidf", help="Returns the Term Freqeuncy - Inverse Document Freqeuncy score of a token in a doc")
    tf_idf_parser.add_argument("doc_id", type=int, help="document id")
    tf_idf_parser.add_argument("term", type=str, help="term to be serached for")

    args = parser.parse_args()

    match args.command:
        case "search":
            print("Searching for:", args.query)
            results = search_command(args.query)

            for i, res in enumerate(results, 1):
                print(f"{i}. ({res['id']}) {res['title']}")

        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")

        case "tf":
            print(f"Finding Term Freqeuncy for: '{args.token}' in document: {args.doc_id}")
            frequency = tf_command(args.doc_id, args.token)
            print(f"Term Frequency for '{args.token}' in document {args.doc_id}: {frequency}")

        case "idf":
            print(f"Finding Inverse Document Frqeuency for '{args.term}'")
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            print(f"Finding Term Frequency - Inverse Document Frequency score for '{args.term}'")
            tf_idf = tf_idf_command(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()

