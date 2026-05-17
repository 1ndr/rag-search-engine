import argparse
from lib.semantic_search import verify_model, embed_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_model_parser = subparsers.add_parser("verify", help="Verify that the embedded model is loaded")

    embed_text_parser = subparsers.add_parser("embed_text", help="embed the given text")
    embed_text_parser.add_argument("text", type=str, help="The text to be embeded")

    args = parser.parse_args()
    match args.command:
        case "verify":
            print("Verifying model used for Semantic Search")
            verify_model()

        case "embed_text":
            print(f'Embedding text: "{args.text}"')
            embed_text(args.text)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
