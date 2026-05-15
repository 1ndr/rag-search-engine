import argparse
from lib.semantic_search import verify_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_model_parser = subparsers.add_parser("verify", help="Verify that the embedded model is loaded")

    args = parser.parse_args()
    match args.command:
        case "verify":
            print("Verifying model used for Semantic Search")
            verify_model()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
