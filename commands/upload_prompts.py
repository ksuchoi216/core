import argparse
from dotenv import find_dotenv, load_dotenv

from core.llm.langfuse import upload_prompts_from_local

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload prompts to Langfuse")
    parser.add_argument(
        "-l",
        "--labels",
        nargs="*",
        default=["production"],
        help="Labels for the uploaded prompts (default: production)",
    )
    args = parser.parse_args()

    load_dotenv(find_dotenv(usecwd=True))
    upload_prompts_from_local(labels=args.labels)
