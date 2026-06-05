import argparse
from dotenv import find_dotenv, load_dotenv

from core.langfuse import create_prompt_keys_from_local_prompt_dir
from core.langfuse.prompt import DEFAULT_LOCAL_PROMPT_DIR, DEFAULT_PROMPT_KEYS_PATH

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create prompt keys from local prompt directory"
    )
    parser.add_argument(
        "-d",
        "--prompt-dir",
        type=str,
        default=str(DEFAULT_LOCAL_PROMPT_DIR),
        help=f"Local prompt directory path (default: {DEFAULT_LOCAL_PROMPT_DIR})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=str(DEFAULT_PROMPT_KEYS_PATH),
        help=f"Path to save the prompt keys YAML file (default: {DEFAULT_PROMPT_KEYS_PATH})",
    )
    args = parser.parse_args()

    load_dotenv(find_dotenv(usecwd=True))

    create_prompt_keys_from_local_prompt_dir(
        local_prompt_dir=args.prompt_dir,
        list_save_path=args.output,
    )
