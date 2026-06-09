import argparse
from dotenv import find_dotenv, load_dotenv

from core.langfuse import (
    upload_prompts_from_local,
    change_project_keys_from_env,
    create_prompt_keys_from_local_prompt_dir,
)
from core.langfuse.prompt import DEFAULT_LOCAL_PROMPT_DIR, DEFAULT_PROMPT_KEYS_PATH

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload prompts to Langfuse")
    parser.add_argument(
        "-d",
        "--prompt-dir",
        type=str,
        default=str(DEFAULT_LOCAL_PROMPT_DIR),
        help=f"Local prompt directory path (default: {DEFAULT_LOCAL_PROMPT_DIR})",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=str(DEFAULT_PROMPT_KEYS_PATH),
        help=f"Path to the prompt keys YAML file (default: {DEFAULT_PROMPT_KEYS_PATH})",
    )
    parser.add_argument(
        "-p",
        "--project",
        type=str,
        default=None,
        help="Project name to map env keys (e.g., document)",
    )
    parser.add_argument(
        "-l",
        "--labels",
        nargs="*",
        default=["production"],
        help="Labels for the uploaded prompts (default: production)",
    )
    args = parser.parse_args()

    load_dotenv(find_dotenv(usecwd=True))

    if args.project:
        change_project_keys_from_env(args.project)

    create_prompt_keys_from_local_prompt_dir(
        local_prompt_dir=args.prompt_dir,
        list_save_path=args.input,
    )

    upload_prompts_from_local(
        local_prompt_dir=args.prompt_dir,
        prompt_list_path=args.input,
        labels=args.labels,
    )
