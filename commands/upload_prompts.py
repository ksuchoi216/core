import argparse
from dotenv import find_dotenv, load_dotenv

from core.langfuse import upload_prompts_from_local, change_project_keys_from_env

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload prompts to Langfuse")
    parser.add_argument(
        "-d",
        "--prompt-dir",
        type=str,
        default=None,
        help="Local prompt directory path"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        help="Path to the prompt keys YAML file"
    )
    parser.add_argument(
        "-p",
        "--project",
        type=str,
        default=None,
        help="Project name to map env keys (e.g., document)"
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
        
    kwargs = {}
    if args.prompt_dir:
        kwargs["local_prompt_dir"] = args.prompt_dir
    if args.input:
        kwargs["prompt_list_path"] = args.input
    kwargs["labels"] = args.labels
        
    upload_prompts_from_local(**kwargs)

