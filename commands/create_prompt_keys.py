import argparse
from dotenv import find_dotenv, load_dotenv

from core.langfuse import create_prompt_keys_from_local_prompt_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create prompt keys from local prompt directory")
    parser.add_argument(
        "-d",
        "--prompt-dir",
        type=str,
        default=None,
        help="Local prompt directory path"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path to save the prompt keys YAML file"
    )
    args = parser.parse_args()

    load_dotenv(find_dotenv(usecwd=True))
    
    kwargs = {}
    if args.prompt_dir:
        kwargs["local_prompt_dir"] = args.prompt_dir
    if args.output:
        kwargs["list_save_path"] = args.output
        
    create_prompt_keys_from_local_prompt_dir(**kwargs)

