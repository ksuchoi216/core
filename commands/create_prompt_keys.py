from dotenv import find_dotenv, load_dotenv

from core.llm.langfuse import create_prompt_keys_from_local_prompt_dir

if __name__ == "__main__":
    load_dotenv(find_dotenv(usecwd=True))
    create_prompt_keys_from_local_prompt_dir()
