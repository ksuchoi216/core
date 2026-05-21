from dotenv import find_dotenv, load_dotenv

from core.llm.langfuse import download_prompts_from_local_prompt_list

if __name__ == "__main__":
    load_dotenv(find_dotenv(usecwd=True))
    download_prompts_from_local_prompt_list()
