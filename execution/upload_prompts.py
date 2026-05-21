from dotenv import find_dotenv, load_dotenv

from core.llm.langfuse import upload_prompts_from_local_by_prompt_keys

if __name__ == "__main__":
    load_dotenv(find_dotenv(usecwd=True))
    upload_prompts_from_local_by_prompt_keys(labels=["develop"])
