import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    """Instantiate an OpenRouter-backed LLM using credentials from .env."""
    return ChatOpenAI(
        model=os.environ["MODEL_NAME"],
        api_key=os.environ["API_KEY"],
        base_url=os.environ["BASE_URL"],
        temperature=temperature,
    )
