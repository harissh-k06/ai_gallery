import os
from dotenv import load_dotenv

load_dotenv()

LLM_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
LLM_BASE_URL: str = "https://api.deepseek.com/beta"
LLM_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
