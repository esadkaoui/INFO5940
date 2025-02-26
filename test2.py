import os
from dotenv import load_dotenv
import openai

load_dotenv()
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

openai.api_key = os.getenv("OPENAI_API_KEY")
try:
    response = openai.Embedding.create(
        input="Test",
        model="text-embedding-ada-002"
    )
    print("API call succeeded:", response)
except Exception as e:
    print("API call failed:", e)
