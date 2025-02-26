import os
from dotenv import load_dotenv
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

load_dotenv()
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

try:
    response = client.embeddings.create(input="Test",
    model="text-embedding-ada-002")
    print("API call succeeded:", response)
except Exception as e:
    print("API call failed:", e)
