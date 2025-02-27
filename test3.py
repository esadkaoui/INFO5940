import os 
import openai
openai.api_key = os.environ.get("OPENAI_API_KEY")
openai.proxy = None  # Explicitly disable proxy usage
