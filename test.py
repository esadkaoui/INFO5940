from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get API Key
api_key = os.getenv("OPENAI_API_KEY")

# Check if the key is loaded
if api_key is None:
    raise ValueError("❌ OPENAI_API_KEY is missing. Add it to .env or export it.")

print("✅ API Key loaded successfully!")  # Debugging line
