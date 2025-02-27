import openai

openai.api_key = "sk-..."
openai.proxy = None  # explicitly disable

try:
    resp = openai.Embedding.create(
        input="Hello world",
        model="text-embedding-ada-002"
    )
    print("Success:", resp)
except Exception as e:
    print("Failed:", e)
