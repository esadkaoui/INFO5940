import streamlit as st
import openai
from os import environ
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from tenacity import retry, stop_after_attempt, wait_fixed
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Unset proxy environment variables if they exist
environ.pop("HTTP_PROXY", None)
environ.pop("HTTPS_PROXY", None)

# Check that the API key is available
api_key = environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set. Please add it to your .env file.")

# Print the API key for debugging purposes (remove or mask before sharing logs)
print(f"OpenAI API Key: {api_key}")

# Set the OpenAI API key for the openai library
openai.api_key = api_key

# Initialize embeddings with the API key
embeddings = OpenAIEmbeddings(openai_api_key=api_key)

# Streamlit UI
st.title("📝 File Q&A with OpenAI")

# Upload multiple documents (.txt and .pdf)
uploaded_files = st.file_uploader("Upload documents", type=["txt", "pdf"], accept_multiple_files=True)

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

if uploaded_files:
    documents = []
    
    # Process uploaded files
    for file in uploaded_files:
        if file.name.lower().endswith(".pdf"):
            try:
                content = extract_text_from_pdf(file)
            except Exception as e:
                st.error(f"Error processing PDF {file.name}: {e}")
                continue
        else:
            content = file.read().decode("utf-8")
        documents.append(content)

    # Debugging: Print the number of documents processed
    st.write(f"Number of documents processed: {len(documents)}")

    # Chunking text for retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = [chunk for doc in documents for chunk in text_splitter.split_text(doc)]

    # Debugging: Print the number of chunks generated
    st.write(f"Number of chunks generated: {len(chunks)}")

    # Embedding and storing in FAISS
    try:
        if not chunks:
            raise ValueError("No chunks to process.")
        vector_db = FAISS.from_texts(chunks, embeddings)
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        raise e

    st.success(f"✅ {len(uploaded_files)} files processed and indexed successfully!")

    # Chat Interface
    question = st.chat_input("Ask something about the uploaded documents")
    if question:
        retrieved_chunks = vector_db.similarity_search(question, k=3)
        context = "\n\n".join([chunk.page_content for chunk in retrieved_chunks])

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"Use this context: {context}"},
                {"role": "user", "content": question}
            ]
        )

        # Display the assistant's response
        st.chat_message("assistant").write(response.choices[0].message["content"])

# Single file upload for Q&A
uploaded_file = st.file_uploader("Upload an article", type=("txt", "md"))

question = st.chat_input(
    "Ask something about the article",
    disabled=not uploaded_file,
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Ask something about the article"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if question and uploaded_file:
    # Read the content of the uploaded file
    file_content = uploaded_file.read().decode("utf-8")
    print(file_content)
    
    # Set the OpenAI API key
    openai.api_key = environ.get('OPENAI_API_KEY')

    # Set proxy environment variables if they exist
    http_proxy = environ.get("HTTP_PROXY")
    https_proxy = environ.get("HTTPS_PROXY")

    if http_proxy:
        openai.proxy = {"http": http_proxy}
    if https_proxy:
        openai.proxy = {"https": https_proxy}

    # Append the user's question to the messages
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    # Create a completion request to OpenAI
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"Use this context: {file_content}"},
            {"role": "user", "content": question}
        ]
    )

    # Append the assistant's response to the messages
    assistant_response = response.choices[0].message["content"]
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    st.chat_message("assistant").write(assistant_response)