import streamlit as st
import fitz  # PyMuPDF; ensure you have PyMuPDF installed (pip install PyMuPDF)
import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file at the very beginning
load_dotenv()

# Unset proxy environment variables if they exist
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

# Check that the API key is available
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set. Please add it to your .env file.")

# Set the OpenAI API key for the openai library
openai.api_key = os.getenv("OPENAI_API_KEY")

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

# Initialize embeddings without explicitly passing the API key
embeddings = OpenAIEmbeddings()

# Streamlit UI
st.title("📄 Retrieval-Augmented Generation (RAG) Chatbot")

# Upload multiple documents (.txt and .pdf)
uploaded_files = st.file_uploader("Upload documents", type=["txt", "pdf"], accept_multiple_files=True)

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    doc = fitz.open("pdf", pdf_file.read())
    return "\n".join([page.get_text() for page in doc])

if uploaded_files:
    documents = []
    
    # Process uploaded files
    for file in uploaded_files:
        if file.name.endswith(".pdf"):
            content = extract_text_from_pdf(file)
        else:
            content = file.read().decode("utf-8")
        documents.append(content)

    # Chunking text for retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = [chunk for doc in documents for chunk in text_splitter.split_text(doc)]

    # Embedding and storing in FAISS
    vector_db = FAISS.from_texts(chunks, embeddings)

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

        # Display the assistant response
        st.chat_message("assistant").write(response.choices[0].message["content"])
