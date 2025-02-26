import os
import streamlit as st
import fitz  # PyMuPDF; ensure you have PyMuPDF installed (pip install PyMuPDF)
import openai
from dotenv import load_dotenv
from io import BytesIO

# Load environment variables from .env file at the very beginning
load_dotenv()

# Unset proxy environment variables if they exist
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

# Debug: Verify the imported fitz module
st.write("Using fitz from:", fitz.__file__)
st.write("Does fitz have an open attribute?", hasattr(fitz, "open"))

# Check that the API key is available
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set. Please add it to your .env file.")

# Set the OpenAI API key for the openai library
openai.api_key = os.getenv("OPENAI_API_KEY")

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

# Initialize embeddings (reads API key from openai.api_key)
embeddings = OpenAIEmbeddings()

# Streamlit UI
st.title("📄 Retrieval-Augmented Generation (RAG) Chatbot")

# Upload multiple documents (.txt and .pdf)
uploaded_files = st.file_uploader("Upload documents", type=["txt", "pdf"], accept_multiple_files=True)

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    # Read the entire file into a bytes variable
    pdf_bytes = pdf_file.read()
    st.write("PDF file size (bytes):", len(pdf_bytes))
    # Wrap the bytes in a BytesIO object
    pdf_stream = BytesIO(pdf_bytes)
    try:
        # Use the BytesIO stream as the source for the PDF
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
    except Exception as e:
        st.error(f"Error opening PDF with fitz: {e}")
        raise e
    return "\n".join([page.get_text() for page in doc])

if uploaded_files:
    documents = []
    
    # Process each uploaded file
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

    # Chunking text for retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = [chunk for doc in documents for chunk in text_splitter.split_text(doc)]

    # Embedding and storing in FAISS
    try:
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
