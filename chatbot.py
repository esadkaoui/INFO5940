import streamlit as st
import fitz  # PyMuPDF
import openai
from os import environ
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from tenacity import retry, stop_after_attempt, wait_fixed
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set proxy environment variables if they exist
http_proxy = environ.get("HTTP_PROXY")
https_proxy = environ.get("HTTPS_PROXY")

if http_proxy:
    openai.proxy = {"http": http_proxy}
if https_proxy:
    openai.proxy = {"https": https_proxy}

# Check that the API key is available
if not environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set. Please add it to your .env file.")

# Set the OpenAI API key for the openai library
openai.api_key = environ.get("OPENAI_API_KEY")

# Initialize embeddings with the API key
embeddings = OpenAIEmbeddings(openai_api_key=environ.get("OPENAI_API_KEY"))

# Streamlit UI
st.title("📄 Retrieval-Augmented Generation (RAG) Chatbot")

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