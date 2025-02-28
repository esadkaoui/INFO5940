import os
import streamlit as st
import openai
from dotenv import load_dotenv
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
import PyPDF2

# Load environment variables from .env file
load_dotenv()

# Unset all proxy environment variables 
for var in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
    os.environ.pop(var, None)

# Check that the API key is available
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set. Please add it to your .env file.")

# Debug: Print masked API key (for verification)
print("API Key loaded:", api_key[:4] + "****" + api_key[-4:])

# Set the OpenAI API key for the openai library
openai.api_key = api_key

# Initialize embeddings
embeddings = OpenAIEmbeddings(openai_api_key=api_key)

# Streamlit UI
st.title("📄 RAG Chatbot")

# Upload multiple documents (.txt and .pdf)
uploaded_files = st.file_uploader("Upload documents (.txt and .pdf)", type=["txt", "pdf"], accept_multiple_files=True)

documents = []
if uploaded_files:
    for file in uploaded_files:
        if file.type == "text/plain":
            content = file.read().decode("utf-8")
            documents.append(content)
        elif file.type == "application/pdf":
            pdf_stream = BytesIO(file.read())
            try:
                import fitz  # PyMuPDF
                pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")
                text = ""
                for page_num in range(len(pdf_document)):
                    page = pdf_document.load_page(page_num)
                    text += page.get_text()
                if text.strip():
                    documents.append(text)
                    continue
            except Exception:
                pass  # Fall back to PyPDF2
            
            pdf_stream.seek(0)
            reader = PyPDF2.PdfReader(pdf_stream)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            if not text.strip():
                raise ValueError("No text extracted using PyMuPDF or PyPDF2.")
            documents.append(text)

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    for doc in documents:
        chunks.extend(text_splitter.split_text(doc))

    # Create a FAISS vector store from the document chunks
    try:
        vector_db = FAISS.from_texts(chunks, embeddings)
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        raise e

    st.success(f"✅ {len(uploaded_files)} file(s) processed and indexed successfully!")
    
    # Chat Interface
    question = st.chat_input("Ask something about the uploaded documents")
    if question:
        # Initialize session messages if not already done
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "Hello! How can I help you today?"}]
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)
        
        # Retrieve the top 5 relevant chunks using the FAISS index
        try:
            relevant_docs = vector_db.similarity_search(question, k=5)
        except Exception as e:
            st.error(f"Error during document retrieval: {e}")
            relevant_docs = []
        
        # Extract text content from the Document objects
        relevant_texts = [doc.page_content for doc in relevant_docs]
        
        # Combine the retrieved chunks to form a context
        context = "\n\n".join(relevant_texts)
        
        # Build the prompt for the ChatCompletion API
        messages = [
            {"role": "system", "content": "You are an assistant that answers questions based on the provided context. If you don't know the answer, say so."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ]
        
        # Optionally add previous conversation history
        if st.session_state.messages:
            # Append only the previous user and assistant messages (if needed)
            for msg in st.session_state.messages:
                if msg["role"] != "system":
                    messages.append(msg)
        
        final_response = ""
        stream = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            stream=True
        )
        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            text = delta.get("content", "")
            final_response += text
        st.session_state.messages.append({"role": "assistant", "content": final_response})
        st.chat_message("assistant").write(final_response)
