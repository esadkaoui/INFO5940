import os
import streamlit as st
import openai
from dotenv import load_dotenv
from io import BytesIO

# Load environment variables from .env file
load_dotenv()

# Unset all proxy environment variables (both uppercase and lowercase)
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

# Initialize embeddings without passing the API key explicitly
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

# Attempt to import PyMuPDF (fitz)
try:
    import fitz
except ImportError:
    st.error("PyMuPDF (fitz) is not installed. Please install it if you want PDF extraction via PyMuPDF.")
    fitz = None

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file using PyMuPDF first, then fall back to PyPDF2 if needed."""
    try:
        # Try to get the PDF bytes (Streamlit's UploadedFile is usually BytesIO-like)
        try:
            pdf_bytes = pdf_file.getvalue()
        except AttributeError:
            pdf_bytes = pdf_file.read()

        pdf_stream = BytesIO(pdf_bytes)
        pdf_stream.seek(0)

        # Try PyMuPDF first if available
        if fitz:
            try:
                doc = fitz.open(pdf_stream, filetype="pdf")
                text = "\n".join(page.get_text() for page in doc)
                if text.strip():
                    return text
                # If no text extracted, fall back to PyPDF2
            except Exception:
                pass  # Silently fail and fall back

        # Fallback to PyPDF2
        import PyPDF2
        pdf_stream.seek(0)
        reader = PyPDF2.PdfReader(pdf_stream)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        if not text.strip():
            raise ValueError("No text extracted using PyMuPDF or PyPDF2.")
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        raise e

# Streamlit UI
st.title("📄 Retrieval-Augmented Generation (RAG) Chatbot")

uploaded_files = st.file_uploader("Upload documents", type=["txt", "pdf"], accept_multiple_files=True)

documents = []
if uploaded_files:
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
    
    combined_text = "\n".join(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = [chunk for doc in documents for chunk in text_splitter.split_text(doc)]
    
    if not chunks:
        st.error("No chunks to process. Please check your uploaded files.")
        st.stop()
    
    try:
        vector_db = FAISS.from_texts(chunks, embeddings)
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        raise e
    
    st.success(f"✅ {len(uploaded_files)} files processed and indexed successfully!")

    # Chat Interface
    question = st.chat_input("Ask something about the uploaded documents")
    if question:
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "Hello! How can I help you today?"}]
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)
        
        messages = [{"role": "system", "content": f"Document content: {combined_text}"}] + st.session_state.messages
        
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
