import os
import streamlit as st
import openai
from dotenv import load_dotenv
from io import BytesIO
import PyPDF2

# Load environment variables from .env file
load_dotenv()

# Unset all proxy environment variables (both uppercase and lowercase)
for var in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
    os.environ.pop(var, None)

# Check that the API key is available
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set. Please add it to your .env file.")

# Set the OpenAI API key for the openai library
openai.api_key = api_key

# Initialize embeddings using your API key
from langchain_community.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(openai_api_key=api_key)

# Streamlit UI
st.title("📄 RAG Chatbot")

# Upload multiple documents (.txt and .pdf)
uploaded_files = st.file_uploader(
    "Upload documents (.txt and .pdf)",
    type=["txt", "pdf"],
    accept_multiple_files=True
)

documents = []
if uploaded_files:
    for file in uploaded_files:
        if file.type == "text/plain":
            content = file.read().decode("utf-8")
            documents.append(content)
        elif file.type == "application/pdf":
            pdf_stream = BytesIO(file.read())
            # First try PyMuPDF (fitz) for extraction
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

            # Reset stream position and use PyPDF2
            pdf_stream.seek(0)
            reader = PyPDF2.PdfReader(pdf_stream)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            if not text.strip():
                raise ValueError("No text extracted from PDF.")
            documents.append(text)

    # Split documents into chunks
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    for doc in documents:
        chunks.extend(text_splitter.split_text(doc))

    # Create a FAISS vector store from the chunks
    from langchain_community.vectorstores import FAISS
    try:
        vector_db = FAISS.from_texts(chunks, embeddings)
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        raise e

    st.success(f"✅ {len(uploaded_files)} file(s) processed and indexed successfully!")
    
    # Chat Interface
    question = st.chat_input("Ask something about the uploaded documents")
    if question:
        # Retrieve the top 5 relevant chunks using the FAISS retriever
        retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        retrieved_docs = retriever.invoke(question)
        
        # Format the retrieved documents into context text
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        context = format_docs(retrieved_docs)
        
        # Build a prompt template
        from langchain_core.prompts import PromptTemplate
        prompt_template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Answer:"""
        prompt = PromptTemplate.from_template(prompt_template)
        final_prompt = prompt.format(question=question, context=context)
        
        # Use ChatOpenAI from langchain.chat_models
        from langchain.chat_models import ChatOpenAI
        from langchain_core.messages import HumanMessage
        
        # Initialize the LLM (ChatOpenAI) with the GPT-4 model
        llm = ChatOpenAI(model="gpt-4", temperature=0.2)
        
        # Generate the answer by invoking the LLM with the constructed prompt
        response = llm.invoke([HumanMessage(content=final_prompt)])
        
        # response is an AIMessage object; just extract .content
        response_text = response.content
        
        # Display text
        st.chat_message("assistant").write(response_text)
