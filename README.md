
# 📄 Retrieval-Augmented Generation (RAG) Chatbot

Welcome to the **Retrieval-Augmented Generation (RAG) Chatbot** repository. This project enables users to upload documents (TXT and PDF), process their content by chunking large files, and interact with that content through a conversational AI interface. The application uses OpenAI’s GPT-4 for generating responses and FAISS for efficient retrieval.

---

## 🚀 Features

- **File Upload:** Supports both `.txt` and `.pdf` files.
- **Multiple Document Handling:** Upload and process multiple documents.
- **PDF Parsing:** Extracts text from PDFs using PyMuPDF with a fallback to PyPDF2.
- **Efficient Chunking:** Splits large documents into smaller, manageable pieces.
- **Conversational Chat Interface:** Chat interface for asking questions about the document content.
- **Vector Search:** Uses FAISS to index and retrieve document chunks.
- **Docker & Devcontainer:** Pre-configured Docker and VS Code Devcontainer setup for a consistent development environment.

---

## 📂 Project Structure

```
RAG-Chatbot/
├── chatbot.py               # Main Streamlit chatbot application
├── Dockerfile               # Docker container configuration
├── docker-compose.yml       # Docker Compose configuration
├── .devcontainer/           # VS Code Devcontainer configuration files
├── requirements.txt         # Python dependencies
├── .env.example             # Sample environment file (with placeholder values)
└── README.md                # This documentation file
```

---

## 🛠️ Prerequisites

Before running the application, ensure you have the following installed:

- [Docker](https://www.docker.com/get-started) (Docker Desktop must be running)
- [VS Code](https://code.visualstudio.com/) with the **Remote - Containers Extension**
- [Git](https://git-scm.com/)
- [Python 3.9+](https://www.python.org/downloads/)
- An **OpenAI API Key**

---

## 🔧 Setup Instructions

### 1. Clone the Repository

Open a terminal and run:

```bash
git clone https://github.com/esadkaoui/INFO5940.git
cd INFO5940
```

---

### 2. Configure the Environment

1. **Create a `.env` File**  
  In the root directory, create a file named `.env`:
  ```bash
  touch .env
  ```
2. **Add Your Environment Variables**  
  Edit the `.env` file to include:
  ```plaintext
  OPENAI_API_KEY=your-api-key-here
  OPENAI_BASE_URL=https://api.openai.com
  TZ=America/New_York
  ```
3. **Ensure Secrets Are Not Committed**  
  Your real `.env` file should be excluded by listing it in your `.gitignore`. Use the provided `.env.example` (with placeholder values) to guide users.

---

### 3. Open in VS Code with Docker/Devcontainer

1. Open **VS Code** and navigate to the project folder.
2. Open the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P`) and select:
  ```
  Remote-Containers: Reopen in Container
  ```
3. VS Code will build and open the project inside the container.

---

### 4. Run the Application

#### **Option A: Inside the Devcontainer**

Once inside the container, open a terminal and run:

```bash
streamlit run chatbot.py
```

Then, open your browser and go to:  
`http://localhost:8501`

#### **Option B: Using Docker Compose**

1. Ensure Docker Desktop is running.
2. In your terminal, run:
  ```bash
  docker-compose up --build
  ```
3. Open your browser at the provided URL (typically `http://localhost:8501`).

---

## 📜 Usage Instructions

1. **Uploading Documents:**  
  - Click **"Upload documents"** and select one or more `.txt` or `.pdf` files.
  - The application will process and index the files.

2. **Chat Interface:**  
  - Enter your question in the chat input.
  - The chatbot uses the uploaded document content to generate contextually relevant responses.
  - The final aggregated answer is displayed as a single paragraph.

---

## 🛠️ Troubleshooting

- **PyMuPDF Not Installed Warning:**  
  If you see a warning about PyMuPDF (fitz) not being installed, install it by running:
  ```bash
  pip install pymupdf
  ```

- **OpenAI API Authentication Error (401):**  
  Ensure your `.env` file contains the correct API key and that all proxy environment variables are unset:
  ```bash
  unset HTTP_PROXY
  unset HTTPS_PROXY
  unset http_proxy
  unset https_proxy
  ```
  Also, verify that `openai.proxy = None` is set in your code.

- **Container Issues:**  
  If the container fails to start:
  ```bash
  docker-compose down
  docker-compose up --build
  ```
  Check the container logs for further details.


