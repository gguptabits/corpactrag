import os
import requests
import json
import hashlib
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env
load_dotenv()

class FinanceRAGService:
    def __init__(self, persist_directory="./chroma_db", credentials_file="user_credentials.json"):
        self.hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not self.hf_token:
            raise ValueError("Please set HUGGINGFACEHUB_API_TOKEN in the .env file")
        
        self.persist_directory = persist_directory
        self.credentials_file = credentials_file
        
        # 1. Initialize Embeddings (HuggingFace)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # 2. Initialize Vector DB (Chroma)
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory, 
            embedding_function=self.embeddings
        )
        
        # 3. Initialize Chat Model using Qwen 2.5 (High Performance, Non-Gated, Conversational-native)
        repo_id = "Qwen/Qwen2.5-7B-Instruct"
        
        self.llm_endpoint = HuggingFaceEndpoint(
            repo_id=repo_id,
            max_new_tokens=512,
            temperature=0.3,
            huggingfacehub_api_token=self.hf_token
        )
        
        # Wrap with ChatHuggingFace to satisfy "conversational" task requirements for the API
        self.chat_model = ChatHuggingFace(llm=self.llm_endpoint)
        
        # 4. Define the Chat Prompt Template
        self.system_prompt = (
            "You are an expert financial analyst specializing in Corporate Action Scenario modeling. "
            "Use the following retrieved context to answer the user's question accurately. "
            "If you do not know the answer based on the context, say so. Do not hallucinate numbers. "
            "\n\n"
            "Context: {context}"
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{input}"),
        ])

    def ingest_document(self, file_path: str, file_name: str):
        """Ingest a document, split it, and store in Chroma DB."""
        if file_name.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_name.endswith('.txt'):
            loader = TextLoader(file_path)
        else:
            raise ValueError("Unsupported file format. Please upload PDF or TXT.")
        
        documents = loader.load()
        
        # Split documents into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = text_splitter.split_documents(documents)
        
        # Save chunks to Vector Store
        self.vectorstore.add_documents(splits)
        return f"Successfully ingested {len(splits)} chunks from {file_name}."

    def _format_docs(self, docs):
        """Helper to combine page contents of retrieved documents."""
        return "\n\n".join(doc.page_content for doc in docs)

    def answer_query(self, query: str) -> str:
        """Retrieve context from Chroma and answer using modern LCEL."""
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        
        # Construct the pipeline with LCEL mapping ChatHuggingFace
        rag_chain = (
            {
                "context": retriever | self._format_docs, 
                "input": RunnablePassthrough()
            }
            | self.prompt
            | self.chat_model
            | StrOutputParser()
        )
        
        # Execute the chain
        return rag_chain.invoke(query)

    def transcribe_audio(self, audio_bytes: bytes) -> str:
        """Convert voice to text using HuggingFace Inference API."""
        API_URL = "https://api-inference.huggingface.co/models/openai/whisper-small"
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        
        try:
            response = requests.post(API_URL, headers=headers, data=audio_bytes)
            response.raise_for_status()
            result = response.json()
            return result.get("text", "Sorry, I could not transcribe the audio.")
        except Exception as e:
            return f"Audio transcription failed: {str(e)}"

    def _hash_password(self, password: str) -> str:
        """Hash a password for secure storage."""
        return hashlib.sha256(password.encode()).hexdigest()

    def _load_credentials(self) -> dict:
        """Loads user credentials from the server-maintained JSON file."""
        if os.path.exists(self.credentials_file):
            try:
                with open(self.credentials_file, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_credentials(self, credentials: dict):
        """Saves user credentials to the server-maintained JSON file."""
        with open(self.credentials_file, "w") as f:
            json.dump(credentials, f, indent=4)

    def authenticate_user(self, username, password) -> bool:
        """Authenticate a user against stored credentials."""
        credentials = self._load_credentials()
        if username in credentials and credentials[username] == self._hash_password(password):
            return True
        return False

    def register_user(self, username, password) -> tuple:
        """Register a new user. Returns (success, message)."""
        credentials = self._load_credentials()
        if username in credentials:
            return False, "Username already exists. Please choose another."
        
        credentials[username] = self._hash_password(password)
        try:
            self._save_credentials(credentials)
            return True, "Account created successfully! You can now login in the Sign In tab."
        except Exception as e:
            return False, f"Server error saving credentials: {e}"