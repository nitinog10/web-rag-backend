"""
RAG (Retrieval-Augmented Generation) Web Application

This Streamlit application demonstrates a complete RAG pipeline that:
1. Loads documents from a web URL
2. Splits documents into manageable chunks
3. Creates embeddings using Google's Gemini embedding model
4. Stores embeddings in an in-memory vector database
5. Retrieves relevant context based on user questions
6. Generates answers using Google's Gemini LLM with retrieved context

Features:
- Web-based interface for easy interaction
- Real-time document processing and indexing
- Question-answering with source context display
- Session state management for persistent vector store

Requirements:
- GEMINI_API_KEY environment variable must be set
- Internet connection for web document loading and API calls

Author: [Your Name]
Date: August 30, 2025
"""

import streamlit as st
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings,GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Function to create embeddings safely in Streamlit with threading
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor

def create_embeddings_in_thread(api_key):
    """Create embeddings in a separate thread with its own event loop"""
    def _create_embeddings():
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",  # Using the same model as your notebook
                google_api_key=api_key
            )
            return embeddings
        except Exception as e:
            print(f"Error in thread: {str(e)}")
            return None
        finally:
            loop.close()
    
    # Run in a separate thread
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_create_embeddings)
        return future.result()

@st.cache_resource
def get_embeddings(api_key):
    """Create and cache embeddings instance"""
    try:
        return create_embeddings_in_thread(api_key)
    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
        return None

# Configure Streamlit page settings
st.set_page_config(page_title="RAG Demo", page_icon="🤖", layout="wide")
st.title("RAG Question-Answering Demo")

# Load environment variables from .env file
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

# Validate API key availability
if not api_key:
    st.error("GEMINI_API_KEY is not set in the environment variables")
    st.stop()

# Initialize session state for vector store persistence
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

# User input for document URL
url = st.text_input("Enter a URL to load documents from:", 
                    value="https://www.govinfo.gov/content/pkg/CDOC-110hdoc50/html/CDOC-110hdoc50.htm")

# Document processing and vector store initialization
if st.button("Initialize RAG System"):
    with st.spinner("Loading and processing documents..."):
        # Load documents from the provided URL
        loader = WebBaseLoader(url)
        documents = loader.load()
        
        # Split documents into smaller chunks for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200, 
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)

        # Create embeddings using Google Gemini embedding model
        try:
            embeddings = get_embeddings(api_key)
            if embeddings is None:
                st.error("Failed to create embeddings. Please check your API key.")
                st.stop()
        except Exception as e:
            st.error(f"Error initializing embeddings: {str(e)}")
            st.stop()
        
        # Store document chunks in vector database
        st.session_state.vectorstore = InMemoryVectorStore.from_documents(
            chunks, 
            embeddings
        )
        st.success("RAG system initialized successfully!")
# Question-answering interface (only available after vector store initialization)
if st.session_state.vectorstore is not None:
    # Initialize the language model for answer generation (using the same approach as notebook)
    from langchain.chat_models import init_chat_model
    llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai", api_key=api_key)

    # Create prompt template for question-answering
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that can answer questions about the provided documents. Use the provided context to answer the question. IMPORTANT: If you are unsure of the answer, say 'I don't know' and don't make up an answer."),
        ("user", "Question: {question}\nContext: {context}")
    ])
    
    # Create processing chain: prompt -> LLM
    chain = prompt | llm
    
    # Create two-column layout for question input and answer display
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ask a Question")
        question = st.text_area("Enter your question:")
        
        # Process question and generate answer
        if st.button("Get Answer"):
            if question:
                with st.spinner("Generating answer..."):
                    # Retrieve relevant documents based on the question
                    retriever = st.session_state.vectorstore.as_retriever()
                    docs = retriever.invoke(question)
                    
                    # Generate answer using retrieved context (matching working code format)
                    response = chain.invoke({
                        "question": question,
                        "context": docs
                    })
                    
                    # Store results in session state for display
                    st.session_state.last_response = response.content
                    st.session_state.last_context = docs
            else:
                st.warning("Please enter a question.")
    
    with col2:
        st.subheader("Answer")
        # Display the generated answer
        if 'last_response' in st.session_state:
            st.write(st.session_state.last_response)
            
            # Show retrieved context documents in an expandable section
            with st.expander("Show Retrieved Context"):
                for i, doc in enumerate(st.session_state.last_context, 1):
                    st.markdown(f"**Relevant Document {i}:**")
                    st.markdown(doc.page_content)
                    st.markdown("---")
else:
    # Guide user to initialize the system first
    st.info("Please initialize the RAG system first by entering a URL and clicking 'Initialize RAG System'")