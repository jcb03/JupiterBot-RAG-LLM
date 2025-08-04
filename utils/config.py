import os
import streamlit as st

class Config:
    """Configuration for Jupiter RAG System with LLM"""
    
    # OpenAI Configuration
    @staticmethod
    def get_openai_api_key():
        """Get OpenAI API key from secrets or environment"""
        try:
            if hasattr(st, 'secrets') and 'openai' in st.secrets:
                return st.secrets.openai.api_key
        except:
            pass
        return os.getenv("OPENAI_API_KEY")
    
    # Database Configuration
    @staticmethod
    def get_database_url():
        """Get database URL from secrets or environment"""
        try:
            if hasattr(st, 'secrets') and 'database' in st.secrets:
                return st.secrets.database.url
        except:
            pass
        return os.getenv("DATABASE_URL", "sqlite:///jupiter_rag.db")
    
    # RAG Configuration
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MAX_RETRIEVED_DOCS = 5
    SIMILARITY_THRESHOLD = 0.3
    
    # LLM Configuration
    LLM_MODEL = "gpt-3.5-turbo"
    LLM_TEMPERATURE = 0.7
    MAX_TOKENS = 1000
    
    # Scraping Configuration
    JUPITER_BASE_URL = "https://jupiter.money"
    DELAY_BETWEEN_REQUESTS = 1
    MAX_PAGES_TO_SCRAPE = 30
    
    # ChromaDB Configuration
    CHROMA_PERSIST_DIR = "./chroma_db"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # UI Configuration
    PAGE_TITLE = "Jupiter AI Assistant"
    PAGE_ICON = "🤖"
