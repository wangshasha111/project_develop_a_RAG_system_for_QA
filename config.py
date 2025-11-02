"""
Configuration Module
Contains configuration settings and constants for the RAG system
"""

from pathlib import Path


class Config:
    """Configuration settings for the RAG system"""
    
    # Application settings
    APP_NAME = "Multimodal RAG System"
    APP_ICON = "📚"
    VERSION = "1.0.0"
    
    # Paths
    BASE_DIR = Path(__file__).parent
    DATASET_DIR = BASE_DIR / "RAG Project Dataset"
    FIGURES_DIR = DATASET_DIR / "figures"
    CHROMA_DB_DIR = BASE_DIR / "chroma_db"
    
    # Model settings
    OPENAI_MODELS = ["gpt-4o", "gpt-4o-mini"]
    GOOGLE_MODELS = ["gemini-2.0-flash-exp", "gemini-1.5-pro"]
    
    DEFAULT_OPENAI_MODEL = "gpt-4o"
    DEFAULT_GOOGLE_MODEL = "gemini-2.0-flash-exp"
    
    # Embedding settings
    OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
    GOOGLE_EMBEDDING_MODEL = "models/embedding-001"
    
    # Retrieval settings
    DEFAULT_K = 5  # Number of documents to retrieve
    MAX_K = 10
    
    # Document processing settings
    CHUNK_SIZE = 4000
    CHUNK_OVERLAP = 2000
    MIN_CHUNK_SIZE = 2000
    
    # Redis settings
    REDIS_HOST = "localhost"
    REDIS_PORT = 6379
    REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}"
    
    # UI settings
    SIDEBAR_STATE = "expanded"
    LAYOUT = "wide"
    
    # Chat settings
    MAX_CHAT_HISTORY = 10  # Maximum number of chat exchanges to keep
    CONTEXT_WINDOW = 3  # Number of previous exchanges to include in context
    
    # Prompt templates
    PROMPT_TEMPLATES = {
        "RAG Components": "What are the main components of a RAG model, and how do they interact?",
        "Transformer Layers": "What are the two sub-layers in each encoder layer of the Transformer model?",
        "Positional Encoding": "Explain how positional encoding is implemented in Transformers and why it is necessary.",
        "Multi-head Attention": "Describe the concept of multi-head attention in the Transformer architecture. Why is it beneficial?",
        "Few-shot Learning": "What is few-shot learning, and how does GPT-3 implement it during inference?",
        "Training Objectives": "What are the different training objectives used in transformer-based language models?",
        "Attention Mechanisms": "Compare and contrast different types of attention mechanisms in modern transformers.",
        "Scaling Laws": "Discuss the scaling laws in large language models and their implications.",
    }
    
    # System prompts
    SYSTEM_PROMPT = """You are an expert AI assistant specializing in analyzing AI research papers.
You have access to information from multiple research papers including text, tables, and images (charts, graphs, diagrams).

Your task is to provide accurate, detailed answers based on the provided context.

INSTRUCTIONS:
1. Analyze ALL provided context including text, tables, and images
2. Provide comprehensive answers citing specific information from the sources
3. If the context contains tables or numerical data, include those details in your answer
4. If images show charts or graphs, describe the trends and key insights
5. Always attribute information to its source (e.g., "According to the paper...", "The table shows...")
6. If you cannot find the answer in the provided context, clearly state that
7. Do NOT make up information that is not in the context
"""
    
    SUMMARIZATION_PROMPT = """You are an assistant tasked with summarizing tables and text particularly for semantic retrieval.
These summaries will be embedded and used to retrieve the raw text or table elements.
Give a detailed summary of the table or text below that is well optimized for retrieval.
For any tables also add in a one line description of what the table is about besides the summary.
Do not add redundant words like Summary.
Just output the actual summary content.
"""
    
    IMAGE_SUMMARIZATION_PROMPT = """You are an assistant tasked with summarizing images for retrieval.
Remember these images could potentially contain graphs, charts or tables also.
These summaries will be embedded and used to retrieve the raw image for question answering.
Give a detailed summary of the image that is well optimized for retrieval.
Do not add additional words like Summary, This image represents, etc.
"""
