"""
Configuration settings for the Cheese Recommendation Agent.
"""
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb+srv://fullmax0111:Lhv4o4AEuoQVMfGg@cheese-agent.ozp8uxm.mongodb.net/")
MONGODB_DB = os.getenv("MONGODB_DB", "Cheese_Data")
MONGODB_COLLECTION = "Cheese_Data"  # Default collection name

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o"  # You can change this to your preferred model
# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
PINECONE_INDEX = "cheese-embeddings"

# Vector embedding dimensions for OpenAI embeddings
EMBEDDING_DIM = 1536  # Dimensions for text-embedding-3-small

# Application settings
DEBUG_MODE = False
MAX_REASONING_STEPS = 5
DEFAULT_TEMPERATURE = 0.7 