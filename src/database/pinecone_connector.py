"""
Pinecone vector database connector for semantic search.
"""
from typing import Dict, List, Any, Optional, Tuple
import json
import pinecone
from openai import OpenAI
from pinecone import ServerlessSpec
from bson.objectid import ObjectId
from src.config import (
    PINECONE_API_KEY,
    PINECONE_ENVIRONMENT,
    PINECONE_INDEX,
    OPENAI_API_KEY,
    EMBEDDING_DIM
)
import httpx


class PineconeConnector:
    """Pinecone vector database connector for semantic search."""
    
    def __init__(self, api_key: str = PINECONE_API_KEY, 
                 environment: str = PINECONE_ENVIRONMENT,
                 index_name: str = PINECONE_INDEX,
                 openai_api_key: str = OPENAI_API_KEY):
        """
        Initialize Pinecone connector.
        
        Args:
            api_key: Pinecone API key
            environment: Pinecone environment
            index_name: Pinecone index name
            openai_api_key: OpenAI API key for generating embeddings
        """
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        
        # Initialize Pinecone with new API (v3.0.0+)
        pc = pinecone.Pinecone(api_key=api_key)
        
        # Check if index exists, if not create it
        index_list = [index.name for index in pc.list_indexes()]
        if index_name not in index_list:
            # For Pinecone v3.0.0+, we need to create a proper spec
            
            
            # Create the index with proper specs
            pc.create_index(
                name=index_name,
                spec=ServerlessSpec(
                    cloud="aws",  # or "gcp" depending on your preference
                    region="us-east-1"  # choose appropriate region
                ),
                dimension=EMBEDDING_DIM,
                metric="cosine"
            )
        print("created index")
        # Connect to index
        self.index = pc.Index(index_name)
        
        # Initialize OpenAI client for embeddings
        self.openai_client = OpenAI(
            api_key=openai_api_key,
            http_client=httpx.Client(
                timeout=60.0,
                follow_redirects=True
            )
        )
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using OpenAI.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Embedding vector
        """
        response = self.openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    
    def upsert_document(self, id: str, document: Dict[str, Any]) -> None:
        """
        Upsert document to Pinecone.
        
        Args:
            id: Document ID
            document: Document data
        """
        # Extract text from document to generate embedding
        text = document.get("text", "")
        if not text:
            # If no 'text' field exists, create a descriptive text from name and other fields
            name = document.get("name", "")
            brand = document.get("brand", "")
            department = document.get("department", "")
            text = f"{name} {brand} {department}"
        
        # Generate embedding
        embedding = self.generate_embedding(text)
        
        # Create metadata (excluding very large fields to stay within limits)
        # Convert ObjectId to str for Pinecone compatibility
        metadata = {}
        for k, v in document.items():
            if k not in ["text", "images"]:
                if isinstance(v, ObjectId):
                    metadata[k] = str(v)
                else:
                    metadata[k] = v
        
        # Upsert to Pinecone with new API
        self.index.upsert(
            vectors=[
                {
                    "id": id, 
                    "values": embedding, 
                    "metadata": metadata
                }
            ]
        )
    
    def upsert_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Upsert multiple documents to Pinecone.
        
        Args:
            documents: List of documents to upsert
        """
        vectors = []
        
        for doc in documents:
            doc_id = str(doc.get("sku", ""))
            
            # Extract text from document to generate embedding
            text = doc.get("text", "")
            if not text:
                # If no 'text' field exists, create a descriptive text from name and other fields
                name = doc.get("name", "")
                brand = doc.get("brand", "")
                department = doc.get("department", "")
                text = f"{name} {brand} {department}"
            
            # Generate embedding
            embedding = self.generate_embedding(text)
            
            # Create metadata (excluding very large fields to stay within limits)
            # Convert ObjectId to str for Pinecone compatibility
            current_doc_metadata = {}
            for k, v in doc.items():
                if k not in ["text", "images"]:
                    if isinstance(v, ObjectId):
                        current_doc_metadata[k] = str(v)
                    else:
                        current_doc_metadata[k] = v
            
            vectors.append({
                "id": doc_id,
                "values": embedding,
                "metadata": current_doc_metadata
            })
        
        # Upsert to Pinecone in batches of 100
        batch_size = 10
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            self.index.upsert(vectors=batch,namespace="Cheese_Data")
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        Search Pinecone for documents similar to query.
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of (id, metadata, score) tuples
        """
        # Generate embedding for query
        query_embedding = self.generate_embedding(query)
        
        # Search Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Format results
        formatted_results = []
        for match in results.matches:
            formatted_results.append((
                match.id,
                match.metadata,
                match.score
            ))
        
        return formatted_results
    
    def delete_document(self, id: str) -> None:
        """
        Delete document from Pinecone.
        
        Args:
            id: Document ID
        """
        self.index.delete(ids=[id])
    
    def delete_all(self) -> None:
        """Delete all documents from Pinecone index."""
        self.index.delete(delete_all=True) 