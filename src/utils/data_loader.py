"""
Data loader utility to populate Pinecone from MongoDB.
"""
from typing import Dict, List, Any, Optional
import logging
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now we can import from src
from src.database.mongodb_connector import MongoDBConnector
from src.database.pinecone_connector import PineconeConnector


def load_data_to_pinecone(batch_size: int = 10, limit: int = 1000) -> None:
    """
    Load data from MongoDB to Pinecone.
    
    Args:
        batch_size: Number of documents to load at once
        limit: Maximum number of documents to load
    """
    # try:
        # Initialize connectors
    mongodb = MongoDBConnector()
    pinecone = PineconeConnector()
    
    # Get documents from MongoDB
    cursor = mongodb.collection.find().limit(limit)
    documents = list(cursor)
    
    logging.info(f"Loading {len(documents)} documents to Pinecone...")
    
    # Load documents to Pinecone in batches
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        pinecone.upsert_documents(batch)
        logging.info(f"Loaded batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1} to Pinecone")
    
    logging.info("Finished loading data to Pinecone")
        
    # except Exception as e:
    #     logging.error(f"Error loading data to Pinecone: {e}")
    # finally:
    #     # Close connections
    #     mongodb.close()


def get_document_by_id(doc_id: str) -> Optional[Dict[str, Any]]:
    """
    Get document by ID from MongoDB.
    
    Args:
        doc_id: Document ID
        
    Returns:
        Document or None if not found
    """
    try:
        mongodb = MongoDBConnector()
        
        # First try to find by ObjectId
        if doc_id.startswith("ObjectId(") and doc_id.endswith(")"):
            oid = doc_id[9:-1]
            document = mongodb.find_by_field("_id.$oid", oid)
            if document:
                return document[0]
        
        # Then try to find by sku
        document = mongodb.find_by_field("sku", doc_id)
        if document:
            return document[0]
        
        return None
    except Exception as e:
        logging.error(f"Error getting document by ID: {e}")
        return None
    finally:
        mongodb.close()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Load data to Pinecone
    load_data_to_pinecone() 