"""
MongoDB connector for the Cheese Recommendation Agent.
"""
from typing import Dict, List, Any, Optional
import json
from pymongo import MongoClient
from pymongo.collection import Collection

from src.config import MONGODB_URI, MONGODB_DB, MONGODB_COLLECTION


class MongoDBConnector:
    """MongoDB connector for accessing cheese data."""
    
    def __init__(self, uri: str = MONGODB_URI, db_name: str = MONGODB_DB, 
                 collection_name: str = MONGODB_COLLECTION):
        """
        Initialize MongoDB connector.
        
        Args:
            uri: MongoDB connection URI
            db_name: MongoDB database name
            collection_name: MongoDB collection name
        """
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
    
    def close(self):
        """Close MongoDB connection."""
        self.client.close() 