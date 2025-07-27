import sqlite3
import pickle
import hashlib
import os
from typing import Optional, List, Tuple
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel


class EmbeddingCache:
    """SQLite cache for storing pre-calculated string embeddings."""
    
    def __init__(self, cache_path: str = "embeddings_cache.db"):
        self.cache_path = cache_path
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database with the embeddings table."""
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    text_hash TEXT PRIMARY KEY,
                    text_content TEXT NOT NULL,
                    embedding_data BLOB NOT NULL,
                    model_name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    def _hash_text(self, text: str) -> str:
        """Create a hash of the text for use as a key."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def _serialize_embedding(self, embedding: torch.Tensor) -> bytes:
        """Serialize embedding tensor to bytes for storage."""
        return pickle.dumps(embedding.detach().cpu().numpy())
    
    def _deserialize_embedding(self, embedding_data: bytes) -> torch.Tensor:
        """Deserialize bytes back to embedding tensor."""
        numpy_array = pickle.loads(embedding_data)
        return torch.from_numpy(numpy_array)
    
    def get(self, text: str, model_name: str) -> Optional[torch.Tensor]:
        """Retrieve embedding from cache if it exists."""
        text_hash = self._hash_text(text)
        
        with sqlite3.connect(self.cache_path) as conn:
            cursor = conn.execute(
                "SELECT embedding_data FROM embeddings WHERE text_hash = ? AND model_name = ?",
                (text_hash, model_name)
            )
            result = cursor.fetchone()
            
            if result:
                return self._deserialize_embedding(result[0])
            return None
    
    def set(self, text: str, embedding: torch.Tensor, model_name: str):
        """Store embedding in cache."""
        text_hash = self._hash_text(text)
        embedding_data = self._serialize_embedding(embedding)
        
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO embeddings 
                (text_hash, text_content, embedding_data, model_name) 
                VALUES (?, ?, ?, ?)
                """,
                (text_hash, text, embedding_data, model_name)
            )
            conn.commit()
    
    def clear(self):
        """Clear all cached embeddings."""
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute("DELETE FROM embeddings")
            conn.commit()
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        with sqlite3.connect(self.cache_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
            total_embeddings = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(DISTINCT model_name) FROM embeddings")
            unique_models = cursor.fetchone()[0]
            
            return {
                "total_embeddings": total_embeddings,
                "unique_models": unique_models,
                "cache_size_mb": os.path.getsize(self.cache_path) / (1024 * 1024) if os.path.exists(self.cache_path) else 0
            }


class CachedBERTEncoder:
    """BERT encoder with SQLite caching for embeddings."""
    
    def __init__(self, model_name: str = "bert-base-uncased", cache_path: str = "output/embeddings_cache.db"):
        self.model_name = model_name
        self.cache = EmbeddingCache(cache_path)
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode
    
    def encode(self, text: str, use_cache: bool = True) -> torch.Tensor:
        """
        Encode text to embedding, using cache if available.
        
        Args:
            text: Text to encode
            use_cache: Whether to use caching (default: True)
            
        Returns:
            torch.Tensor: The embedding vector
        """
        if use_cache:
            # Try to get from cache first
            cached_embedding = self.cache.get(text, self.model_name)
            if cached_embedding is not None:
                return cached_embedding
        
        # Calculate embedding
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
        
        # Cache the result
        if use_cache:
            self.cache.set(text, embedding, self.model_name)
        
        return embedding
    
    def encode_batch(self, texts: List[str], use_cache: bool = True) -> List[torch.Tensor]:
        """
        Encode a batch of texts, using cache where possible.
        
        Args:
            texts: List of texts to encode
            use_cache: Whether to use caching (default: True)
            
        Returns:
            List[torch.Tensor]: List of embedding vectors
        """
        embeddings = []
        
        for text in texts:
            embedding = self.encode(text, use_cache=use_cache)
            embeddings.append(embedding)
        
        return embeddings
    
    def get_cache_stats(self) -> dict:
        """Get statistics about the cache."""
        return self.cache.get_stats()
    
    def clear_cache(self):
        """Clear all cached embeddings."""
        self.cache.clear()



