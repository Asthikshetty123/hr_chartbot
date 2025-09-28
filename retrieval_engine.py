from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Dict, Tuple
import hashlib
import json
import time
import os
from embedding_manager import EmbeddingManager

class RetrievalEngine:
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self.bm25_index = None
        self.cache = {}
        self.cache_file = "query_cache.json"
        self.load_cache()
        
    def build_bm25_index(self):
        """
        Build BM25 index for keyword-based re-ranking
        """
        texts = [doc['text'] for doc in self.embedding_manager.documents]
        tokenized_texts = [text.lower().split() for text in texts]
        self.bm25_index = BM25Okapi(tokenized_texts)
    
    def rerank_results(self, query: str, initial_results: List[Tuple[Dict, float]], 
                      top_k: int = 3) -> List[Dict]:
        """
        Re-rank results using BM25 and cosine similarity combination
        """
        if not self.bm25_index:
            self.build_bm25_index()
        
        query_tokens = query.lower().split()
        reranked_results = []
        
        for doc, cosine_score in initial_results:
            # Get BM25 score
            doc_index = self.embedding_manager.documents.index(doc)
            bm25_score = self.bm25_index.get_scores(query_tokens)[doc_index]
            
            # Combine scores (you can adjust weights)
            combined_score = 0.7 * cosine_score + 0.3 * (bm25_score / 10)  # Normalize BM25
            
            reranked_results.append({
                'document': doc,
                'cosine_score': float(cosine_score),  # Convert to float
                'bm25_score': float(bm25_score),      # Convert to float
                'combined_score': float(combined_score)  # Convert to float
            })
        
        # Sort by combined score
        reranked_results.sort(key=lambda x: x['combined_score'], reverse=True)
        return reranked_results[:top_k]
    
    def get_cache_key(self, query: str) -> str:
        """
        Generate unique cache key for query
        """
        return hashlib.md5(query.encode()).hexdigest()
    
    def load_cache(self):
        """
        Load cache from file - handle corrupt files
        """
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    content = f.read().strip()
                    if content:  # Check if file is not empty
                        self.cache = json.loads(content)
                    else:
                        self.cache = {}
                        print("Cache file is empty, starting with empty cache")
            else:
                self.cache = {}
                print("No cache file found, starting with empty cache")
                
        except (json.JSONDecodeError, Exception) as e:
            print(f"Warning: Cache file is corrupted. Starting with empty cache. Error: {e}")
            self.cache = {}
            # Optionally backup the corrupt file
            if os.path.exists(self.cache_file):
                backup_name = f"{self.cache_file}.corrupted.{int(time.time())}"
                os.rename(self.cache_file, backup_name)
                print(f"Backed up corrupt cache file as {backup_name}")
    
    def save_cache(self):
        """
        Save cache to file
        """
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def get_cached_response(self, query: str) -> Dict | None:
        """
        Get cached response for query
        """
        cache_key = self.get_cache_key(query)
        cached_data = self.cache.get(cache_key)
        
        if cached_data and time.time() - cached_data['timestamp'] < 3600:  # 1 hour cache
            return cached_data['response']
        return None
    
    def cache_response(self, query: str, response: Dict):
        """
        Cache response for query
        """
        try:
            cache_key = self.get_cache_key(query)
            self.cache[cache_key] = {
                'timestamp': time.time(),
                'response': response,
                'query': query
            }
            self.save_cache()
        except Exception as e:
            print(f"Error caching response: {e}")
    
    def retrieve(self, query: str, top_k: int = 3) -> Dict:  # Changed return type to Dict
        """
        Main retrieval function with caching and re-ranking
        """
        # Check cache first
        cached_response = self.get_cached_response(query)
        if cached_response:
            # Ensure cached response has all required fields
            cached_response['from_cache'] = True
            if 'error' not in cached_response:
                cached_response['error'] = False
            return cached_response
        
        # Perform initial FAISS search
        initial_results = self.embedding_manager.similarity_search(query, k=top_k*2)
        
        # Re-rank results
        reranked_results = self.rerank_results(query, initial_results, top_k=top_k)
        
        # Prepare final results with all required fields
        final_results = {
            'query': query,
            'results': reranked_results,
            'from_cache': False,
            'error': False
        }
        
        # Cache the results
        self.cache_response(query, final_results)
        
        return final_results