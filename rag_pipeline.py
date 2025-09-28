from retrieval_engine import RetrievalEngine
from groq import Groq
import os
from typing import Dict, List
import numpy as np

class RAGPipeline:
    def __init__(self, retrieval_engine: RetrievalEngine, api_key: str = None):
        self.retrieval_engine = retrieval_engine
        
        # Get API key from parameter or environment variable
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found. Please set it in .env file or pass as parameter")
        
        self.client = Groq(api_key=self.api_key)
        
        # Use correct model names - try these available models
        self.available_models = [
            "llama3-8b-8192",      # Fast and efficient
            "llama3-70b-8192",     # More powerful but slower
            "mixtral-8x7b-32768",  # Good balance
            "gemma2-9b-it"         # Alternative option
        ]
        self.model = self.available_models[0]  # Start with llama3-8b-8192
        
        self.prompt_template = """You are an HR assistant. Use the following context to answer the question. 
If the context doesn't contain the answer, say you don't know. Be precise and helpful.

Context:
{context}

Question: {question}

Answer:"""
    
    def format_context(self, results: List[Dict]) -> str:
        """
        Format retrieved documents into context string
        """
        context_parts = []
        for i, result in enumerate(results):
            doc = result['document']
            context_parts.append(
                f"Source {i+1} (Section: {doc['section_heading']}):\n{doc['text']}\n"
            )
        return "\n".join(context_parts)
    
    def try_different_models(self, prompt: str):
        """
        Try different models if one fails
        """
        for model in self.available_models:
            try:
                print(f"Trying model: {model}")
                response = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                    temperature=0.1,
                    max_tokens=500
                )
                return response, model
            except Exception as e:
                print(f"Model {model} failed: {e}")
                continue
        raise Exception("All models failed")
    
    def generate_answer(self, query: str, max_context_length: int = 2000) -> Dict:
        """
        Generate answer using RAG pipeline
        """
        try:
            # Retrieve relevant documents
            retrieval_results = self.retrieval_engine.retrieve(query)
            
            # Check if this is a cached response
            if retrieval_results.get('from_cache', False) and 'answer' in retrieval_results:
                return retrieval_results
            
            # For non-cached responses, we need to generate the answer
            if not retrieval_results.get('results'):
                return {
                    'query': query,
                    'answer': "I couldn't find relevant information in the HR policies to answer your question.",
                    'sources': [],
                    'from_cache': False,
                    'error': False
                }
            
            # Format context
            context = self.format_context(retrieval_results['results'])
            
            # Truncate context if too long
            if len(context) > max_context_length:
                context = context[:max_context_length] + "..."
            
            # Create prompt
            prompt = self.prompt_template.format(
                context=context,
                question=query
            )
            
            # Call Groq API with model fallback
            response, used_model = self.try_different_models(prompt)
            print(f"Successfully used model: {used_model}")
            
            answer = response.choices[0].message.content
            
            # Prepare sources
            sources = []
            for result in retrieval_results['results']:
                doc = result['document']
                source_data = {
                    'section': str(doc.get('section_heading', 'Unknown Section')),
                    'content': str(doc.get('text', '')[:200]) + "...",
                    'score': float(result.get('combined_score', 0.0))
                }
                sources.append(source_data)
            
            result_dict = {
                'query': query,
                'answer': str(answer),
                'sources': sources,
                'from_cache': False,
                'error': False,
                'model_used': used_model  # For debugging
            }
            
            return result_dict
            
        except Exception as e:
            print(f"Error in generate_answer: {str(e)}")
            
            # Return a fallback answer using the retrieved context
            if 'retrieval_results' in locals() and retrieval_results.get('results'):
                # We have context but LLM failed - create a simple answer
                context_snippets = [result['document']['text'][:100] + "..." 
                                  for result in retrieval_results['results'][:2]]
                fallback_answer = f"I found relevant information about your question '{query}'. Key points: {', '.join(context_snippets)}"
                
                sources = []
                for result in retrieval_results['results']:
                    doc = result['document']
                    source_data = {
                        'section': str(doc.get('section_heading', 'Unknown Section')),
                        'content': str(doc.get('text', '')[:200]) + "...",
                        'score': float(result.get('combined_score', 0.0))
                    }
                    sources.append(source_data)
                
                return {
                    'query': query,
                    'answer': fallback_answer,
                    'sources': sources,
                    'from_cache': False,
                    'error': False,
                    'note': 'LLM service unavailable, using retrieved context'
                }
            else:
                return {
                    'query': query,
                    'answer': f"I encountered an error: {str(e)}",
                    'sources': [],
                    'from_cache': False,
                    'error': True
                }