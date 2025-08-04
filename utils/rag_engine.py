import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import json
import os
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime
import tiktoken
import streamlit as st
from .config import Config
from .database import DatabaseManager


class JupiterRAGEngine:
    """Complete RAG system with LLM for Jupiter Q&A - FIXED VERSION"""
    
    # Emergency fallback threshold if config is too high
    DEFAULT_SIMILARITY_THRESHOLD = 0.3
    
    def __init__(self):
        self.config = Config()
        self.db_manager = DatabaseManager()
        
        # Initialize OpenAI client
        api_key = self.config.get_openai_api_key()
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY.")
        
        self.openai_client = OpenAI(api_key=api_key)
        
        # Initialize embedding model
        print("🔄 Loading embedding model...")
        self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
        print("✅ Embedding model loaded")
        
        # Setup ChromaDB
        self._setup_chromadb()
        
        # Setup database
        if self.db_manager.connect():
            self.db_manager.create_tables()
        
        # Token encoder for managing context length
        self.tokenizer = tiktoken.encoding_for_model(self.config.LLM_MODEL)
    
    def _setup_chromadb(self):
        """Setup ChromaDB client and collection"""
        try:
            os.makedirs(self.config.CHROMA_PERSIST_DIR, exist_ok=True)
            
            self.chroma_client = chromadb.PersistentClient(
                path=self.config.CHROMA_PERSIST_DIR
            )
            
            self.collection = self.chroma_client.get_or_create_collection(
                name="jupiter_rag_collection",
                metadata={"description": "Jupiter RAG with LLM collection"}
            )
            
            print(f"✅ ChromaDB initialized with {self.collection.count()} documents")
            
        except Exception as e:
            print(f"❌ ChromaDB setup error: {e}")
            raise
    
    def _chunk_content(self, content: str) -> List[str]:
        """Split content into chunks for RAG"""
        if len(content) <= self.config.CHUNK_SIZE:
            return [content]
        
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + self.config.CHUNK_SIZE
            
            # Find sentence boundary
            if end < len(content):
                for i in range(end, max(start + self.config.CHUNK_SIZE - 200, start + self.config.CHUNK_SIZE // 2), -1):
                    if content[i:i+2] in ['. ', '! ', '? ', '.\n']:
                        end = i + 1
                        break
            
            chunk = content[start:end].strip()
            
            if chunk and len(chunk) > 100:
                chunks.append(chunk)
            
            start = end - self.config.CHUNK_OVERLAP
            if start >= len(content):
                break
        
        return chunks
    
    def build_knowledge_base(self, scraped_data: List[Dict], progress_callback=None) -> bool:
        """Build RAG knowledge base from scraped data"""
        if not scraped_data:
            print("❌ No data provided for knowledge base")
            return False
        
        try:
            print(f"🔄 Building knowledge base from {len(scraped_data)} documents...")
            
            all_chunks = []
            all_metadatas = []
            all_ids = []
            
            total_docs = len(scraped_data)
            
            for doc_idx, doc in enumerate(scraped_data):
                if progress_callback:
                    progress = (doc_idx + 1) / total_docs
                    progress_callback(progress, f"Processing document {doc_idx + 1}/{total_docs}")
                
                content = doc.get('content', '')
                if not content or len(content) < 150:
                    continue
                
                # Create chunks
                chunks = self._chunk_content(content)
                
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_id = f"doc_{doc_idx}_chunk_{chunk_idx}"
                    
                    metadata = {
                        'source_url': doc.get('url', ''),
                        'title': doc.get('title', 'Untitled'),
                        'category': doc.get('category', 'general'),
                        'chunk_index': chunk_idx,
                        'total_chunks': len(chunks),
                        'keywords': ','.join(doc.get('keywords', [])),
                        'doc_index': doc_idx
                    }
                    
                    all_chunks.append(chunk)
                    all_metadatas.append(metadata)
                    all_ids.append(chunk_id)
            
            print(f"📊 Created {len(all_chunks)} knowledge chunks")
            
            # Add to ChromaDB in batches
            batch_size = 50
            total_batches = (len(all_chunks) + batch_size - 1) // batch_size
            
            for i in range(0, len(all_chunks), batch_size):
                batch_chunks = all_chunks[i:i + batch_size]
                batch_metadatas = all_metadatas[i:i + batch_size]
                batch_ids = all_ids[i:i + batch_size]
                
                try:
                    self.collection.add(
                        documents=batch_chunks,
                        metadatas=batch_metadatas,
                        ids=batch_ids
                    )
                    
                    batch_num = i // batch_size + 1
                    print(f"📦 Added batch {batch_num}/{total_batches} to knowledge base")
                    
                    if progress_callback:
                        progress = min(batch_num / total_batches, 1.0)
                        progress_callback(progress, f"Building knowledge base: batch {batch_num}/{total_batches}")
                        
                except Exception as e:
                    print(f"⚠️ Error adding batch {batch_num}: {e}")
                    continue
            
            # Save to database
            self.db_manager.save_scraped_data_to_db(scraped_data)
            
            print("✅ Knowledge base built successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error building knowledge base: {e}")
            return False
    
    def retrieve_context(self, query: str) -> Tuple[str, List[Dict], float]:
        """Retrieve relevant context for the query - COMPLETELY FIXED"""
        try:
            # Check if collection exists and has documents
            if not self.collection or self.collection.count() == 0:
                print("❌ No documents in collection")
                return "", [], 0.0
            
            print(f"🔍 Querying ChromaDB for: {query}")
            
            # Query ChromaDB for relevant chunks
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=min(self.config.MAX_RETRIEVED_DOCS, self.collection.count()),
                    include=['documents', 'metadatas', 'distances']
                )
            except Exception as query_error:
                print(f"❌ ChromaDB query error: {query_error}")
                return "", [], 0.0
            
            # Validate results structure
            if not results or not results.get('documents') or not results['documents'][0]:
                print("❌ No matching documents found")
                return "", [], 0.0
            
            # Process results safely
            documents = results['documents'][0]
            metadatas = results.get('metadatas', [[]])[0] if results.get('metadatas') else []
            distances = results.get('distances', [[]])[0] if results.get('distances') else []
            
            print(f"📊 Processing {len(documents)} retrieved documents")
            
            # Use fallback threshold if config threshold is too high
            threshold = getattr(self.config, 'SIMILARITY_THRESHOLD', self.DEFAULT_SIMILARITY_THRESHOLD)
            if threshold > 0.8:  # If threshold is unreasonably high
                threshold = self.DEFAULT_SIMILARITY_THRESHOLD
                print(f"⚠️ Using fallback similarity threshold: {threshold}")
            
            retrieved_docs = []
            context_parts = []
            total_relevance = 0
            
            for i in range(len(documents)):
                try:
                    # Safe access to arrays
                    distance = distances[i] if i < len(distances) else 1.0
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    document = documents[i]
                    
                    # Convert distance to similarity (more generous conversion)
                    similarity_score = max(0, 1 - (distance / 1.5))  # More lenient than /2
                    
                    print(f"📊 Document {i+1}: similarity={similarity_score:.3f}, threshold={threshold:.3f}")
                    
                    # Filter by similarity threshold
                    if similarity_score >= threshold:
                        doc_info = {
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'source_url': metadata.get('source_url', ''),
                            'title': metadata.get('title', 'Unknown')
                        }
                        
                        retrieved_docs.append(doc_info)
                        total_relevance += similarity_score
                        
                        # Add to context with source attribution
                        context_parts.append(
                            f"[Source: {doc_info['title']}]\n{doc_info['content']}\n"
                        )
                        
                except Exception as doc_error:
                    print(f"⚠️ Error processing document {i}: {doc_error}")
                    continue
            
            # If no documents pass threshold, lower it and try again
            if not retrieved_docs and threshold > 0.1:
                print(f"⚠️ No documents passed threshold {threshold:.2f}, trying with 0.1")
                threshold = 0.1
                
                for i in range(len(documents)):
                    try:
                        distance = distances[i] if i < len(distances) else 1.0
                        metadata = metadatas[i] if i < len(metadatas) else {}
                        document = documents[i]
                        
                        similarity_score = max(0, 1 - (distance / 1.5))
                        
                        if similarity_score >= threshold:
                            doc_info = {
                                'content': document,
                                'metadata': metadata,
                                'similarity_score': similarity_score,
                                'source_url': metadata.get('source_url', ''),
                                'title': metadata.get('title', 'Unknown')
                            }
                            
                            retrieved_docs.append(doc_info)
                            total_relevance += similarity_score
                            
                            context_parts.append(
                                f"[Source: {doc_info['title']}]\n{doc_info['content']}\n"
                            )
                            
                    except Exception as doc_error:
                        continue
            
            # Combine context
            full_context = "\n---\n".join(context_parts)
            avg_relevance = total_relevance / len(retrieved_docs) if retrieved_docs else 0
            
            print(f"✅ Retrieved {len(retrieved_docs)} relevant documents (avg relevance: {avg_relevance:.2f})")
            
            return full_context, retrieved_docs, avg_relevance
            
        except Exception as e:
            print(f"❌ Error retrieving context: {e}")
            import traceback
            traceback.print_exc()
            return "", [], 0.0
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        try:
            return len(self.tokenizer.encode(text))
        except Exception:
            # Fallback to approximate token count
            return int(len(text.split()) * 1.3)
    
    def _truncate_context(self, context: str, max_tokens: int = 2000) -> str:
        """Truncate context to fit within token limit"""
        try:
            if self._count_tokens(context) <= max_tokens:
                return context
            
            # Split into parts and keep as much as possible
            parts = context.split("\n---\n")
            truncated_parts = []
            current_tokens = 0
            
            for part in parts:
                part_tokens = self._count_tokens(part)
                if current_tokens + part_tokens <= max_tokens:
                    truncated_parts.append(part)
                    current_tokens += part_tokens
                else:
                    break
            
            return "\n---\n".join(truncated_parts)
        except Exception as e:
            print(f"⚠️ Error truncating context: {e}")
            return context[:max_tokens * 4]  # Rough character-based fallback
    
    def generate_response(self, query: str, context: str, retrieved_docs: List[Dict]) -> Dict:
        """Generate AI response using OpenAI GPT"""
        try:
            # Truncate context if too long
            context = self._truncate_context(context)
            
            # Create system prompt
            system_prompt = """You are a helpful AI assistant for Jupiter, India's leading fintech company. Your role is to provide accurate, helpful, and friendly responses about Jupiter's banking services, features, and policies.

Instructions:
1. Use ONLY the provided context from Jupiter's official website to answer questions
2. Be conversational, helpful, and friendly in your tone
3. If the context doesn't contain enough information, politely say so and suggest contacting customer support
4. Focus on Jupiter's financial services, banking features, and customer benefits
5. Always be accurate - never make up information not in the context
6. If asked about competitors, politely redirect to Jupiter's offerings
7. Provide specific details when available (fees, features, requirements, etc.)
8. End responses with a helpful suggestion or next step when appropriate

Remember: You represent Jupiter's brand, so be professional yet approachable."""
            
            # Create user prompt with context
            user_prompt = f"""Context from Jupiter's website:
{context}

User Question: {query}

Please provide a comprehensive and helpful answer based on the context above. Be specific and actionable where possible."""
            
            # Generate response using OpenAI
            response = self.openai_client.chat.completions.create(
                model=self.config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.config.LLM_TEMPERATURE,
                max_tokens=self.config.MAX_TOKENS
            )
            
            ai_response = response.choices[0].message.content
            
            # Extract sources used
            sources_used = []
            for doc in retrieved_docs:
                source_info = {
                    'title': doc['title'],
                    'url': doc['source_url'],
                    'relevance': doc['similarity_score']
                }
                if source_info not in sources_used:
                    sources_used.append(source_info)
            
            return {
                'response': ai_response,
                'sources': sources_used,
                'context_used': len(context),
                'tokens_used': response.usage.total_tokens if hasattr(response, 'usage') else 0
            }
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            import traceback
            traceback.print_exc()
            return {
                'response': "I apologize, but I'm experiencing technical difficulties. Please try again or contact Jupiter customer support for assistance.",
                'sources': [],
                'context_used': 0,
                'tokens_used': 0
            }
    
    def ask_question(self, query: str, session_id: str = None) -> Dict:
        """Main RAG function - ask a question and get AI response - EMERGENCY FIXED VERSION"""
        start_time = datetime.now()
        max_attempts = 3  # CRITICAL: Limit attempts to prevent infinite loop
        
        for attempt in range(1, max_attempts + 1):
            try:
                print(f"🔄 Attempt {attempt}/{max_attempts} for query: {query}")
                
                # Step 1: Retrieve relevant context
                retrieval_start = datetime.now()
                context, retrieved_docs, avg_relevance = self.retrieve_context(query)
                retrieval_time = (datetime.now() - retrieval_start).total_seconds()
                
                # CRITICAL: If no context found, return immediately - DON'T RETRY
                if not context:
                    if attempt == max_attempts:  # Last attempt
                        return {
                            'query': query,
                            'response': "I couldn't find relevant information about your question in Jupiter's knowledge base. This might be because the similarity threshold is too high, or the question is outside Jupiter's domain. Please try rephrasing your question or contact Jupiter customer support for assistance.",
                            'sources': [],
                            'relevance_score': 0.0,
                            'response_time': (datetime.now() - start_time).total_seconds(),
                            'context_length': 0
                        }
                    print(f"❌ No context found on attempt {attempt}, trying again...")
                    continue  # Try next attempt
                
                # SUCCESS: We have context, generate response
                print("🤖 Generating AI response...")
                result = self.generate_response(query, context, retrieved_docs)
                
                # Calculate total response time
                response_time = (datetime.now() - start_time).total_seconds()
                
                # Prepare final result
                final_result = {
                    'query': query,
                    'response': result['response'],
                    'sources': result['sources'],
                    'relevance_score': avg_relevance,
                    'response_time': response_time,
                    'context_length': result['context_used'],
                    'tokens_used': result['tokens_used'],
                    'num_sources': len(retrieved_docs)
                }
                
                # Log conversation (best effort)
                if session_id:
                    try:
                        source_urls = [s['url'] for s in result['sources']]
                        conversation_id = self.db_manager.log_conversation(
                            session_id, query, result['response'], 
                            source_urls, avg_relevance, response_time
                        )
                        final_result['conversation_id'] = conversation_id
                    except Exception as log_error:
                        print(f"⚠️ Error logging conversation: {log_error}")
                
                print(f"✅ Response generated in {response_time:.2f}s")
                return final_result
                
            except Exception as e:
                print(f"❌ Error on attempt {attempt}: {e}")
                if attempt == max_attempts:  # Last attempt
                    return {
                        'query': query,
                        'response': "I apologize for the technical error. Please try again or contact Jupiter support.",
                        'sources': [],
                        'relevance_score': 0.0,
                        'response_time': (datetime.now() - start_time).total_seconds(),
                        'context_length': 0,
                        'error': str(e)
                    }
        
        # Should never reach here, but just in case
        return {
            'query': query,
            'response': "Unexpected error occurred.",
            'sources': [],
            'relevance_score': 0.0,
            'response_time': (datetime.now() - start_time).total_seconds(),
            'context_length': 0
        }
    
    def get_system_info(self) -> Dict:
        """Get RAG system information"""
        try:
            return {
                'knowledge_base_size': self.collection.count() if self.collection else 0,
                'embedding_model': self.config.EMBEDDING_MODEL,
                'llm_model': self.config.LLM_MODEL,
                'chunk_size': self.config.CHUNK_SIZE,
                'max_retrieved_docs': self.config.MAX_RETRIEVED_DOCS,
                'similarity_threshold': getattr(self.config, 'SIMILARITY_THRESHOLD', self.DEFAULT_SIMILARITY_THRESHOLD),
                'database_connected': self.db_manager.connection is not None,
                'analytics': self.db_manager.get_analytics()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def load_from_file(self, file_path: str = 'data/scraped_content.json', 
                      progress_callback=None) -> bool:
        """Load data and build knowledge base"""
        try:
            if not os.path.exists(file_path):
                print(f"❌ Data file not found: {file_path}")
                return False
            
            with open(file_path, 'r', encoding='utf-8') as f:
                scraped_data = json.load(f)
            
            if not scraped_data:
                print("❌ No data found in file")
                return False
            
            print(f"📚 Loaded {len(scraped_data)} documents from {file_path}")
            return self.build_knowledge_base(scraped_data, progress_callback)
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
