import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Optional


class DatabaseManager:
    """Database manager for RAG system"""
    
    def __init__(self):
        self.db_path = "data/jupiter_rag.db"
        self.connection = None
        self._ensure_data_directory()
    
    def _ensure_data_directory(self):
        """Ensure data directory exists"""
        os.makedirs('data', exist_ok=True)
    
    def connect(self):
        """Connect to SQLite database"""
        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row
            return True
        except Exception as e:
            print(f"❌ Database connection error: {e}")
            return False
    
    def create_tables(self):
        """Create necessary tables"""
        if not self.connection:
            if not self.connect():
                return False
        
        try:
            cursor = self.connection.cursor()
            
            # Scraped content table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scraped_content (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT NOT NULL UNIQUE,
                    title TEXT,
                    content_hash TEXT UNIQUE,
                    category TEXT DEFAULT 'general',
                    keywords TEXT,
                    content_length INTEGER,
                    word_count INTEGER,
                    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    user_question TEXT NOT NULL,
                    ai_response TEXT NOT NULL,
                    sources_used TEXT,
                    relevance_score REAL DEFAULT 0.0,
                    response_time REAL DEFAULT 0.0,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # User feedback table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER,
                    feedback_type TEXT CHECK(feedback_type IN ('thumbs_up', 'thumbs_down')),
                    feedback_text TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (id)
                )
            """)
            
            self.connection.commit()
            print("✅ Database tables created")
            return True
            
        except Exception as e:
            print(f"❌ Error creating tables: {e}")
            return False
    
    def insert_scraped_content(self, content_data: Dict) -> bool:
        """Insert scraped content metadata"""
        if not self.connection:
            return False
        
        try:
            cursor = self.connection.cursor()
            keywords_json = json.dumps(content_data.get('keywords', []))
            
            cursor.execute("""
                INSERT OR REPLACE INTO scraped_content 
                (url, title, content_hash, category, keywords, 
                 content_length, word_count, scraped_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                content_data['url'],
                content_data['title'],
                content_data['content_hash'],
                content_data['category'],
                keywords_json,
                content_data['length'],
                content_data['word_count'],
                content_data['scraped_at']
            ))
            
            self.connection.commit()
            return True
            
        except Exception as e:
            print(f"❌ Error inserting content: {e}")
            return False
    
    def save_scraped_data_to_db(self, scraped_data: List[Dict]) -> bool:
        """Save all scraped data to database - MISSING METHOD ADDED"""
        if not scraped_data:
            print("❌ No data to save to database")
            return False
        
        if not self.connection:
            print("❌ No database connection")
            return False
        
        success_count = 0
        total_count = len(scraped_data)
        
        print(f"💾 Saving {total_count} items to database...")
        
        for i, content in enumerate(scraped_data):
            try:
                if self.insert_scraped_content(content):
                    success_count += 1
                
                # Progress update every 5 items
                if (i + 1) % 5 == 0 or (i + 1) == total_count:
                    print(f"💾 Saved {i + 1}/{total_count} items...")
                    
            except Exception as e:
                print(f"⚠️ Error saving item {i + 1}: {e}")
                continue
        
        print(f"✅ Successfully saved {success_count}/{total_count} content items to database")
        return success_count > 0
    
    def log_conversation(self, session_id: str, question: str, response: str, 
                        sources: List[str], relevance_score: float, response_time: float) -> int:
        """Log conversation to database"""
        if not self.connection:
            return 0
        
        try:
            cursor = self.connection.cursor()
            sources_json = json.dumps(sources)
            
            cursor.execute("""
                INSERT INTO conversations 
                (session_id, user_question, ai_response, sources_used, 
                 relevance_score, response_time)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (session_id, question, response, sources_json, relevance_score, response_time))
            
            self.connection.commit()
            return cursor.lastrowid
            
        except Exception as e:
            print(f"❌ Error logging conversation: {e}")
            return 0
    
    def log_feedback(self, conversation_id: int, feedback_type: str, feedback_text: str = None) -> bool:
        """Log user feedback"""
        if not self.connection:
            return False
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO user_feedback (conversation_id, feedback_type, feedback_text)
                VALUES (?, ?, ?)
            """, (conversation_id, feedback_type, feedback_text))
            
            self.connection.commit()
            return True
            
        except Exception as e:
            print(f"❌ Error logging feedback: {e}")
            return False
    
    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Get conversation history"""
        if not self.connection:
            return []
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT user_question, ai_response, timestamp, sources_used
                FROM conversations 
                WHERE session_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (session_id, limit))
            
            results = []
            for row in cursor.fetchall():
                sources = json.loads(row['sources_used']) if row['sources_used'] else []
                results.append({
                    'question': row['user_question'],
                    'response': row['ai_response'],
                    'timestamp': row['timestamp'],
                    'sources': sources
                })
            
            return list(reversed(results))  # Return chronological order
            
        except Exception as e:
            print(f"❌ Error getting conversation history: {e}")
            return []
    
    def get_analytics(self) -> Dict:
        """Get system analytics"""
        if not self.connection:
            return {}
        
        try:
            cursor = self.connection.cursor()
            
            # Basic stats
            cursor.execute("SELECT COUNT(*) as total_conversations FROM conversations")
            total_conversations = cursor.fetchone()['total_conversations']
            
            cursor.execute("SELECT COUNT(*) as total_content FROM scraped_content")
            total_content = cursor.fetchone()['total_content']
            
            cursor.execute("SELECT AVG(relevance_score) as avg_relevance FROM conversations")
            avg_relevance = cursor.fetchone()['avg_relevance'] or 0
            
            cursor.execute("SELECT AVG(response_time) as avg_response_time FROM conversations")
            avg_response_time = cursor.fetchone()['avg_response_time'] or 0
            
            return {
                'total_conversations': total_conversations,
                'total_content': total_content,
                'avg_relevance': round(avg_relevance, 2),
                'avg_response_time': round(avg_response_time, 3)
            }
            
        except Exception as e:
            print(f"❌ Error getting analytics: {e}")
            return {}
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
