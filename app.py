import streamlit as st
import os
import sys
import json
from datetime import datetime
import time
import warnings


# Suppress PyTorch warnings
warnings.filterwarnings("ignore", message=".*torch.*")
warnings.filterwarnings("ignore", message=".*Examining the path.*")


# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from utils.config import Config
from utils.scraper import JupiterWebScraper, run_scraping_with_progress
from utils.rag_engine import JupiterRAGEngine
from utils.database import DatabaseManager
from utils.helpers import (
    generate_session_id, format_sources, validate_question,
    truncate_text, format_response_time, get_category_display,
    create_conversation_export, calculate_relevance_color,
    get_jupiter_quick_questions, suggest_follow_up_questions
)


# Page configuration
st.set_page_config(
    page_title="Jupiter AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .chat-message {
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        border-radius: 12px;
    }
    
    .user-message {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    .ai-message {
        background: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    
    .source-card {
        background: #fff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
    }
    
    .quick-question-btn {
        margin: 0.25rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        background: #e9ecef;
        border: 1px solid #dee2e6;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .quick-question-btn:hover {
        background: #667eea;
        color: white;
        border-color: #667eea;
    }
    
    .system-status {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .status-ready { background: #d4edda; color: #155724; }
    .status-loading { background: #fff3cd; color: #856404; }
    .status-error { background: #f8d7da; color: #721c24; }
    
    .ai-ready-banner {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        font-size: 1.2rem;
        font-weight: bold;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


class JupiterAIAssistant:
    """Main application class for Jupiter AI Assistant"""
    
    def __init__(self):
        self.config = Config()
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'session_id' not in st.session_state:
            st.session_state.session_id = generate_session_id()
        
        if 'rag_engine' not in st.session_state:
            st.session_state.rag_engine = None
        
        if 'system_ready' not in st.session_state:
            st.session_state.system_ready = False
        
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        
        if 'current_question' not in st.session_state:
            st.session_state.current_question = ""
        
        # CRITICAL FIX: Add processing state tracking
        if 'processing_question' not in st.session_state:
            st.session_state.processing_question = None
        
        if 'last_processed_question' not in st.session_state:
            st.session_state.last_processed_question = ""
        
        # NEW: Track if we just built the AI system
        if 'ai_just_built' not in st.session_state:
            st.session_state.ai_just_built = False
    
    def render_header(self):
        """Render main header"""
        st.markdown("""
        <div class="main-header">
            <h1 style="margin: 0; font-size: 3rem;">🤖 Jupiter AI Assistant</h1>
            <p style="margin: 1rem 0 0 0; opacity: 0.9; font-size: 1.3rem;">
                Your intelligent guide to Jupiter's financial services
            </p>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.8; font-size: 1rem;">
                Ask me anything about Jupiter's banking, investments, and features!
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar controls"""
        with st.sidebar:
            st.header("⚙️ System Controls")
            
            # System status
            self.show_system_status()
            
            st.divider()
            
            # Data management
            st.subheader("📊 Data Management")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔍 Scrape Data", use_container_width=True, type="primary"):
                    self.run_data_scraping()
            
            with col2:
                data_exists = os.path.exists('data/scraped_content.json')
                if st.button("🏗️ Build AI", use_container_width=True, disabled=not data_exists):
                    self.build_rag_system()
            
            # Data status
            if os.path.exists('data/scraped_content.json'):
                with open('data/scraped_content.json', 'r') as f:
                    data = json.load(f)
                st.success(f"✅ {len(data)} pages available")
            else:
                st.warning("⚠️ No data found")
            
            st.divider()
            
            # System information
            if st.session_state.rag_engine:
                self.show_system_metrics()
            
            st.divider()
            
            # Conversation management
            st.subheader("💬 Conversation")
            
            if st.session_state.conversation_history:
                if st.button("📥 Export Chat", use_container_width=True):
                    export_data = create_conversation_export(st.session_state.conversation_history)
                    st.download_button(
                        "Download CSV",
                        export_data,
                        f"jupiter_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv"
                    )
                
                if st.button("🗑️ Clear Chat", use_container_width=True):
                    st.session_state.conversation_history = []
                    st.session_state.last_processed_question = ""  # Reset this too
                    st.success("Chat cleared!")
                    st.rerun()
    
    def show_system_status(self):
        """Display system status"""
        st.subheader("📊 System Status")
        
        # Check OpenAI API key
        api_key_status = "✅ Ready" if self.config.get_openai_api_key() else "❌ Missing"
        st.markdown(f"**OpenAI API:** {api_key_status}")
        
        # Check data
        data_status = "✅ Ready" if os.path.exists('data/scraped_content.json') else "❌ Missing"
        st.markdown(f"**Data:** {data_status}")
        
        # Check RAG system
        rag_status = "✅ Ready" if st.session_state.system_ready else "❌ Not Ready"
        st.markdown(f"**AI System:** {rag_status}")
        
        # Overall status
        if st.session_state.system_ready and self.config.get_openai_api_key():
            st.markdown('<div class="system-status status-ready">🟢 System Ready</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown('<div class="system-status status-error">🔴 System Not Ready</div>', 
                       unsafe_allow_html=True)
    
    def show_system_metrics(self):
        """Show system metrics - UPDATED: Removed Avg Relevance"""
        try:
            info = st.session_state.rag_engine.get_system_info()
            
            st.subheader("📈 Metrics")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Knowledge Base", f"{info.get('knowledge_base_size', 0):,} chunks")
            with col2:
                analytics = info.get('analytics', {})
                st.metric("Total Questions Asked", f"{analytics.get('total_conversations', 0):,}")
            
            if analytics:
                col3, col4 = st.columns(2)
                with col3:
                    st.metric("Avg Response", f"{analytics.get('avg_response_time', 0):.1f}s")
                with col4:
                    st.metric("System Ready", "✅ Active")
                    
        except Exception as e:
            st.error(f"Error loading metrics: {e}")
    
    def run_data_scraping(self):
        """Run data scraping process"""
        st.info("🔄 Starting data scraping...")
        
        with st.spinner("Scraping Jupiter website..."):
            result = run_scraping_with_progress()
        
        if result.get('success'):
            st.success(f"✅ Successfully scraped {result['pages_scraped']} pages!")
            st.session_state.system_ready = False
            st.balloons()
        else:
            st.error(f"❌ Scraping failed: {result.get('error', 'Unknown error')}")
    
    def build_rag_system(self):
        """Build RAG system"""
        if not os.path.exists('data/scraped_content.json'):
            st.error("❌ No scraped data found. Please scrape data first.")
            return
        
        # Check OpenAI API key
        if not self.config.get_openai_api_key():
            st.error("❌ OpenAI API key not found. Please set your API key in Streamlit secrets.")
            st.info("Add your OpenAI API key in the Streamlit Cloud dashboard under 'Secrets'")
            return
        
        st.info("🔄 Building AI system...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(progress, message):
            progress_bar.progress(progress)
            status_text.text(message)
        
        try:
            # Initialize RAG engine
            rag_engine = JupiterRAGEngine()
            
            # Build knowledge base
            success = rag_engine.load_from_file(
                'data/scraped_content.json',
                progress_callback=update_progress
            )
            
            if success:
                st.session_state.rag_engine = rag_engine
                st.session_state.system_ready = True
                st.session_state.ai_just_built = True  # NEW: Mark as just built
                
                progress_bar.progress(1.0)
                status_text.text("✅ AI system ready!")
                st.success("🎉 Jupiter AI Assistant is ready!")
                st.balloons()
                
                # Auto-rerun to show chat interface
                time.sleep(2)  # Brief pause to see the success message
                st.rerun()
            else:
                st.error("❌ Failed to build AI system")
                
        except Exception as e:
            st.error(f"❌ Error building AI system: {str(e)}")
            if "api" in str(e).lower():
                st.info("This might be an API key issue. Please check your OpenAI API key.")
    
    def render_quick_questions(self):
        """Render quick question buttons - UPDATED to actually work"""
        st.subheader("💡 Quick Questions - Click to Get Answers!")
        
        questions = get_jupiter_quick_questions()
        
        cols = st.columns(2)
        for i, question in enumerate(questions):
            with cols[i % 2]:
                if st.button(question, key=f"quick_{i}", use_container_width=True):
                    # FIXED: Actually ask the question when clicked
                    st.session_state.current_question = ""  # Clear input
                    self.ask_question(question)
                    # Don't need st.rerun() as ask_question handles it
    
    def render_chat_interface(self):
        """Render main chat interface - UPDATED VERSION"""
        
        # NEW: Show AI Ready banner if just built
        if st.session_state.ai_just_built:
            st.markdown("""
            <div class="ai-ready-banner">
                🎉 AI System Ready! You can now start asking questions about Jupiter's services.
            </div>
            """, unsafe_allow_html=True)
            st.session_state.ai_just_built = False  # Reset flag
        
        st.header("💬 Chat with Jupiter AI")
        
        # Question input
        col1, col2 = st.columns([4, 1])
        
        with col1:
            question = st.text_input(
                "Ask me anything about Jupiter:",
                placeholder="e.g., How do I open a Jupiter savings account?",
                value=st.session_state.current_question,
                key="question_input"
            )
        
        with col2:
            st.write("")  # Spacing
            ask_clicked = st.button("🚀 Ask", type="primary", use_container_width=True)
        
        # CRITICAL FIX: Proper question handling to prevent loops
        if ask_clicked and question and question.strip():
            # Check if this exact question is already being processed
            if st.session_state.processing_question == question:
                st.warning("⏳ This question is already being processed. Please wait...")
            # Check if this exact question was just processed
            elif st.session_state.last_processed_question == question:
                st.info("💡 This question was already answered. Please ask a different question or scroll down to see the previous answer.")
            else:
                # Process the new question
                st.session_state.current_question = ""  # Clear input immediately
                self.ask_question(question)
        elif question and question != st.session_state.current_question:
            # Update current question for display purposes only
            st.session_state.current_question = question
        
        # Display conversation history
        if st.session_state.conversation_history:
            st.subheader("📜 Conversation History")
            
            for i, conv in enumerate(reversed(st.session_state.conversation_history)):
                self.render_conversation_item(conv, len(st.session_state.conversation_history) - i)
    
    def render_conversation_item(self, conv: dict, item_number: int):
        """Render a single conversation item - UPDATED: No relevance scores"""
        # User question
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>🙋 You:</strong> {conv['question']}
        </div>
        """, unsafe_allow_html=True)
        
        # AI response
        response_html = conv['response'].replace('\n', '<br>')
        st.markdown(f"""
        <div class="chat-message ai-message">
            <strong>🤖 Jupiter AI:</strong><br>
            {response_html}
        </div>
        """, unsafe_allow_html=True)
        
        # Metadata and sources - UPDATED: Only 2 columns, no relevance
        col1, col2 = st.columns(2)
        
        with col1:
            response_time_formatted = format_response_time(conv.get('response_time', 0))
            st.markdown(f"""
            <div class="metric-card">
                <strong>Response Time</strong><br>
                {response_time_formatted}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            sources_count = len(conv.get('sources', []))
            st.markdown(f"""
            <div class="metric-card">
                <strong>Sources Used</strong><br>
                {sources_count} sources
            </div>
            """, unsafe_allow_html=True)
        
        # Sources section - UPDATED: No relevance percentages
        if conv.get('sources'):
            sources_len = len(conv['sources'])
            with st.expander(f"📚 View Sources ({sources_len} sources)", expanded=False):
                for j, source in enumerate(conv['sources'][:3], 1):
                    source_title = source.get('title', 'Unknown Source')
                    source_url = source.get('url', 'N/A')
                    st.markdown(f"""
                    <div class="source-card">
                        <strong>{j}. {source_title}</strong><br>
                        <small>URL: {source_url}</small>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Feedback buttons
        col_feedback1, col_feedback2, col_feedback3 = st.columns([1, 1, 2])
        
        conversation_id = conv.get('conversation_id')
        
        with col_feedback1:
            if st.button("👍", key=f"thumbs_up_{item_number}_{conversation_id}"):
                self.log_feedback(conversation_id, 'thumbs_up')
                st.success("Thanks for the feedback!")
        
        with col_feedback2:
            if st.button("👎", key=f"thumbs_down_{item_number}_{conversation_id}"):
                self.log_feedback(conversation_id, 'thumbs_down')
                st.success("Feedback logged. We'll improve!")
        
        with col_feedback3:
            # Suggest follow-up questions - FIXED to prevent loops
            suggestions = suggest_follow_up_questions(conv['question'], 'general')
            if suggestions:
                selected_suggestion = st.selectbox(
                    "Ask a follow-up:",
                    [""] + suggestions[:3],
                    key=f"followup_{item_number}_{conversation_id}"
                )
                if selected_suggestion and selected_suggestion != st.session_state.get('last_followup'):
                    st.session_state.last_followup = selected_suggestion
                    self.ask_question(selected_suggestion)
        
        st.divider()
    
    def ask_question(self, question: str):
        """Process a question through RAG - COMPLETELY FIXED VERSION"""
        # Validate question
        is_valid, message = validate_question(question)
        if not is_valid:
            st.error(f"❌ {message}")
            return
        
        # Check system readiness
        if not st.session_state.system_ready or not st.session_state.rag_engine:
            st.error("⚠️ AI system not ready. Please build the AI system first.")
            return
        
        # CRITICAL FIX: Prevent duplicate processing
        if st.session_state.processing_question == question:
            st.warning("⏳ This question is already being processed...")
            return
        
        if st.session_state.last_processed_question == question:
            st.info("💡 This question was already answered. Check the conversation history below.")
            return
        
        # Mark as processing
        st.session_state.processing_question = question
        
        # Show processing message
        with st.spinner("🤖 Thinking..."):
            try:
                # Get AI response
                result = st.session_state.rag_engine.ask_question(
                    question, 
                    st.session_state.session_id
                )
                
                # Add to conversation history
                conversation_item = {
                    'question': question,
                    'response': result['response'],
                    'sources': result['sources'],
                    'relevance_score': result['relevance_score'],
                    'response_time': result['response_time'],
                    'conversation_id': result.get('conversation_id'),
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                st.session_state.conversation_history.append(conversation_item)
                
                # Mark as processed and clear processing state
                st.session_state.last_processed_question = question
                st.session_state.processing_question = None
                st.session_state.current_question = ""
                
                # Show success message
                st.success(f"✅ Response generated in {format_response_time(result['response_time'])}")
                
                # Rerun to show new conversation
                st.rerun()
                
            except Exception as e:
                # Clear processing state on error
                st.session_state.processing_question = None
                
                st.error(f"❌ Error generating response: {str(e)}")
                if "api" in str(e).lower():
                    st.info("This might be an API quota or key issue. Please check your OpenAI account.")
    
    def log_feedback(self, conversation_id: int, feedback_type: str):
        """Log user feedback"""
        if st.session_state.rag_engine and conversation_id:
            try:
                st.session_state.rag_engine.db_manager.log_feedback(
                    conversation_id, feedback_type
                )
            except Exception as e:
                print(f"Error logging feedback: {e}")
    
    def render_footer(self):
        """Render footer"""
        st.divider()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🤖 About This AI
            - **RAG Technology**: Retrieval-Augmented Generation
            - **LLM Powered**: OpenAI GPT for responses
            - **Source Grounded**: Every answer cites sources
            """)
        
        with col2:
            st.markdown("""
            ### ✅ Key Features
            - **Conversational AI**: Natural language interaction
            - **Real-time Answers**: Instant responses
            - **Source Attribution**: Transparent information
            - **Continuous Learning**: Improves with feedback
            """)
        
        with col3:
            st.markdown("""
            ### 🏦 Jupiter Focus
            - Banking and financial services
            - Account management and features
            - Investment and savings options
            - Customer support and help
            """)
        
        st.markdown("""
        ---
        <div style='text-align: center; color: gray; padding: 1rem;'>
            <p>🤖 Jupiter AI Assistant | Powered by RAG + OpenAI GPT</p>
            <p>Ask me anything about Jupiter's financial services!</p>
        </div>
        """, unsafe_allow_html=True)
    
    def run(self):
        """Main application runner - UPDATED"""
        # Check for OpenAI API key on startup
        if not self.config.get_openai_api_key():
            st.error("⚠️ OpenAI API key not found!")
            st.info("""
            Please add your OpenAI API key:
            1. Go to Streamlit Cloud dashboard
            2. Click on your app settings
            3. Add to secrets: `openai.api_key = "your-api-key-here"`
            4. Or set environment variable: `OPENAI_API_KEY`
            """)
            return
        
        # Render components
        self.render_header()
        
        # Main content
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if not st.session_state.system_ready:
                # UPDATED: Show setup instructions and working FAQs
                st.warning("🔧 System not ready. Please scrape data and build AI system first.")
                
                # Show setup steps
                st.markdown("""
                ### 📋 Setup Steps:
                1. **🔍 Scrape Data** - Collect Jupiter website content
                2. **🏗️ Build AI** - Create the RAG system
                3. **💬 Start Chatting** - Ask questions about Jupiter!
                """)
                
                # Show working FAQ buttons
                self.render_quick_questions()
            else:
                # UPDATED: Go directly to chat interface when ready
                self.render_chat_interface()
        
        # Sidebar
        self.render_sidebar()
        
        # Footer
        self.render_footer()


def main():
    """Application entry point"""
    try:
        app = JupiterAIAssistant()
        app.run()
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        st.info("Please refresh the page and try again.")
        
        # Show detailed error in expander for debugging
        with st.expander("🔍 Technical Details"):
            st.code(str(e))


if __name__ == "__main__":
    main()
