import uuid
import re
from datetime import datetime
from typing import List, Dict
import streamlit as st

def generate_session_id() -> str:
    """Generate unique session ID"""
    return str(uuid.uuid4())

def format_sources(sources: List[Dict]) -> str:
    """Format sources for display"""
    if not sources:
        return "No sources"
    
    formatted = []
    for i, source in enumerate(sources[:3], 1):  # Show max 3 sources
        title = source.get('title', 'Unknown')[:50]
        url = source.get('url', '')
        relevance = int(source.get('relevance', 0) * 100)
        
        if url:
            formatted.append(f"{i}. [{title}]({url}) ({relevance}% relevant)")
        else:
            formatted.append(f"{i}. {title} ({relevance}% relevant)")
    
    return "\n".join(formatted)

def validate_question(question: str) -> tuple:
    """Validate user question"""
    if not question or not question.strip():
        return False, "Please enter a question"
    
    if len(question.strip()) < 3:
        return False, "Question too short (minimum 3 characters)"
    
    if len(question) > 500:
        return False, "Question too long (maximum 500 characters)"
    
    # Check for potentially problematic content
    if re.match(r'^[^a-zA-Z0-9\s]*$', question):
        return False, "Please enter a meaningful question"
    
    return True, "Valid question"

def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text with ellipsis"""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."

def format_response_time(seconds: float) -> str:
    """Format response time for display"""
    if seconds < 1:
        return f"{int(seconds * 1000)}ms"
    else:
        return f"{seconds:.1f}s"

def get_category_display(category: str) -> Dict[str, str]:
    """Get display info for content categories"""
    categories = {
        'general': {'icon': '📄', 'label': 'General', 'color': '#6c757d'},
        'faq': {'icon': '❓', 'label': 'FAQ', 'color': '#17a2b8'},
        'legal': {'icon': '⚖️', 'label': 'Legal', 'color': '#dc3545'},
        'product': {'icon': '🏦', 'label': 'Product', 'color': '#007bff'},
        'about': {'icon': 'ℹ️', 'label': 'About', 'color': '#6f42c1'},
        'pricing': {'icon': '💰', 'label': 'Pricing', 'color': '#fd7e14'},
        'security': {'icon': '🔒', 'label': 'Security', 'color': '#e83e8c'}
    }
    return categories.get(category, categories['general'])

def create_conversation_export(conversation_history: List[Dict]) -> str:
    """Create exportable conversation data"""
    if not conversation_history:
        return ""
    
    lines = ["Timestamp,Question,Response,Sources"]
    
    for conv in conversation_history:
        timestamp = conv.get('timestamp', '')
        question = conv.get('question', '').replace(',', ';').replace('\n', ' ')
        response = conv.get('response', '').replace(',', ';').replace('\n', ' ')
        sources = len(conv.get('sources', []))
        
        lines.append(f'"{timestamp}","{question}","{response}",{sources}')
    
    return "\n".join(lines)

def calculate_relevance_color(score: float) -> str:
    """Get color based on relevance score"""
    if score >= 0.8:
        return "#28a745"  # Green
    elif score >= 0.6:
        return "#ffc107"  # Yellow
    else:
        return "#dc3545"  # Red

def format_number(num: int) -> str:
    """Format number with commas"""
    return f"{num:,}"

def get_response_quality(relevance_score: float, response_time: float) -> str:
    """Determine response quality based on metrics"""
    if relevance_score >= 0.8 and response_time < 3:
        return "Excellent"
    elif relevance_score >= 0.6 and response_time < 5:
        return "Good"
    elif relevance_score >= 0.4:
        return "Fair"
    else:
        return "Poor"

# Jupiter-specific helpers
def get_jupiter_quick_questions() -> List[str]:
    """Get list of common Jupiter questions"""
    return [
        "What are Jupiter's main features?",
        "How do I open a Jupiter account?",
        "What fees does Jupiter charge?",
        "How do I contact Jupiter customer support?",
        "What security features does Jupiter have?",
        "How does Jupiter's UPI payment work?",
        "What investment options are available?",
        "How do I download the Jupiter app?"
    ]

def suggest_follow_up_questions(current_question: str, category: str) -> List[str]:
    """Suggest relevant follow-up questions"""
    suggestions = {
        'account': [
            "What documents are needed to open an account?",
            "How long does account activation take?",
            "What are the account benefits?"
        ],
        'fees': [
            "Are there any hidden charges?",
            "How do Jupiter's fees compare to other banks?",
            "What transactions are free?"
        ],
        'security': [
            "How is my money protected?",
            "What should I do if I lose my phone?",
            "How do I report suspicious activity?"
        ]
    }
    
    current_lower = current_question.lower()
    
    # Determine category based on question content
    if any(word in current_lower for word in ['account', 'open', 'registration']):
        return suggestions.get('account', [])
    elif any(word in current_lower for word in ['fee', 'charge', 'cost', 'price']):
        return suggestions.get('fees', [])
    elif any(word in current_lower for word in ['security', 'safe', 'protect']):
        return suggestions.get('security', [])
    
    # Default suggestions
    return [
        "What are Jupiter's key benefits?",
        "How do I get started with Jupiter?",
        "What makes Jupiter different?"
    ]
