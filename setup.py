#!/usr/bin/env python3
"""
Setup script for Jupiter RAG system with LLM
"""

import os
import sys
import subprocess
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.scraper import JupiterWebScraper
from utils.rag_engine import JupiterRAGEngine

def install_requirements():
    """Install requirements"""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ['data', 'chroma_db', '.streamlit']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("✅ Directories created")

def setup_streamlit_config():
    """Create Streamlit configuration"""
    config_content = """[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"

[server]
maxUploadSize = 200
enableCORS = false
"""
    
    os.makedirs('.streamlit', exist_ok=True)
    with open('.streamlit/config.toml', 'w') as f:
        f.write(config_content)
    
    print("✅ Streamlit config created")

def main():
    """Main setup function"""
    print("🚀 Jupiter RAG with LLM Setup")
    print("=" * 40)
    
    # Create directories
    create_directories()
    
    # Setup Streamlit config
    setup_streamlit_config()
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed")
        return False
    
    print("\n🎉 Setup Complete!")
    print("=" * 40)
    print("✅ Jupiter RAG system is ready")
    print("\n🔑 Next Steps:")
    print("1. Set your OpenAI API key:")
    print("   - Environment: export OPENAI_API_KEY='your-key'")
    print("   - Or add to Streamlit secrets")
    print("\n🚀 To start the application:")
    print("   streamlit run app.py")
    print("\n📖 App will be at: http://localhost:8501")
    
    return True

if __name__ == "__main__":
    main()
