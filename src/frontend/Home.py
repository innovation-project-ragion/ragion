# 📁 src/frontend/Home.py
import streamlit as st
import sys
import time
from pathlib import Path

# Add the project root to Python path
root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

from src.frontend.components.chat.chat_interface import ChatInterface
from src.frontend.components.shared.document_upload import DocumentUpload
from src.frontend.utils.session_state import init_session_state

class HomePage:
    def __init__(self):
        # Initialize components
        self.chat_interface = ChatInterface()
        self.doc_upload = DocumentUpload()
        
        # Initialize session state
        init_session_state()
        
        # Set up the page configuration
        st.set_page_config(
            page_title="RAG Chat Assistant",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded",
        )

    def render_sidebar(self):
        """Render the sidebar with document upload and settings"""
        with st.sidebar:
            st.title("📚 Documents")
            
            # Document upload section
            self.doc_upload.render_upload_section()
            self.doc_upload.render_file_list()
            
            st.divider()
            
            # Chat settings
            st.subheader("⚙️ Chat Settings")
            
            # Model selection
            st.selectbox(
                "Select Model",
                ["GPT-3.5", "GPT-4", "Claude", "Llama"],
                key="selected_model"
            )
            
            # Temperature slider
            st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                key="temperature"
            )
            
            st.divider()
            
            # Chat controls
            st.subheader("💬 Chat Controls")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Clear History", use_container_width=True):
                    self.chat_interface.clear_chat_history()
                    st.session_state.showing_welcome = True
                    st.rerun()
            with col2:
                if st.button("New Chat", use_container_width=True):
                    st.session_state.messages = []
                    st.session_state.showing_welcome = True
                    st.rerun()

    def render_main_content(self):
        """Render the main chat interface"""
        # Initialize showing_welcome in session state if not exists
        if "showing_welcome" not in st.session_state:
            st.session_state.showing_welcome = True
            
        # Header
        st.title("🤖 RAG Chat Assistant")
        
        # Show welcome message only when showing_welcome is True
        if st.session_state.showing_welcome:
            welcome_container = st.container()
            with welcome_container:
                st.markdown("""
                👋 **Welcome to RAG Chat Assistant!**
                
                I can help you with:
                - Analyzing documents you upload
                - Answering questions about your data
                - Generating insights and summaries
                
                To get started:
                1. Upload your documents using the sidebar 📄
                2. Ask any questions about your documents 💭
                3. Get AI-powered responses based on your data 🤖
                """)
                
                # Document check
                if not st.session_state.get("uploaded_files"):
                    st.info("👈 Please upload your documents in the sidebar to get started!")
        
        # Chat interface
        st.divider()
        self.chat_interface.display_chat_history()
        
        # If a message is sent, hide the welcome message
        if prompt := st.chat_input("What would you like to know?"):
            st.session_state.showing_welcome = False
            self.process_and_display_message(prompt)
            st.rerun()

    def process_and_display_message(self, prompt: str):
        """Process and display the message with streaming response"""
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get response
        model = st.session_state.get("selected_model", "GPT-3.5")
        temperature = st.session_state.get("temperature", 0.7)
        response = f"Model: {model} (temp={temperature})\nThis is a mock response to: {prompt}"
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def main():
    home = HomePage()
    chat_interface = ChatInterface()
    
    home.render_sidebar()
    
    # Main chat area
    st.title("🤖 RAG Chat Assistant")
    chat_interface.display_chat_history()
    chat_interface.handle_user_input()

if __name__ == "__main__":
    main()