# 📁 src/frontend/Home.py
import streamlit as st
import sys
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

            # Model selection (if you have multiple models)
            model = st.selectbox(
                "Select Model",
                ["GPT-3.5", "GPT-4", "Claude", "Llama"],
                key="selected_model",
            )

            # Temperature slider
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                key="temperature",
            )

            st.divider()

            # Chat controls
            st.subheader("💬 Chat Controls")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Clear History", use_container_width=True):
                    self.chat_interface.clear_chat_history()
                    st.rerun()
            with col2:
                if st.button("New Chat", use_container_width=True):
                    st.session_state.messages = []
                    st.rerun()

    def render_main_content(self):
        """Render the main chat interface"""
        # Header with welcome message
        st.title("🤖 RAG Chat Assistant")

        # Show welcome message for new chat
        if not st.session_state.get("messages"):
            st.markdown(
                """
            👋 **Welcome to RAG Chat Assistant!**
            
            I can help you with:
            - Analyzing documents you upload
            - Answering questions about your data
            - Generating insights and summaries
            
            To get started:
            1. Upload your documents using the sidebar 📄
            2. Ask any questions about your documents 💭
            3. Get AI-powered responses based on your data 🤖
            """
            )

        # Document check
        if not st.session_state.get("uploaded_files"):
            st.info("👈 Please upload your documents in the sidebar to get started!")

        # Chat interface
        st.divider()
        self.chat_interface.display_chat_history()
        self.chat_interface.handle_user_input(self.process_message)

    def process_message(self, prompt: str) -> str:
        """Process the user message and return a response"""
        try:
            # Here you would typically:
            # 1. Get the selected model and temperature from session state
            model = st.session_state.get("selected_model", "GPT-3.5")
            temperature = st.session_state.get("temperature", 0.7)

            # 2. Process the message with your RAG pipeline
            # This is a mock response - replace with your actual RAG logic
            response = f"Model: {model} (temp={temperature})\nThis is a mock response to: {prompt}"

            return response

        except Exception as e:
            st.error(f"Error processing message: {str(e)}")
            return "I apologize, but I encountered an error processing your message. Please try again."


def main():
    try:
        if "authenticated" not in st.session_state:
            st.session_state.authenticated = True  # Default for now

        if st.session_state.authenticated:
            home = HomePage()
            home.render_sidebar()
            home.render_main_content()
        else:
            st.error("Please log in to access the chat interface")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
