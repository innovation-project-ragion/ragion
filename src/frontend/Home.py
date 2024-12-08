# 📁 src/frontend/Home.py
import streamlit as st
import sys
import time
from pathlib import Path
import uuid

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
        
        # Initialize query tracking
        if "current_query_id" not in st.session_state:
            st.session_state.current_query_id = None
        if "is_processing" not in st.session_state:
            st.session_state.is_processing = False
        if "showing_welcome" not in st.session_state:
            st.session_state.showing_welcome = True
        
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
            # st.selectbox(
            #     "Select Model",
            #     ["GPT-3.5", "GPT-4", "Claude", "Llama"],
            #     key="selected_model"
            # )
            
            # Temperature slider
            # st.slider(
            #     "Temperature",
            #     min_value=0.0,
            #     max_value=1.0,
            #     value=0.7,
            #     step=0.1,
            #     key="temperature"
            # )
            
            st.divider()
            
            st.subheader("💬 Chat Controls")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Clear History", use_container_width=True):
                    if hasattr(self.chat_interface, 'clear_chat_history'):
                        self.chat_interface.clear_chat_history()
                    st.session_state.showing_welcome = True
                    st.session_state.current_query_id = None
                    st.session_state.is_processing = False
                    st.rerun()
            with col2:
                if st.button("New Chat", use_container_width=True):
                    if hasattr(st.session_state, 'messages'):
                        st.session_state.messages = []
                    st.session_state.showing_welcome = True
                    st.session_state.current_query_id = None
                    st.session_state.is_processing = False
                    st.rerun()

    def render_main_content(self):
        """Render the main chat interface"""
        st.title("🤖 RAG Chat Assistant")
        
        # Show welcome message only when showing_welcome is True
        if st.session_state.showing_welcome:
            welcome_container = st.container()
            with welcome_container:
                st.markdown("""
                👋 **Welcome to RAG Chat Assistant!**
                """)
                
                if not st.session_state.get("uploaded_files"):
                    st.info("👈 Please upload your documents in the sidebar to get started!")
        
        # Divider for separating the welcome message from chat area
        st.divider()
        
        # === Hardcoded Test: Add This Block ===
        st.markdown("### Testing Hardcoded Answer and Sources")
        
        # Hardcoded answer
        hardcoded_answer = "Elisa on 66-vuotias. Luottamus: 95%"
        
        # Hardcoded sources
        hardcoded_sources = [
            {
                "text": "Kansalaisuus: Suomi",
                "document_id": "M2042",
                "score": "95%"
            },
            {
                "text": "Maakunta: ?",
                "document_id": "KL-49",
                "score": "85%"
            }
        ]
        
        # Display the hardcoded answer
        st.markdown(f"**Answer:** {hardcoded_answer}")
        
        # Display the hardcoded sources
        if hardcoded_sources:
            with st.expander("📚 Sources", expanded=False):
                for idx, source in enumerate(hardcoded_sources, 1):
                    st.markdown(f"""
                    **Source {idx}** (Document {source['document_id']}) - Relevance: {source['score']}
                    ```
                    {source['text']}
                    ```
                    """)
        # === End of Hardcoded Test ===
        if hasattr(self.chat_interface, 'display_chat_history'):
            self.chat_interface.display_chat_history()

        if not st.session_state.is_processing:
            if prompt := st.chat_input("What would you like to know?"):
                st.session_state.showing_welcome = False
                st.session_state.is_processing = True
                self.chat_interface.handle_user_input(prompt)
        else:

            st.text_input(
                "What would you like to know?", 
                disabled=True, 
                value="Processing your query... please wait"
            )
        if st.session_state.get("show_debug", False):
            with st.expander("🔍 Debug Information", expanded=False):
                st.write("Session State:")
                st.json({
                    "current_query_id": st.session_state.get("current_query_id"),
                    "is_processing": st.session_state.is_processing,
                    "showing_welcome": st.session_state.showing_welcome
                })


def main():
    home = HomePage()
    home.render_sidebar()
    home.render_main_content()

if __name__ == "__main__":
    main()