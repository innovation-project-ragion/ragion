import streamlit as st

def init_session_state():
    """Initialize all session state variables"""
    # Chat-related states
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_input_key" not in st.session_state:
        st.session_state.chat_input_key = 0
    if "show_welcome" not in st.session_state:
        st.session_state.show_welcome = True  # New flag for welcome message

    # Document-related states
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "file_uploader_key" not in st.session_state:
        st.session_state.file_uploader_key = 0

    # Settings-related states
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "GPT-3.5"
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7

    # Authentication states
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = True
