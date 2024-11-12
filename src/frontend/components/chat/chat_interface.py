import streamlit as st
from typing import List, Dict


class ChatInterface:
    def __init__(self):
        # Initialize chat history in session state if not exists
        if "messages" not in st.session_state:
            st.session_state.messages = []

        if "chat_input_key" not in st.session_state:
            st.session_state.chat_input_key = 0

    def display_message(self, message: Dict[str, str]):
        """Display a single message in the chat interface"""
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    def display_chat_history(self):
        """Display all messages in the chat history"""
        for message in st.session_state.messages:
            self.display_message(message)

    def handle_user_input(self, callback_fn=None):
        """Handle user input and process it through the callback function"""
        # Create a unique key for the chat input
        if prompt := st.chat_input(
            "What would you like to know?",
            key=f"chat_input_{st.session_state.chat_input_key}",
        ):
            # Add user message to chat history
            user_message = {"role": "user", "content": prompt}
            st.session_state.messages.append(user_message)
            self.display_message(user_message)

            # Get AI response through callback if provided
            if callback_fn:
                response = callback_fn(prompt)
            else:
                response = f"Echo: {prompt}"  # Default echo response

            # Add assistant response to chat history
            assistant_message = {"role": "assistant", "content": response}
            st.session_state.messages.append(assistant_message)
            self.display_message(assistant_message)

            # Increment the chat input key to force a refresh
            st.session_state.chat_input_key += 1

    def clear_chat_history(self):
        """Clear the chat history"""
        st.session_state.messages = []
        st.session_state.chat_input_key = 0
