# 📁 src/frontend/components/chat/chat_interface.py
import streamlit as st
from typing import List, Dict, Callable
import time

class ChatInterface:
    def __init__(self):
        # Initialize chat history in session state if not exists
        if "messages" not in st.session_state:
            st.session_state.messages = []

    def display_message(self, message: Dict[str, str]):
        """Display a single message in the chat interface"""
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    def display_chat_history(self):
        """Display all messages in the chat history"""
        for message in st.session_state.messages:
            self.display_message(message)

    def handle_user_input(self, callback_fn: Callable):
        """Handle user input and process it through the callback function"""
        # Get user input
        if prompt := st.chat_input("What would you like to know?"):
            # Add user message to chat history
            user_message = {"role": "user", "content": prompt}
            st.session_state.messages.append(user_message)

            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response with streaming
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                # Process the message and get streaming response
                for response_chunk in self._stream_response(prompt, callback_fn):
                    full_response += response_chunk
                    # Add a blinking cursor to simulate typing
                    message_placeholder.markdown(full_response + "▌")
                
                # Remove the cursor and display the final response
                message_placeholder.markdown(full_response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response
                })

    def _stream_response(self, prompt: str, callback_fn: Callable):
        """Stream the response from the callback function"""
        model = st.session_state.get("selected_model", "GPT-3.5")
        temperature = st.session_state.get("temperature", 0.7)
        
        # Generate the full response
        full_response = f"Model: {model} (temp={temperature})\nThis is a mock response to: {prompt}"
        
        # Stream the response word by word
        words = full_response.split()
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            time.sleep(0.05)  # Reduced delay for better UX

    def clear_chat_history(self):
        """Clear the chat history"""
        st.session_state.messages = []