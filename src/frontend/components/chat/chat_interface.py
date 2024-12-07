# src/frontend/components/chat/chat_interface.py
import streamlit as st
from typing import Dict
import asyncio
import json
import time
import logging
from src.frontend.utils.api_client import RAGApiClient

logger = logging.getLogger(__name__)

class ChatInterface:
    def __init__(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "api_client" not in st.session_state:
            st.session_state.api_client = RAGApiClient()
        self.timeout = 300  # 5 minutes timeout
        self.max_retries = 3

    def _show_thinking_animation(self, placeholder):
        """Show thinking animation with rotating dots."""
        thinking_states = ["🤔 Thinking", "🤔 Thinking.", "🤔 Thinking..", "🤔 Thinking..."]
        for state in thinking_states:
            placeholder.markdown(f"{state}")
            time.sleep(0.3)

    async def process_query(self, query: str) -> None:
        """Process a query through the RAG pipeline with enhanced error handling."""
        try:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                # Show initial loading state
                with message_placeholder:
                    col1, col2 = st.columns([1, 5])
                    with col1:
                        st.markdown("🔄")
                    with col2:
                        st.markdown("Connecting to backend...")
                        progress_bar = st.progress(0)

                # Submit query with retries
                retry_count = 0
                response = None
                
                while retry_count < self.max_retries:
                    try:
                        response = await st.session_state.api_client.submit_query(query)
                        progress_bar.progress(0.2)
                        break
                    except Exception as e:
                        retry_count += 1
                        if retry_count == self.max_retries:
                            raise Exception(f"Failed to connect after {self.max_retries} attempts: {str(e)}")
                        logger.warning(f"Retry {retry_count}/{self.max_retries}: {str(e)}")
                        await asyncio.sleep(2 ** retry_count)  # Exponential backoff

                if response and response.get("status") == "processing" and response.get("job_id"):
                    # Update loading state to show processing
                    with message_placeholder:
                        col1, col2 = st.columns([1, 5])
                        with col1:
                            st.markdown("⚡")
                        with col2:
                            status_text = st.empty()
                            status_text.markdown("Processing your query...")
                            progress_bar.progress(0.3)

                    start_time = time.time()
                    found_answer = False
                    
                    async for chunk in self.api_client.stream_response(response["job_id"]):
                        try:
                            if time.time() - start_time > self.timeout:
                                raise TimeoutError("Query processing timeout")

                            # Update progress while waiting
                            if not found_answer:
                                progress_value = min(0.3 + ((time.time() - start_time) / self.timeout) * 0.6, 0.9)
                                progress_bar.progress(progress_value)

                            # Parse the full response
                            response_data = json.loads(chunk)
                            found_answer = True
                            
                            # Complete the progress bar
                            progress_bar.progress(1.0)
                            
                            # Extract just the answer/response
                            answer = response_data.get('response', '')
                            
                            # Clear the loading state and show the answer
                            message_placeholder.empty()
                            with message_placeholder:
                                st.markdown(answer)
                                
                                # Show sources in expander
                                with st.expander("📚 View sources", expanded=False):
                                    sources = response_data.get('sources', [])
                                    if sources:
                                        for idx, source in enumerate(sources[:3], 1):
                                            st.markdown(f"""
                                            **Source {idx}** (Document {source['document_id']})
                                            ```
                                            {source['text'][:200]}...
                                            ```
                                            """)
                                    else:
                                        st.info("ℹ️ No source documents available for this response.")
                            
                            # Store in chat history
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": answer,
                                "metadata": {
                                    "sources": response_data.get('sources', []),
                                    "person_contexts": response_data.get('person_contexts', [])
                                }
                            })
                            
                        except json.JSONDecodeError as e:
                            logger.error(f"Error parsing response: {str(e)}")
                            message_placeholder.markdown(f"❌ Error: Invalid response format")
                        except Exception as e:
                            logger.error(f"Error processing chunk: {str(e)}")
                            message_placeholder.markdown(f"❌ Error: {str(e)}")
                    
        except TimeoutError:
            st.error("❌ Query processing timed out. Please try again.")
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            st.error(f"❌ Error: {str(e)}")

    def display_chat_history(self):
        """Display the chat history with clean formatting."""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                # Display just the message content
                st.markdown(message["content"])
                
                # If it's an assistant message and has sources, show them in an expander
                if message["role"] == "assistant" and "metadata" in message:
                    with st.expander("📚 View supporting documents", expanded=False):
                        sources = message["metadata"].get("sources", [])
                        if sources:
                            for idx, source in enumerate(sources[:3], 1):
                                st.markdown(f"""
                                **Source {idx}** (Document {source['document_id']})
                                ```
                                {source['text'][:200]}...
                                ```
                                """)
                        else:
                            st.info("ℹ️ No source documents available for this response.")

    def handle_user_input(self):
        """Handle user input and process it through the RAG pipeline."""
        try:
            if prompt := st.chat_input("What would you like to know?"):
                # Add user message to chat history
                st.session_state.messages.append({
                    "role": "user", 
                    "content": prompt
                })
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Process the query asynchronously
                asyncio.run(self.process_query(prompt))
        except Exception as e:
            logger.error(f"Error handling user input: {str(e)}")
            st.error(f"❌ Error: Something went wrong. Please try again.")

    def clear_chat_history(self):
        """Clear the chat history."""
        st.session_state.messages = []