## src/frontend/components/chat/chat_interface.py
import streamlit as st
from typing import Dict, Optional, List
import asyncio
import json
import time
import logging
import sys
import re
from datetime import datetime
from src.frontend.utils.api_client import RAGApiClient

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("frontend.chat")
logger.setLevel(logging.DEBUG)

class ChatInterface:
    def __init__(self):
        self.initialize_session_state()
        self.timeout = 300
        self.max_retries = 3
        self.max_source_length = 200

    def initialize_session_state(self):
        """Initialize all required session state variables."""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "debug_enabled" not in st.session_state:
            st.session_state.debug_enabled = False
        if "is_processing" not in st.session_state:
            st.session_state.is_processing = False

    def _clean_answer(self, answer: str) -> str:
        """Clean up duplicated content and format the answer."""
        if not answer:
            return ""
            
        # Split on first occurrence of "Luottamus:"
        main_answer = answer.split("Luottamus:")[0].strip()
        
        # If there's a confidence part, add it back once
        if "Luottamus:" in answer:
            confidence_part = "Luottamus:" + answer.split("Luottamus:")[1].split(".")[0] + "."
            main_answer = f"{main_answer} {confidence_part}"
            
        return main_answer

    async def _process_query_async(self, query: str) -> None:
        async with RAGApiClient() as client:
            try:
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    
                    with message_placeholder:
                        st.markdown("```\nProcessing query...\n```")
                    
                    response = await client.submit_query(query)
                    
                    if response and response.get("status") == "processing":
                        job_id = response.get("job_id")
                        if not job_id:
                            raise Exception("No job ID received from backend")
                        
                        while True:
                            status_response = await client.check_query_status(job_id)
                            
                            if status_response.get("status") == "completed":
                                logger.info("-" * 80)
                                logger.info("RESPONSE RECEIVED")
                                
                                # Clear placeholder
                                message_placeholder.empty()
                                
                                # Process and render response
                                formatted_response = self._format_response(status_response)
                                logger.debug(f"Original answer before cleaning: {formatted_response['answer']}")
                                
                                cleaned_answer = self._clean_answer(formatted_response["answer"])
                                logger.debug(f"Cleaned answer: {cleaned_answer}")
                                
                                # Render the cleaned answer
                                message_placeholder.markdown(cleaned_answer)
                                
                                # Render sources
                                if formatted_response["sources"]:
                                    with st.expander("📚 Sources", expanded=False):
                                        for idx, source in enumerate(formatted_response["sources"], 1):
                                            st.markdown(f"""
                                            **Source {idx}** (Document {source['document_id']}) - Relevance: {source['score']}
                                            ```
                                            {source['text'][:self.max_source_length]}...
                                            ```
                                            """)
                                
                                # Update session state
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": cleaned_answer,
                                    "sources": formatted_response["sources"],
                                    "timestamp": datetime.now().strftime("%H:%M:%S")
                                })
                                logger.debug(f"Updated session state: {st.session_state.messages}")
                                break
                            
                            elif status_response.get("status") == "failed":
                                error_msg = status_response.get("error", "Unknown error")
                                raise Exception(error_msg)
                            
                            await asyncio.sleep(1)
                    else:
                        raise Exception(f"Unexpected response status: {response.get('status')}")
                        
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}", exc_info=True)
                message_placeholder.error(f"❌ Error: {str(e)}")


    def _format_response(self, response_data: dict) -> dict:
        """Format the response data into a structured format."""
        try:
            if response_data.get("status") == "completed" and "result" in response_data:
                answer = response_data["result"].get("answer", "")
                sources = response_data["result"].get("sources", [])
            else:
                answer = response_data.get("response", "")
                sources = response_data.get("sources", [])

            formatted_sources = []
            for source in sources:
                formatted_sources.append({
                    "text": source.get("text", ""),
                    "document_id": source.get("document_id", "Unknown"),
                    "score": f"{float(source.get('score', 0)):.2%}"
                })

            return {
                "answer": answer,
                "sources": formatted_sources
            }
        except Exception as e:
            logger.error(f"Error formatting response: {str(e)}", exc_info=True)
            return {"answer": "Error formatting response", "sources": []}

    def handle_user_input(self, prompt: str):
        """Handle user input and process it through the RAG pipeline."""
        try:
            # Store user message
            st.session_state.messages.append({
                "role": "user",
                "content": prompt,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Set processing state
            st.session_state.is_processing = True
            
            # Create new event loop and run async process
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._process_query_async(prompt))
            finally:
                loop.close()
                st.session_state.is_processing = False
            
        except Exception as e:
            logger.error(f"Error handling user input: {str(e)}", exc_info=True)
            st.error(f"❌ Error: {str(e)}")
            st.session_state.is_processing = False

    def display_chat_history(self):
        """Display the chat history."""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("📚 Sources", expanded=False):
                        for idx, source in enumerate(message["sources"], 1):
                            st.markdown(f"""
                            **Source {idx}** (Document {source['document_id']}) - Relevance: {source['score']}
                            ```
                            {source['text'][:self.max_source_length]}...
                            ```
                            """)

    def clear_chat_history(self):
        """Clear the chat history."""
        st.session_state.messages = []
