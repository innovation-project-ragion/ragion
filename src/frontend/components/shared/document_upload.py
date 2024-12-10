## src/frontend/components/shared/document_upload.py
import streamlit as st
from typing import List, Optional
import os
import asyncio
from src.frontend.utils.document_api import DocumentAPIClient
import logging
import time
logger = logging.getLogger(__name__)

class DocumentUpload:
    ALLOWED_EXTENSIONS = [".pdf", ".txt", ".doc", ".docx"]

    def __init__(self):
        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = []
        if "upload_status" not in st.session_state:
            st.session_state.upload_status = {}
        self.api_client = DocumentAPIClient()

    def is_valid_file(self, file) -> bool:
        """Check if the file has an allowed extension"""
        file_ext = os.path.splitext(file.name)[1].lower()
        return file_ext in self.ALLOWED_EXTENSIONS

    async def upload_file(self, file) -> None:
        """Upload a file to the backend service"""
        try:
            result = await self.api_client.upload_document(file)
            job_id = result["job_id"]
            st.session_state.upload_status[file.name] = {
                "status": "processing",
                "job_id": job_id,
                "start_time": time.time()
            }
            max_wait_time = 3600  # 1 hour total maximum wait
            check_interval = 30   # Check every 30 seconds
            start_time = time.time()
            # Monitor the job status
            while True:
                if time.time() - start_time > max_wait_time:
                    st.session_state.upload_status[file.name] = {
                        "status": "failed",
                        "error": "Processing timeout exceeded (1 hour)"
                    }
                    break
                    
                try:
                    status = await self.api_client.get_document_status(job_id)
                    
                    # Update progress information
                    elapsed_time = int(time.time() - start_time)
                    st.session_state.upload_status[file.name].update({
                        "elapsed_time": elapsed_time,
                        "last_check": time.time()
                    })
                    
                    if status["status"] == "completed":
                        st.session_state.upload_status[file.name].update({
                            "status": "completed",
                            "processing_time": elapsed_time
                        })
                        break
                    elif status["status"] == "failed":
                        st.session_state.upload_status[file.name].update({
                            "status": "failed",
                            "error": status.get("error", "Unknown error"),
                            "processing_time": elapsed_time
                        })
                        break
                    
                    # Show processing status
                    st.session_state.upload_status[file.name].update({
                        "status": "processing",
                        "message": f"Processing for {elapsed_time//60}m {elapsed_time%60}s"
                    })
                    
                except Exception as e:
                    logger.warning(f"Status check failed: {str(e)}")
                    # Continue monitoring despite temporary errors
                    
                await asyncio.sleep(check_interval)

        except Exception as e:
            logger.error(f"Error uploading file {file.name}: {str(e)}")
            st.session_state.upload_status[file.name] = {
                "status": "failed",
                "error": str(e)
            }

    def render_upload_section(self) -> Optional[List]:
        """Render the document upload section"""
        st.subheader("📁 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Upload your documents",
            accept_multiple_files=True,
            type=["pdf", "txt", "doc", "docx"],
            help="Supported formats: PDF, TXT, DOC, DOCX",
        )

        if uploaded_files:
            for file in uploaded_files:
                if self.is_valid_file(file):
                    if file.name not in [f.name for f in st.session_state.uploaded_files]:
                        st.session_state.uploaded_files.append(file)
                        # Start upload process
                        asyncio.run(self.upload_file(file))

        return uploaded_files

    def render_file_list(self):
        """Display the list of uploaded files"""
        if st.session_state.uploaded_files:
            st.subheader("📚 Uploaded Documents")
            
            for idx, file in enumerate(st.session_state.uploaded_files):
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                
                with col1:
                    st.text(file.name)
                
                with col2:
                    status = st.session_state.upload_status.get(file.name, {}).get("status", "unknown")
                    if status == "processing":
                        st.info("Processing...")
                    elif status == "completed":
                        st.success("Completed")
                    elif status == "failed":
                        error = st.session_state.upload_status[file.name].get("error", "Unknown error")
                        st.error(f"Failed: {error}")
                
                with col3:
                    if st.button("View", key=f"view_{idx}"):
                        self.display_file_content(file)
                
                with col4:
                    if st.button("Remove", key=f"remove_{idx}"):
                        # Remove from both lists
                        st.session_state.uploaded_files.remove(file)
                        if file.name in st.session_state.upload_status:
                            del st.session_state.upload_status[file.name]
                        st.rerun()

    def display_file_content(self, file):
        """Display the content of the selected file"""
        try:
            if file.name.endswith(".txt"):
                content = file.read().decode("utf-8")
                with st.expander(f"📄 {file.name}", expanded=True):
                    st.text(content)
            else:
                st.info(f"Preview not available for {file.name}")
        except Exception as e:
            st.error(f"Error displaying file: {str(e)}")