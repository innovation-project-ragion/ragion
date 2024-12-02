import streamlit as st
from typing import List, Optional
import os


class DocumentUpload:
    ALLOWED_EXTENSIONS = [".pdf", ".txt", ".doc", ".docx"]

    def __init__(self):
        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = []

    def is_valid_file(self, file) -> bool:
        """Check if the file has an allowed extension"""
        file_ext = os.path.splitext(file.name)[1].lower()
        return file_ext in self.ALLOWED_EXTENSIONS

    def render_upload_section(self) -> Optional[List]:
        """Render the document upload section"""
        st.subheader("📁 Document Upload")

        # File uploader
        uploaded_files = st.file_uploader(
            "Upload your documents",
            accept_multiple_files=True,
            type=["pdf", "txt", "doc", "docx"],
            help="Supported formats: PDF, TXT, DOC, DOCX",
        )

        if uploaded_files:
            for file in uploaded_files:
                if self.is_valid_file(file):
                    if file not in st.session_state.uploaded_files:
                        st.session_state.uploaded_files.append(file)

        return uploaded_files

    def render_file_list(self):
        """Display the list of uploaded files"""
        if st.session_state.uploaded_files:
            st.subheader("📚 Uploaded Documents")

            for idx, file in enumerate(st.session_state.uploaded_files):
                col1, col2, col3 = st.columns([3, 1, 1])

                with col1:
                    st.text(file.name)

                with col2:
                    if st.button("View", key=f"view_{idx}"):
                        self.display_file_content(file)

                with col3:
                    if st.button("Remove", key=f"remove_{idx}"):
                        st.session_state.uploaded_files.remove(file)
                        st.rerun()

    def display_file_content(self, file):
        """Display the content of the selected file"""
        try:
            # For demonstration, we'll just show text files
            # In a real app, you'd want to handle different file types appropriately
            if file.name.endswith(".txt"):
                content = file.read().decode("utf-8")
                with st.expander(f"📄 {file.name}", expanded=True):
                    st.text(content)
            else:
                st.info(f"Preview not available for {file.name}")
        except Exception as e:
            st.error(f"Error displaying file: {str(e)}")
