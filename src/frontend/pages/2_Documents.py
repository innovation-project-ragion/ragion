import streamlit as st
from components.shared.document_upload import DocumentUpload


class DocumentsPage:
    def __init__(self):
        self.doc_upload = DocumentUpload()

    def render(self):
        # Remove set_page_config as it's now in Home.py
        st.title("📚 Document Management")

        # Document upload section
        self.doc_upload.render_upload_section()
        self.doc_upload.render_file_list()

        # Option to delete documents
        if st.button("Delete All Documents"):
            st.session_state.uploaded_files = []
            st.success("All documents deleted.")
            st.rerun()


# Initialize the page
page = DocumentsPage()
page.render()
