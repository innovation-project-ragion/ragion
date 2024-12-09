## src/frontend/pages/2_Documents.py
import streamlit as st
from components.shared.document_upload import DocumentUpload
from src.frontend.utils.document_api import DocumentAPIClient
import asyncio
from datetime import datetime

class DocumentsPage:
    def __init__(self):
        self.doc_upload = DocumentUpload()
        self.api_client = DocumentAPIClient()
        if "existing_documents" not in st.session_state:
            st.session_state.existing_documents = []
        
    async def fetch_existing_documents(self):
        """Fetch existing documents from the backend"""
        try:
            result = await self.api_client.get_document_list()
            st.session_state.existing_documents = result.get("documents", [])
        except Exception as e:
            st.error(f"Error fetching existing documents: {str(e)}")

    async def delete_document(self, document_id: str):
        """Delete a document from the backend"""
        try:
            await self.api_client.delete_document(document_id)
            # Refresh the document list
            await self.fetch_existing_documents()
            st.success(f"Document deleted successfully")
        except Exception as e:
            st.error(f"Error deleting document: {str(e)}")

    def render(self):
        st.title("📚 Document Management")

        # Create tabs for different sections
        upload_tab, existing_tab = st.tabs(["Upload Documents", "Existing Documents"])

        with upload_tab:
            # Document upload section
            self.doc_upload.render_upload_section()
            self.doc_upload.render_file_list()
            
            if st.button("Delete All Uploaded Files"):
                st.session_state.uploaded_files = []
                st.session_state.upload_status = {}
                st.success("All uploaded files cleared.")
                st.rerun()

        with existing_tab:
            st.subheader("📚 Existing Documents")
            
            # Button to refresh document list
            if st.button("🔄 Refresh Document List"):
                asyncio.run(self.fetch_existing_documents())

            # Display existing documents
            if st.session_state.existing_documents:
                for doc in st.session_state.existing_documents:
                    with st.expander(f"📄 {doc['document_id']}", expanded=False):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.write("**Document Details:**")
                            st.write(f"👤 Person: {doc['person_name']}")
                            st.write(f"📊 Chunks: {doc['chunk_count']}")
                            if doc.get('created_at'):
                                created_at = datetime.fromisoformat(doc['created_at'].replace('Z', '+00:00'))
                                st.write(f"📅 Created: {created_at.strftime('%Y-%m-%d %H:%M')}")
                            if doc.get('person_age'):
                                st.write(f"🎂 Person Age: {doc['person_age']}")
                        
                        with col2:
                            if st.button("🗑️ Delete", key=f"delete_{doc['document_id']}"):
                                asyncio.run(self.delete_document(doc['document_id']))
                                st.rerun()
            else:
                st.info("No existing documents found. Try refreshing the list or upload new documents.")

            # Add some spacing
            st.markdown("---")
            
            # Statistics section
            if st.session_state.existing_documents:
                st.subheader("📊 Document Statistics")
                total_docs = len(st.session_state.existing_documents)
                total_chunks = sum(doc['chunk_count'] for doc in st.session_state.existing_documents)
                unique_persons = len(set(doc['person_name'] for doc in st.session_state.existing_documents))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Documents", total_docs)
                with col2:
                    st.metric("Total Chunks", total_chunks)
                with col3:
                    st.metric("Unique Persons", unique_persons)

# Initialize and render the page
def main():
    page = DocumentsPage()
    # Fetch existing documents when page loads
    asyncio.run(page.fetch_existing_documents())
    page.render()

if __name__ == "__main__":
    main()