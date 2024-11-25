from docx import Document
from src.models.document import ProcessedChunk
from src.core.config import settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
import re
from typing import List, Dict, Any
import os

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, chunk_size=400, chunk_overlap=80):
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", ":", ";", ",", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            keep_separator=True,
            add_start_index=True
        )
        self.clean_patterns = [
            (r'\s+', ' '),
            (r'[\(\{\[\]\}\)]', ''),
            (r'[^\w\s\.\,\?\!\-\:\;äöåÄÖÅ]', ''),
            (r'\s+\.', '.'),
            (r'\.+', '.'),
        ]

    async def process_file(self, file) -> List[ProcessedChunk]:
        try:
            # Save uploaded file temporarily
            temp_path = f"temp_{file.filename}"
            with open(temp_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Process the document
            chunks = self.process_document(temp_path)
            
            # Clean up
            os.remove(temp_path)
            
            return chunks
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            raise
        
    def process_document(self, file_path: str) -> List[ProcessedChunk]:
        try:
            # Read document
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            
            # Extract metadata
            filename = os.path.basename(file_path)
            name, age, doc_id = self.extract_metadata_from_filename(filename)
            
            # Preprocess and split text
            clean_text = self.preprocess_text(text)
            chunks = self.text_splitter.split_text(clean_text)
            
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                importance_score = self._calculate_chunk_importance(chunk)
                
                processed_chunks.append(ProcessedChunk(
                    text=chunk,
                    metadata={
                        "source": filename,
                        "person_name": name,
                        "person_age": age,
                        "document_id": doc_id,
                        "chunk_index": i,
                        "importance_score": importance_score,
                    }
                ))
            
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise

    def _calculate_chunk_importance(self, chunk: str) -> float:
        score = 1.0
        chunk_lower = chunk.lower()
        
        # Key phrase indicators
        key_phrases = {
            'high': ['erittäin tärkeä', 'merkittävä', 'olennainen', 'keskeinen'],
            'medium': ['tärkeä', 'huomattava', 'kiinnostava'],
            'low': ['mainittava', 'mahdollinen']
        }
        
        for phrase in key_phrases['high']:
            if phrase in chunk_lower:
                score *= 1.3
        for phrase in key_phrases['medium']:
            if phrase in chunk_lower:
                score *= 1.2
        for phrase in key_phrases['low']:
            if phrase in chunk_lower:
                score *= 1.1
                
        return score

    def extract_metadata_from_filename(self, filename: str) -> tuple:
        title = os.path.splitext(filename)[0]
        match = re.match(r'([A-Za-z]+)\s+(\d{1,3})v\s+([A-Za-z0-9\-]+)', title)
        if match:
            return match.group(1), int(match.group(2)), match.group(3)
        return None, None, None

    def preprocess_text(self, text: str) -> str:
        try:
            for pattern, replacement in self.clean_patterns:
                text = re.sub(pattern, replacement, text)
            text = re.sub(r'([.!?])\s*([A-ZÄÖÅ])', r'\1\n\2', text)
            text = '\n'.join(line.strip() for line in text.split('\n'))
            return text.strip()
        except Exception as e:
            logger.error(f"Error preprocessing text: {str(e)}")
            raise