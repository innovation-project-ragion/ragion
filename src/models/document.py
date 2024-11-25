from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime

class DocumentBase(BaseModel):
    content: str
    metadata: Dict

class DocumentCreate(DocumentBase):
    pass

class DocumentResponse(DocumentBase):
    id: str
    chunk_index: int
    person_name: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True

class ProcessedChunk(BaseModel):
    text: str
    metadata: Dict
    embedding: Optional[List[float]]

class SearchResult(BaseModel):
    text: str
    score: float
    document_id: str
    person_name: Optional[str]
    chunk_index: int