from pydantic import BaseModel
from typing import List, Optional, Dict

class Query(BaseModel):
    text: str
    top_k: Optional[int] = 5
    use_neo4j: Optional[bool] = True
    include_relationships: Optional[bool] = True

class QueryStatus(BaseModel):
    status: str
    results: Optional[Dict] = None
        

class Source(BaseModel):
    text: str
    document_id: str
    score: float
    person_name: Optional[str]
    chunk_index: Optional[int]

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    confidence: float
    metadata: Optional[Dict] = None
    person_context: Optional[Dict] = None

class QueryHistory(BaseModel):
    question: str
    answer: str
    timestamp: str
    sources: List[Source]