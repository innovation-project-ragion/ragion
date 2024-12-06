from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    text: str
    max_tokens: Optional[int] = 300
    temperature: Optional[float] = 0.1
    top_k: Optional[int] = 5

class QuerySource(BaseModel):
    text: str
    document_id: str
    score: float

class PersonContext(BaseModel):
    name: str
    age: Optional[int]
    relationships: List[str]
    document_count: int

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[QuerySource]
    person_contexts: List[PersonContext]
    confidence: float