from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union

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

class InitialQueryResponse(BaseModel):
    status: str
    message: str
    job_id: Optional[str] = None
    error: Optional[str] = None
    data: Dict[str, Any] = {
        "milvus_hits": 0,
        "person_contexts": 0,
        "mentioned_persons": [],
        "context_length": 0
    }

class CompletedQueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[QuerySource]
    person_contexts: List[PersonContext]
    confidence: float

class ErrorResponse(BaseModel):
    status: str = "error"
    message: str
    error: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

# Combined response type for API endpoint
QueryResponse = Union[InitialQueryResponse, CompletedQueryResponse, ErrorResponse]

class QueryStatusResponse(BaseModel):
    status: str
    job_id: str
    message: Optional[str] = None
    result: Optional[CompletedQueryResponse] = None
    error: Optional[str] = None