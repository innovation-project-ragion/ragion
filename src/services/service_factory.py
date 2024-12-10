
from typing import Optional
from .query_service import QueryService

class ServiceFactory:
    _query_service: Optional[QueryService] = None

    @classmethod
    async def get_query_service(cls) -> QueryService:
        if cls._query_service is None:
            cls._query_service = QueryService()
        return cls._query_service