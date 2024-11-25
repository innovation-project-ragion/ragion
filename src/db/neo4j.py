from neo4j import GraphDatabase
from src.core.config import settings
import logging

logger = logging.getLogger(__name__)

class Neo4jClient:
    def __init__(self):
        self.driver = None
        self.connect()

    def connect(self):
        """Establish connection to Neo4j."""
        try:
            self.driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
            )
            # Verify connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Connected to Neo4j successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise

    async def get_context(self, question: str, mentioned_persons: list = None):
        """Get context from Neo4j for a question."""
        try:
            with self.driver.session() as session:
                # Get related documents
                documents = []
                if mentioned_persons:
                    for person in mentioned_persons:
                        result = session.run("""
                            MATCH (p:Person {name: $name})-[:APPEARS_IN]->(d:Document)
                            RETURN d.content as content, d.id as doc_id, 
                                   d.chunk_index as chunk_index
                            ORDER BY d.chunk_index
                        """, {"name": person})
                        
                        for record in result:
                            documents.append({
                                "content": record["content"],
                                "doc_id": record["doc_id"],
                                "chunk_index": record["chunk_index"],
                                "relevance": 1.0
                            })
                
                # Get person contexts
                person_contexts = []
                if mentioned_persons:
                    for person_name in mentioned_persons:
                        context = self._get_person_context(session, person_name)
                        if context:
                            person_contexts.append(context)
                
                return {
                    "documents": documents,
                    "person_contexts": person_contexts
                }
                
        except Exception as e:
            logger.error(f"Error getting Neo4j context: {str(e)}")
            raise

    def _get_person_context(self, session, person_name: str):
        """Get comprehensive context for a person."""
        result = session.run("""
            MATCH (p:Person {name: $name})
            OPTIONAL MATCH (p)-[:APPEARS_IN]->(d:Document)
            OPTIONAL MATCH (p)-[r]->(other:Person)
            
            WITH p,
                 COLLECT(DISTINCT type(r)) as relationship_types,
                 COLLECT(DISTINCT d.content) as contents,
                 COLLECT(DISTINCT d.id) as doc_ids
            
            RETURN {
                name: p.name,
                relationship_types: relationship_types,
                document_count: SIZE(contents),
                document_ids: doc_ids
            } as context
        """, {"name": person_name})
        
        return result.single()["context"] if result.peek() else None

    def close(self):
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Closed Neo4j connection")