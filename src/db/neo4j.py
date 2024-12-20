## src/db/neo4j.py
from neo4j import GraphDatabase
from src.core.config import settings
import logging
import re

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

    async def find_mentioned_persons(self, text: str) -> list:
        """Find persons mentioned in the text that exist in the database."""
        try:
            with self.driver.session() as session:
                # First, get all person names from the database
                result = session.run("""
                    MATCH (p:Person)
                    RETURN p.name as name
                """)
                names = [record["name"] for record in result]

                # Look for these names in the query text
                mentioned = []
                text_lower = text.lower()
                for name in names:
                    if name.lower() in text_lower:
                        mentioned.append(name)

                logger.info(f"Found mentioned persons: {mentioned}")
                return mentioned

        except Exception as e:
            logger.error(f"Error finding mentioned persons: {str(e)}")
            return []
    
    async def create_document_chunk(self, document_id: str, text: str, person_name: str, chunk_index: int):
        """Create a document chunk node and link it to a person."""
        try:
            with self.driver.session() as session:
                # Create document chunk and link to person in a single transaction
                session.run("""
                    MATCH (p:Person {name: $person_name})
                    MERGE (d:Document {id: $doc_id})
                    SET d.content = $text,
                        d.chunk_index = $chunk_index
                    MERGE (p)-[:APPEARS_IN]->(d)
                    
                    // Create relationships between chunks from same document
                    WITH d
                    MATCH (other:Document)
                    WHERE other.id <> $doc_id 
                    AND other.id STARTS WITH split($doc_id, '_')[0]
                    MERGE (d)-[:NEXT_CHUNK]->(other)
                """, {
                    "doc_id": document_id,
                    "text": text,
                    "person_name": person_name,
                    "chunk_index": chunk_index
                })
                
                logger.info(f"Created document chunk: {document_id} for person: {person_name}")
                
        except Exception as e:
            logger.error(f"Error creating document chunk: {str(e)}")
            raise

    async def get_person_context(self, person_name: str):
        """Get comprehensive context for a person."""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (p:Person {name: $name})
                    OPTIONAL MATCH (p)-[r]->(d:Document)
                    OPTIONAL MATCH (p)-[rel:APPEARS_IN]->(doc:Document)
                        
                    WITH p,
                        COLLECT(DISTINCT type(r)) as relationship_types,
                        COUNT(DISTINCT doc) as doc_count,
                        p.age as age
                        
                    RETURN {
                        name: p.name,
                        age: age,
                        relationships: relationship_types,
                        document_count: doc_count
                    } as context
                """, name=person_name)
                    
                record = result.single()
                return record["context"] if record else None
                    
        except Exception as e:
            logger.error(f"Error getting person context: {str(e)}")
            return None
        
    async def create_person(self, name: str, age: int = None):
        """Create a new Person node in the Neo4j database."""
        try:
            with self.driver.session() as session:
                session.run("""
                    MERGE (p:Person {name: $name})
                    ON CREATE SET p.age = $age
                """, name=name, age=age)
                logger.info(f"Created or updated person: {name}")
        except Exception as e:
            logger.error(f"Error creating person: {str(e)}")
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
                        context = await self.get_person_context(person_name)
                        if context:
                            person_contexts.append(context)
                
                return {
                    "documents": documents,
                    "person_contexts": person_contexts
                }
                
        except Exception as e:
            logger.error(f"Error getting Neo4j context: {str(e)}")
            raise

    def close(self):
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Closed Neo4j connection")