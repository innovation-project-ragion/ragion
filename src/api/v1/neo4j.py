from fastapi import APIRouter, HTTPException, Depends
from src.db.neo4j import Neo4jClient
from typing import List, Dict, Any
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

async def get_neo4j_client():
    return Neo4jClient()

@router.get("/nodes")
async def get_nodes(
    neo4j_client: Neo4jClient = Depends(get_neo4j_client)
):
    """
    Retrieve all nodes from Neo4j.
    """
    try:
        with neo4j_client.driver.session() as session:
            result = session.run("""
                MATCH (n)
                RETURN n, labels(n) as labels
                LIMIT 1000
            """)
            
            nodes = []
            for record in result:
                node = record["n"]
                labels = record["labels"]
                nodes.append({
                    "id": node.id,
                    "labels": labels,
                    "properties": dict(node)
                })
            
            return nodes
    except Exception as e:
        logger.error(f"Error retrieving nodes: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.post("/query")
async def execute_query(
    query: str,
    parameters: Dict[str, Any] = None,
    neo4j_client: Neo4jClient = Depends(get_neo4j_client)
):
    """
    Execute a custom Cypher query.
    """
    try:
        with neo4j_client.driver.session() as session:
            result = session.run(query, parameters or {})
            records = [dict(record) for record in result]
            return records
    except Exception as e:
        logger.error(f"Error executing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.get("/relationships/{person_name}")
async def get_person_relationships(
    person_name: str,
    neo4j_client: Neo4jClient = Depends(get_neo4j_client)
):
    """
    Get all relationships for a specific person.
    """
    try:
        with neo4j_client.driver.session() as session:
            result = session.run("""
                MATCH (p:Person {name: $name})-[r]->(other)
                RETURN type(r) as relationship_type,
                       other.name as related_person,
                       labels(other) as related_labels
            """, {"name": person_name})
            
            relationships = []
            for record in result:
                relationships.append({
                    "type": record["relationship_type"],
                    "related_person": record["related_person"],
                    "labels": record["related_labels"]
                })
            
            return relationships
    except Exception as e:
        logger.error(f"Error retrieving relationships: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )