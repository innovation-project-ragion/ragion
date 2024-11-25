from typing import Dict, List, Any
from src.db.milvus import MilvusClient
from src.db.neo4j import Neo4jClient
from src.services.embedding_manager import EmbeddingManager
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from src.core.config import settings
import torch
import logging
from langchain.memory import ConversationBufferMemory

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self):
        self._initialize_components()
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.min_similarity_threshold = 0.65

    def _initialize_components(self):
        try:
            # Initialize database clients
            self.milvus_client = MilvusClient()
            self.neo4j_client = Neo4jClient()
            self.embedding_manager = EmbeddingManager()

            # Initialize LLM components
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.MODEL_ID,
                cache_dir=settings.CACHE_DIR
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                settings.MODEL_ID,
                quantization_config=bnb_config,
                device_map="auto",
                cache_dir=settings.CACHE_DIR
            )

            logger.info("Successfully initialized RAG pipeline components")
        except Exception as e:
            logger.error(f"Error initializing RAG pipeline: {str(e)}")
            raise

    async def process_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        try:
            # Generate query embedding
            query_embedding = self.embedding_manager.generate_embeddings([query])[0]
            
            # Get relevant documents from Milvus
            vector_results = await self.milvus_client.search(
                query_embedding.tolist(),
                limit=top_k
            )
            
            # Get Neo4j context
            mentioned_persons = self._extract_person_names(query)
            graph_context = await self.neo4j_client.get_context(query, mentioned_persons)
            
            # Combine contexts
            combined_context = self._combine_contexts(vector_results, graph_context)
            
            # Generate response
            response = await self._generate_response(query, combined_context)
            
            # Store in memory
            self.memory.save_context(
                {"input": query},
                {"output": response["answer"]}
            )
            
            return response
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise

    def _extract_person_names(self, text: str) -> List[str]:
        """Extract potential person names from text using Finnish patterns."""
        names = set()
        patterns = [
            r'\b[A-ZÄÖÅ][a-zäöå]+\b',
            r'(?:herra|rouva|neiti)\s+[A-ZÄÖÅ][a-zäöå]+',
            r'(?:ystävä|naapuri)\s+[A-ZÄÖÅ][a-zäöå]+'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                name = match.group().split()[-1]
                if len(name) > 2:
                    names.add(name)
        
        return list(names)

    def _combine_contexts(self, vector_results: List[Dict], graph_context: Dict) -> Dict:
        """Combine vector search results with graph context."""
        combined = {
            "documents": [],
            "relationships": [],
            "persons": []
        }
        
        # Add vector search results
        for result in vector_results:
            combined["documents"].append({
                "content": result["text"],
                "source": result["document_id"],
                "relevance": result["score"]
            })
        
        # Add graph context
        if graph_context.get("documents"):
            for doc in graph_context["documents"]:
                if doc not in combined["documents"]:
                    combined["documents"].append(doc)
        
        # Add person contexts
        if graph_context.get("person_contexts"):
            combined["persons"].extend(graph_context["person_contexts"])
            
        return combined

    async def _generate_response(self, question: str, context: Dict) -> Dict:
        try:
            prompt = self._create_prompt(question, context)
            
            # Generate response
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.1,
                top_p=0.95,
                repetition_penalty=1.2
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Process and clean response
            answer = self._process_response(response)
            
            return {
                "answer": answer,
                "sources": context["documents"][:3],
                "confidence": max((doc.get("relevance", 0) for doc in context["documents"]), default=0),
                "person_context": context["persons"][0] if context["persons"] else None
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

    def _create_prompt(self, question: str, context: Dict) -> str:
        formatted_context = self._format_context(context)
        
        return f"""Tehtävä: Vastaa annettuun kysymykseen käyttäen vain annettua kontekstia.

Kysymys: {question}

Konteksti:
{formatted_context}

Vastausohjeet:
1. Jos löydät suoran vastauksen:
   - Mainitse dokumentti, josta vastaus löytyy
   - Käytä suoria lainauksia
   - Arvioi vastauksen luotettavuus
2. Jos et löydä vastausta:
   - Ilmoita selkeästi, ettei vastausta löydy annetusta kontekstista
3. Jos löydät vain osittaisen vastauksen:
   - Kerro mitä tietoa löysit ja mitä puuttuu

Vastaus:"""

    def _format_context(self, context: Dict) -> str:
        formatted_parts = []
        
        # Format documents
        for i, doc in enumerate(context["documents"], 1):
            formatted_parts.append(
                f"Dokumentti {i}:\n"
                f"Lähde: {doc['source']}\n"
                f"Teksti: {doc['content']}\n"
            )
        
        # Format person information
        for person in context["persons"]:
            formatted_parts.append(
                f"Henkilö: {person['name']}\n"
                f"Suhteet: {', '.join(person['relationship_types'])}\n"
            )
            
        return "\n".join(formatted_parts)

    def _process_response(self, response: str) -> str:
        """Clean and process the generated response."""
        # Remove common prefixes
        prefixes_to_remove = [
            "Vastaus:",
            "Tämän perusteella",
            "Kontekstin perusteella",
        ]
        
        result = response
        for prefix in prefixes_to_remove:
            if result.startswith(prefix):
                result = result[len(prefix):].strip()
        
        return result.strip()