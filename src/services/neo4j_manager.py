import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel
from src.core.config import settings
import logging
import hashlib
from typing import List, Optional

logger = logging.getLogger(__name__)

class EnhancedEmbeddingManager:
    def __init__(self, 
                 model_name: str = "TurkuNLP/sbert-cased-finnish-paraphrase",
                 cache_dir: str = "/scratch/project_2011638/huggingface_cache",
                 device: str = None):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_cache = {}
        self._initialize_model()

    def _initialize_model(self):
        """Initialize model with proper error handling."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            self.model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            ).to(self.device)
            self.model.eval()
            
            self.embedding_dim = self.model.config.hidden_size
            logger.info(f"Loaded {self.model_name} (dim={self.embedding_dim})")
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise

    def _mean_pooling(self, model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Perform mean pooling on token embeddings."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def generate(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """Generate embeddings with improved batching and error handling."""
        try:
            all_embeddings = []
            
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Check cache first
                batch_embeddings = []
                uncached_texts = []
                uncached_indices = []
                
                for j, text in enumerate(batch_texts):
                    cache_key = hash(text)
                    if cache_key in self.embedding_cache:
                        batch_embeddings.append(self.embedding_cache[cache_key])
                    else:
                        uncached_texts.append(text)
                        uncached_indices.append(j)
                
                if uncached_texts:
                    # Generate new embeddings
                    with torch.no_grad():
                        encoded_input = self.tokenizer(
                            uncached_texts,
                            padding=True,
                            truncation=True,
                            max_length=512,
                            return_tensors='pt'
                        ).to(self.device)
                        
                        model_output = self.model(**encoded_input)
                        sentence_embeddings = self._mean_pooling(
                            model_output,
                            encoded_input['attention_mask']
                        )
                        # Normalize embeddings
                        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
                        
                        # Move to CPU and convert to numpy
                        embeddings_np = sentence_embeddings.cpu().numpy()
                        
                        # Cache new embeddings
                        for text, embedding in zip(uncached_texts, embeddings_np):
                            self.embedding_cache[hash(text)] = embedding
                            
                        # Insert new embeddings into correct positions
                        for idx, embedding in zip(uncached_indices, embeddings_np):
                            batch_embeddings.insert(idx, embedding)
                
                all_embeddings.extend(batch_embeddings)
            
            final_embeddings = np.vstack(all_embeddings)
            
            # Verify embedding dimension
            if final_embeddings.shape[1] != self.embedding_dim:
                raise ValueError(
                    f"Embedding dimension mismatch. Expected {self.embedding_dim}, "
                    f"got {final_embeddings.shape[1]}"
                )
            
            return final_embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def compute_similarity(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and documents."""
        return np.dot(doc_embeddings, query_embedding.T).squeeze()

    def batch_compute_similarity(self, queries: List[str], documents: List[str]) -> np.ndarray:
        """Compute similarities between multiple queries and documents efficiently."""
        query_embeddings = self.generate(queries)
        doc_embeddings = self.generate(documents)
        
        # Compute similarity matrix
        similarity_matrix = np.dot(query_embeddings, doc_embeddings.T)
        
        return similarity_matrix

    def clear_cache(self):
        """Clear the embedding cache."""
        self.embedding_cache.clear()
        logger.info("Cleared embedding cache")