from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np
from src.core.config import settings
import logging
from typing import List

logger = logging.getLogger(__name__)

class EmbeddingManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() and settings.USE_GPU else "cpu"
        self._initialize_model()
        self.embedding_cache = {}

    def _initialize_model(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.EMBEDDING_MODEL,
                cache_dir=settings.CACHE_DIR
            )
            self.model = AutoModel.from_pretrained(
                settings.EMBEDDING_MODEL,
                cache_dir=settings.CACHE_DIR
            ).to(self.device)
            
            self.model.eval()
            logger.info(f"Initialized embedding model on {self.device}")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {str(e)}")
            raise

    def generate_embeddings(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        try:
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Use cache when possible
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
                    with torch.no_grad():
                        inputs = self.tokenizer(
                            uncached_texts,
                            padding=True,
                            truncation=True,
                            max_length=512,
                            return_tensors='pt'
                        ).to(self.device)
                        
                        outputs = self.model(**inputs)
                        embeddings = self._mean_pooling(
                            outputs,
                            inputs['attention_mask']
                        )
                        embeddings = F.normalize(embeddings, p=2, dim=1)
                        
                        # Cache new embeddings
                        embeddings_np = embeddings.cpu().numpy()
                        for text, embedding in zip(uncached_texts, embeddings_np):
                            self.embedding_cache[hash(text)] = embedding
                            
                        # Insert new embeddings into correct positions
                        for idx, embedding in zip(uncached_indices, embeddings_np):
                            batch_embeddings.insert(idx, embedding)
                
                all_embeddings.extend(batch_embeddings)
            
            return np.vstack(all_embeddings)
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def clear_cache(self):
        """Clear the embedding cache."""
        self.embedding_cache.clear()
        logger.info("Cleared embedding cache")