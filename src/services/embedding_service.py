import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from pathlib import Path
import logging
import os
from src.core.config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        # Get the path to hugging_face_models in project root
        src_dir = Path(__file__).parent.parent
        project_root = src_dir.parent 
        self.model_path = project_root / "hugging_face_models"
        
        logger.info(f"Model path set to: {self.model_path}")
        
        if not self.model_path.exists():
            logger.error(f"Model path does not exist: {self.model_path}")
            raise FileNotFoundError(f"Model not found at {self.model_path}")
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()

    def load_model(self):
        """Load model and tokenizer from local path."""
        try:
            logger.info(f"Loading model from: {self.model_path}")
            if not (self.model_path / "tokenizer_config.json").exists():
                raise FileNotFoundError(f"No tokenizer_config.json found in {self.model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path), 
                local_files_only=True
            )
            
            self.model = AutoModel.from_pretrained(
                str(self.model_path),
                local_files_only=True
            ).to(self.device)
            
            self.model.eval()
            logger.info(f"Model loaded successfully on {self.device}")
            
            # Verify output embedding dimension
            test_input = self.tokenizer("test", return_tensors="pt").to(self.device)
            with torch.no_grad():
                test_output = self.model(**test_input)
                output_dim = test_output.last_hidden_state.size(-1)
                logger.info(f"Model output embedding dimension: {output_dim}")
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    async def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        try:
            with torch.no_grad():
                # Tokenize and move to device
                inputs = self.tokenizer(
                    text,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)

                outputs = self.model(**inputs)

                hidden_states = outputs.last_hidden_state
                

                pooled = hidden_states.mean(dim=1)  # [batch_size, hidden_size]

                if pooled.shape[-1] != settings.EMBEDDING_DIM:
                    if pooled.shape[-1] > settings.EMBEDDING_DIM:
                        pooled = pooled[:, :settings.EMBEDDING_DIM]
                    else:
                        padding = torch.zeros(1, settings.EMBEDDING_DIM - pooled.shape[-1], device=self.device)
                        pooled = torch.cat([pooled, padding], dim=1)
                
                # Normalize
                normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
                
                return normalized[0].cpu().numpy()

        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise