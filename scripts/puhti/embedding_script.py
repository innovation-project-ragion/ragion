## this the embedding script which I have it on the puhti server
import argparse
import json
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch
from typing import Tuple, List
import os
from docx import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "TurkuNLP/sbert-cased-finnish-paraphrase"
CACHE_DIR = "/scratch/project_2011638/safdarih/huggingface_cache"

def read_docx(file_path: str) -> List[str]:
    """Read text from a .docx file."""
    try:
        doc = Document(file_path)
        # Extract text from paragraphs
        texts = [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()]
        logger.info(f"Extracted {len(texts)} paragraphs from document")
        return texts
    except Exception as e:
        logger.error(f"Error reading .docx file: {str(e)}")
        raise

def setup_gpu() -> None:
    """Setup GPU device if available."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("No GPU available, using CPU")
    return device

def load_model_and_tokenizer(cache_dir: str) -> Tuple[AutoTokenizer, AutoModel]:
    """
    Load the model and tokenizer from local cache or download from HuggingFace.
    """
    try:
        logger.info(f"Loading model and tokenizer from: {MODEL_NAME}")
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load tokenizer and model with explicit cache directory
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=cache_dir,
            local_files_only=False  # Allow downloading if not in cache
        )
        
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            cache_dir=cache_dir,
            local_files_only=False  # Allow downloading if not in cache
        )
        
        return tokenizer, model
    
    except Exception as e:
        logger.error(f"Error loading model and tokenizer: {str(e)}")
        raise

def generate_embeddings(texts: List[str], tokenizer: AutoTokenizer, model: AutoModel, device: torch.device) -> torch.Tensor:
    """Generate embeddings for the given texts."""
    try:
        # Move model to the appropriate device
        model = model.to(device)
        model.eval()
        
        with torch.no_grad():
            # Tokenize texts
            encoded_input = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(device)
            
            # Generate embeddings
            outputs = model(**encoded_input)
            embeddings = outputs.last_hidden_state[:, 0, :]  # Use CLS token embedding
            
            return embeddings.cpu()  # Return embeddings back to CPU
            
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise

def main(input_path: str):
    try:
        # Setup device
        device = setup_gpu()
        
        # Load input data
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Handle .docx files
        if input_file.suffix.lower() == '.docx':
            texts = read_docx(input_path)
        else:
            # Keep existing JSON handling
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                texts = data.get('texts', [])
        
        if not texts:
            raise ValueError("No texts found in input file")
            
        logger.info(f"Processing {len(texts)} text segments")
            
        # Load model and tokenizer
        tokenizer, model = load_model_and_tokenizer(CACHE_DIR)
        
        # Generate embeddings
        embeddings = generate_embeddings(texts, tokenizer, model, device)
        
        # Save results
        output_path = input_file.parent / f"embeddings_{input_file.stem}.pt"
        torch.save(embeddings, output_path)
        
        # Also save the texts for reference
        texts_output_path = input_file.parent / f"texts_{input_file.stem}.json"
        with open(texts_output_path, 'w', encoding='utf-8') as f:
            json.dump({'texts': texts}, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Embeddings saved to: {output_path}")
        logger.info(f"Texts saved to: {texts_output_path}")
        
        # Log embedding tensor shape for verification
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        
    except Exception as e:
        logger.error(f"Error during embedding generation: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for input texts")
    parser.add_argument("--input", type=str, required=True, help="Path to input file (.docx or .json)")
    args = parser.parse_args()
    main(args.input)