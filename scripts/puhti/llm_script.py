import argparse
import json
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMProcessor:
    def __init__(self):
        self.model_name = "Finnish-NLP/llama-7b-finnish-instruct-v0.2"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()

    def load_model(self):
        """Load model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            logger.info(f"Model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def generate_response(self, query: str, context: str, params: dict) -> str:
        """Generate response using the model."""
        try:
            # Format prompt
            prompt = f"""Tehtävä: Käytä annettua kontekstia vastataksesi kysymykseen mahdollisimman tarkasti.

Konteksti:
{context}

Kysymys: {query}

Vastausohjeet:
1. Jos löydät suoran vastauksen kontekstista:
   - Mainitse dokumentti, josta vastaus löytyy
   - Käytä suoria lainauksia
   - Mainitse vastauksen luotettavuus
2. Jos löydät vain osittaisen vastauksen:
   - Kerro mitä tietoa löysit ja mitä puuttuu
3. Jos et löydä vastausta:
   - Ilmoita selkeästi, ettei vastausta löydy
   
Vastaus:"""

            # Generate response
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=params.get("max_tokens", 300),
                temperature=params.get("temperature", 0.1),
                top_p=params.get("top_p", 0.95),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

            response = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            ).split("Vastaus:")[1].strip()

            return response

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

def main(input_path: str):
    try:
        # Load input data
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        # Initialize processor
        processor = LLMProcessor()
        
        # Generate response
        response = processor.generate_response(
            query=data["query"],
            context=data["context"],
            params=data.get("params", {})
        )
        
        # Save output
        output_path = Path(input_path).parent / f"response_{Path(input_path).stem}.json"
        with open(output_path, 'w') as f:
            json.dump({
                "response": response,
                "status": "completed"
            }, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Response saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON file")
    args = parser.parse_args()
    main(args.input)