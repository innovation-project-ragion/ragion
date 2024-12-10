import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_query(input_file: str, output_file: str):
    try:
        # Load input data
        with open(input_file, 'r') as f:
            input_data = json.load(f)
        
        # Initialize models
        tokenizer = AutoTokenizer.from_pretrained(
            "Finnish-NLP/llama-7b-finnish-instruct-v0.2",
            cache_dir="/scratch/project_2011638/huggingface_cache"
        )
        
        llm_pipeline = pipeline(
            "text-generation",
            model="Finnish-NLP/llama-7b-finnish-instruct-v0.2",
            tokenizer=tokenizer,
            device="cuda"
        )
        
        # Process query
        outputs = llm_pipeline(
            input_data['prompt'],
            max_new_tokens=input_data.get('max_tokens', 300),
            do_sample=True,
            temperature=input_data.get('temperature', 0.1),
            top_p=input_data.get('top_p', 0.95)
        )
        
        # Save results
        results = {
            "job_id": input_data.get('job_id'),
            "query_id": input_data.get('query_id'),
            "generated_text": outputs[0]["generated_text"]
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Successfully processed query {input_data.get('query_id')}")
        
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        logger.error(error_msg)
        
        # Save error information
        with open(output_file, 'w') as f:
            json.dump({
                "error": error_msg,
                "job_id": input_data.get('job_id'),
                "query_id": input_data.get('query_id')
            }, f)
        
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input JSON file path')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    args = parser.parse_args()
    
    process_query(args.input, args.output)