import requests
import json


class OllamaModel:
    def __init__(self, model_name="llama3.2"):
        # Initialize model name
        print(f"Initializing model with name: {model_name}")
        self.model_name = model_name

    def generate_embedding(self, text):
        """Generate embedding for the input text using Ollama's /api/embed endpoint."""
        try:
            # Prepare the request payload
            data = {"model": self.model_name, "input": text}

            # Send POST request to the /api/embed endpoint
            response = requests.post("http://ollama:11434/api/embed", json=data)

            # Raise an exception if the request was unsuccessful
            response.raise_for_status()

            # Parse the response to get the embeddings
            embeddings = response.json().get("embeddings", None)

            if embeddings:
                print("Embedding generated successfully.")
                return embeddings[0]  # Return the first embedding if available
            else:
                print("Failed to generate embedding.")
                return None
        except Exception as e:
            print(f"An error occurred while generating the embedding: {str(e)}")
            return None
