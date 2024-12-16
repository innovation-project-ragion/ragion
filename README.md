# Multimodal RAG Application

This project is a Retrieval-Augmented Generation (RAG) application built with FastAPI. The application leverages Milvus for vector storage, Neo4j for knowledge graphs, and integrates with a large language model (LLM) to provide multimodal capabilities.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Running Tests](#running-tests)
- [License](#license)

## Introduction
Ragion is a multimodal RAG application designed to assist users and answer questions on specific topics. One potential use case for Ragion is assisting with information related to sheltered accommodation residents' experiences and living conditions. However, the application has many other potential uses. The main requirement for setting up this application is the availability of trustworthy material to serve as the source of the bot's knowledge. This material can include interviews, observations, experiment results, or any other kind of reliable and proven information. During the testing phase, interviews with sheltered accommodation residents were used to build the chatbot's knowledge base, and this material proved to be effective for bot training.

The application was implemented using open-source products, technologies, and free versions of commercial tools. It is divided into components, each running in an individual Docker container. The application logic is written in Python, using the FastAPI framework for the frontend and Streamlit for the backend. The free version of the Neo4j vector database was used as the knowledge base, while the Milvus database served as embedding storage. Langchain, Hugging Face Transformer Models, and TurkuNLP were used for query processing and embedding generation. Finally, the Finnish-NLP/llama-7b-finnish-instruct-v0.2 Large Language Model (LLM) was used to generate responses.

The application is resource-intensive and requires a high-powered environment for smooth operation. During development, it was deployed on a supercomputer, where it took approximately two to three minutes to generate a response. On a regular home computer, the waiting time was around 15 minutes.

More information about the essential technologies used in Ragion's development can be found in the following chapters.


## Requirements

Before you begin, ensure you have met the following requirements:

- Docker: [Install Docker](https://docs.docker.com/get-docker/)
- Docker Compose: [Install Docker Compose](https://docs.docker.com/compose/install/)

## Installation

Follow these steps to set up and run the application:

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/tahasafdari/ragion
    cd ragion
    ```

2. **Build and Start the Application:**
    For local development create network:
    ```bash
    docker network create app_network
    ```

    Using Docker Compose, you can build and start the application along with its dependencies (Milvus and Neo4j):

    ```bash
    docker-compose up --build
    ```

    This command will:
    - Build the Docker image for the FastAPI application.
    - Start the Milvus and Neo4j services.
    - Launch the FastAPI application, accessible at `http://localhost:8000`.

3. **Access Neo4j Browser (Optional):**

    Neo4j’s web interface is available at `http://localhost:7474`. The default login credentials are:
    - Username: `neo4j`
    - Password: `test`

## Usage

    in case if you get the permission denied for multimodal_rag_app container please execute the following:
    
    
    
    Linux Machine:
    ```bash
    chmod +x ./scripts/start-dev.sh
    ```
    
    Windows Machine:



## Running Tests
    Make sure the services are running and run the tests using pytest:
    ```bash
    docker exec -it multimodal_rag_app pytest
    ```


    

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


