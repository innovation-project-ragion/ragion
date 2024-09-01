# Multimodal RAG Application

This project is a Retrieval-Augmented Generation (RAG) application built with FastAPI. The application leverages Milvus for vector storage, Neo4j for knowledge graphs, and integrates with a large language model (LLM) to provide multimodal capabilities.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Running Tests](#running-tests)
- [License](#license)

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



## Running Tests
    Make sure the services are running and run the tests using pytest:
    ```bash
    docker exec -it multimodal_rag_app pytest
    ```


    

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


