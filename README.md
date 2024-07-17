# Chat RAG - based on a website + uploaded PDFs
A simple RAG based chatbot service (no chat history) that

## Getting Started
First, set the environment variable for the URL to scrape in docker-compose.yml, under the headfittedchat service. Alternatively, change the URL in app/app.py

To build a container using Docker, then run the Compose config, you can follow these steps:

1. Open a terminal or command prompt.
2. Navigate to the project directory.
3. Run the following command to build the Docker image:
    ```
    docker build . -t proinators/headfittedchat:0.1
    ```
    This command builds the Docker image using the Dockerfile in the current directory and tags it with the specified name and version.

Once the image is built, you can use Docker Compose to start the container:
```
docker compose up
```

This command will start the container based on the configuration specified in the `docker-compose.yml` file.


