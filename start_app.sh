#!/bin/bash

# Navigate to the docker-compose directory
cd docker/postgres

CONTAINER_NAME="llamaindex-postgres"


# Check if the container exists
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "Container $CONTAINER_NAME already exists."

    # Check if the container is running
    if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
        echo "Stopping the running container $CONTAINER_NAME..."
        docker stop $CONTAINER_NAME
    fi

    echo "Removing the existing container $CONTAINER_NAME..."
    docker rm $CONTAINER_NAME
fi

# Start the PostgreSQL service with docker-compose
echo "Starting PostgreSQL service..."
docker-compose up -d

# Wait for the PostgreSQL container to become healthy
echo "Waiting for PostgreSQL to become ready..."
CONTAINER_NAME="llamaindex-postgres"
until docker inspect -f '{{.State.Health.Status}}' $CONTAINER_NAME | grep -q "healthy"; do
  sleep 2
  echo "Waiting for PostgreSQL container to be healthy..."
done

echo "PostgreSQL is ready."

# Navigate back to the project root
cd ../../

# Start the Streamlit app
echo "Starting Streamlit app..."
PYTHONPATH=$(pwd) streamlit run src/ui/app.py
