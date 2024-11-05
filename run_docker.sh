#!/bin/bash

# Build the Docker image
docker build -t energy-predict-bdg2 .

# Run the Docker container
docker run -p 8000:8000 energy-predict-bdg2
