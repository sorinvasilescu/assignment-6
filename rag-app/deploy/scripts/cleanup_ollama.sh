#!/bin/bash

# Cleanup script for Ollama Docker containers, images, volumes, and local files

echo "Stopping and removing Ollama Docker containers..."

# Stop all running containers related to Ollama
docker ps -a | grep ollama | awk '{print $1}' | xargs -I {} docker stop {}
docker ps -a | grep ollama | awk '{print $1}' | xargs -I {} docker rm {}

echo "Removing Ollama Docker images..."

# Remove all Docker images related to Ollama
docker images | grep ollama | awk '{print $3}' | xargs -I {} docker rmi -f {}

echo "Removing unused Docker volumes..."

# Remove all unused Docker volumes
docker volume prune -f

echo "Removing unused Docker networks..."

# Remove all unused Docker networks
docker network prune -f

echo "Clearing local Ollama files..."

# Delete Ollama-related files in macOS's Application Support
rm -rf ~/Library/Application\ Support/Ollama

echo "Clearing Docker cache..."

# Clear Docker build cache to free up additional space
docker builder prune -f

echo "Checking Docker system space usage..."

# Display Docker disk usage to verify that space was cleared
docker system df

echo "Ollama cleanup completed successfully."
