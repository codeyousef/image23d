#!/bin/bash

# NeuralForge Studio Production Deployment Script
# Deploys to Google Cloud Run

set -e

# Configuration
PROJECT_ID=${GCP_PROJECT_ID:-"neuralforge-studio"}
REGION=${GCP_REGION:-"us-central1"}
SERVICE_NAME="neuralforge-api"
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting NeuralForge Studio deployment...${NC}"

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}Error: gcloud CLI not found. Please install Google Cloud SDK.${NC}"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker not found. Please install Docker.${NC}"
    exit 1
fi

# Authenticate with Google Cloud
echo -e "${YELLOW}Authenticating with Google Cloud...${NC}"
gcloud auth configure-docker

# Set project
gcloud config set project $PROJECT_ID

# Build backend image
echo -e "${YELLOW}Building backend Docker image...${NC}"
docker build -f deployment/Dockerfile.backend -t $IMAGE_NAME:latest .

# Tag with version
VERSION=$(git rev-parse --short HEAD)
docker tag $IMAGE_NAME:latest $IMAGE_NAME:$VERSION

# Push to Container Registry
echo -e "${YELLOW}Pushing image to Container Registry...${NC}"
docker push $IMAGE_NAME:latest
docker push $IMAGE_NAME:$VERSION

# Deploy to Cloud Run
echo -e "${YELLOW}Deploying to Cloud Run...${NC}"
gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_NAME:latest \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --set-env-vars "RUNPOD_API_KEY=$RUNPOD_API_KEY" \
    --set-env-vars "SECRET_KEY=$SECRET_KEY" \
    --set-env-vars "REDIS_URL=$REDIS_URL" \
    --cpu 2 \
    --memory 4Gi \
    --min-instances 0 \
    --max-instances 100 \
    --concurrency 100 \
    --timeout 300

# Get service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)')

echo -e "${GREEN}Deployment completed successfully!${NC}"
echo -e "${GREEN}Service URL: $SERVICE_URL${NC}"

# Deploy frontend to Firebase Hosting (optional)
read -p "Deploy frontend to Firebase Hosting? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Building frontend...${NC}"
    cd frontend
    npm run build
    
    echo -e "${YELLOW}Deploying to Firebase...${NC}"
    firebase deploy --only hosting
    
    cd ..
fi

echo -e "${GREEN}Deployment complete!${NC}"