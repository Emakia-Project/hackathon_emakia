#!/bin/bash
# Emakia - Automated Cloud Deployment Script

PROJECT_ID="emakia"
REGION="us-central1"
SERVICE_NAME="emakia-app"
IMAGE="gcr.io/$PROJECT_ID/$SERVICE_NAME"

echo "🚀 Building Docker image..."
docker build -t $IMAGE .

echo "📦 Pushing to Google Container Registry..."
docker push $IMAGE

echo "☁️ Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --set-env-vars="GOOGLE_CLOUD_PROJECT=$PROJECT_ID"

echo "✅ Deployment complete!"
gcloud run services describe $SERVICE_NAME \
  --region $REGION \
  --format="value(status.url)"