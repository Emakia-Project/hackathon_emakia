#!/bin/bash
set -e

PROJECT="emakia"
SERVICE="arize-agent"
REGION="us-central1"
IMAGE="gcr.io/${PROJECT}/${SERVICE}"

echo "🔨 Building container..."
gcloud builds submit --tag "${IMAGE}" .

echo "🚀 Deploying to Cloud Run..."
gcloud run deploy "${SERVICE}" \
  --image "${IMAGE}" \
  --platform managed \
  --region "${REGION}" \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 2 \
  --timeout 120 \
  --port 8080 \
  --set-secrets "OPENAI_API_KEY=OPENAI_API_KEY:latest,ANTHROPIC_API_KEY=ANTHROPIC_API_KEY:latest,GEMINI_API_KEY=GEMINI_API_KEY:latest,GROK_API_KEY=GROK_API_KEY:latest,PHOENIX_API_KEY=PHOENIX_API_KEY:latest" \
  --set-env-vars "PHOENIX_COLLECTOR_ENDPOINT=https://app.phoenix.arize.com/s/corinne"

echo "✅ Deployed! Service URL:"
gcloud run services describe "${SERVICE}" \
  --region "${REGION}" \
  --format "value(status.url)"
