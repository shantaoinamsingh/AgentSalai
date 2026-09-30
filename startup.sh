#!/bin/bash

# Startup script for Azure App Services
# This script runs when the app starts on Azure

set -e

echo "Starting Salai on Azure App Services..."

# Create necessary directories
mkdir -p /home/site/wwwroot/uploads
mkdir -p /home/site/wwwroot/vectorstore

# Install dependencies
pip install -r requirements-azure.txt

# Run migrations/initialization if needed
# python manage_kb.py reindex --if-stale

# Start the application with gunicorn
gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:8000 --timeout 120 app:app
