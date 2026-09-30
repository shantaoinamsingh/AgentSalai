# Deployment Guide for Salai

This document covers deploying Salai to Azure App Services and Vercel.

## Overview

Salai is a Flask + SocketIO web application with persistent storage (ChromaDB) and document processing capabilities. Different platforms have different strengths:

- **Azure App Services**: Ideal for this full-featured Flask app with WebSockets and persistent storage
- **Vercel**: Optimized for serverless/frontend apps; limited support for long-running processes and WebSockets

## Recommended Architecture

```
┌─────────────────────────────────────────────────┐
│           Azure App Services                     │
│  ┌───────────────────────────────────────────┐  │
│  │  Salai Flask Backend + Frontend           │  │
│  │  - SocketIO WebSockets                    │  │
│  │  - ChromaDB Vector Store                  │  │
│  │  - Document Processing                    │  │
│  │  - Knowledge Base Management              │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

## Deployment to Azure App Services (Recommended)

### Prerequisites

- Azure Subscription
- Azure CLI installed (`az login`)
- Git repository

### Step 1: Create Azure App Service

```bash
# Create a resource group
az group create --name salai-rg --location eastus

# Create an App Service Plan (Free or B1 tier minimum)
az appservice plan create \
  --name salai-plan \
  --resource-group salai-rg \
  --sku F1 \
  --is-linux

# Create the web app
az webapp create \
  --resource-group salai-rg \
  --plan salai-plan \
  --name salai-chat \
  --runtime "PYTHON|3.11"
```

### Step 2: Configure Environment Variables

```bash
# Set required environment variables
az webapp config appsettings set \
  --resource-group salai-rg \
  --name salai-chat \
  --settings \
    SECRET_KEY="$(openssl rand -hex 32)" \
    ADMIN_API_KEY="$(openssl rand -hex 32)" \
    LOG_LEVEL="INFO" \
    FLASK_ENV="production"

# For LLM providers (optional - users set in Settings UI)
az webapp config appsettings set \
  --resource-group salai-rg \
  --name salai-chat \
  --settings \
    OPENAI_API_KEY="your-key" \
    ANTHROPIC_API_KEY="your-key"
```

### Step 3: Deploy from Git

```bash
# Connect to your GitHub repository
az webapp up \
  --name salai-chat \
  --resource-group salai-rg \
  --plan salai-plan \
  --runtime "PYTHON:3.11"

# Or use git push deployment
cd your-repo
git remote add azure <azure-git-url>
git push azure main
```

### Step 4: Configure Startup Command

```bash
az webapp config set \
  --resource-group salai-rg \
  --name salai-chat \
  --startup-file "startup.sh"
```

### Step 5: Enable WebSocket Support

```bash
az webapp config set \
  --resource-group salai-rg \
  --name salai-chat \
  --web-sockets-enabled true
```

### Step 6: Verify Deployment

```bash
# Get the URL
az webapp show \
  --resource-group salai-rg \
  --name salai-chat \
  --query defaultHostName

# Test the endpoint
curl https://salai-chat.azurewebsites.net/
```

## Configuration for Azure

### Environment Variables to Set

| Variable | Example | Notes |
|----------|---------|-------|
| `SECRET_KEY` | `$(openssl rand -hex 32)` | Flask session key (generate randomly) |
| `ADMIN_API_KEY` | `$(openssl rand -hex 32)` | Admin endpoint key (generate randomly) |
| `HOST` | `0.0.0.0` | Listen on all interfaces |
| `PORT` | `8000` | Azure passes through port 8000 |
| `FLASK_ENV` | `production` | Disable debug mode |
| `LOG_LEVEL` | `INFO` | Logging level |
| `VECTORSTORE_DIR` | `/home/site/wwwroot/vectorstore` | Persistent storage path |
| `UPLOADS_DIR` | `/home/site/wwwroot/uploads` | Upload storage path |

### Persistent Storage on Azure

Azure App Services provide `/home/site/wwwroot` for persistent files between restarts:

```bash
# Configure in your .env
VECTORSTORE_DIR=/home/site/wwwroot/vectorstore
UPLOADS_DIR=/home/site/wwwroot/uploads
```

## Deployment to Vercel (Not Recommended)

Vercel is optimized for serverless applications and has limitations for Salai:

- **Time Limits**: Vercel functions timeout after 60 seconds
- **WebSocket Limitations**: Limited WebSocket support
- **Stateless**: Each request goes to a separate function instance
- **Storage**: No persistent filesystem between requests

If you want to try Vercel:

### Prerequisites

- Vercel account
- Vercel CLI (`npm install -g vercel`)
- GitHub repository

### Step 1: Deploy

```bash
# Login to Vercel
vercel login

# Deploy from project root
vercel
```

### Step 2: Configure Build Settings

In Vercel dashboard:
- **Framework**: Flask
- **Build Command**: `pip install -r requirements.txt`
- **Output Directory**: (leave empty)
- **Root Directory**: (leave empty)

### Step 3: Set Environment Variables

In Vercel dashboard → Settings → Environment Variables:

```
SECRET_KEY: (generate with: python -c "import secrets; print(secrets.token_urlsafe(32))")
ADMIN_API_KEY: (generate with same method)
LOG_LEVEL: INFO
```

### Known Limitations on Vercel

1. **No Persistent Vector Store**: ChromaDB data will be lost on each function restart
2. **WebSocket Issues**: Real-time chat may be unreliable
3. **File Uploads**: Limited by serverless storage constraints
4. **Knowledge Base**: Organization context won't persist

**Workaround**: For Vercel, you would need to:
- Move ChromaDB to a managed service (Azure Cosmos DB, MongoDB Atlas)
- Use Azure Blob Storage for file uploads
- Implement API-only mode without real-time WebSockets

## Production Checklist

- [ ] Set `FLASK_ENV=production`
- [ ] Generate strong `SECRET_KEY` and `ADMIN_API_KEY`
- [ ] Enable HTTPS/SSL
- [ ] Set up logging and monitoring
- [ ] Configure backup strategy for knowledge base
- [ ] Set resource limits and quotas
- [ ] Enable authentication if needed
- [ ] Test with different LLM providers
- [ ] Set up CI/CD pipeline
- [ ] Monitor application logs
- [ ] Set up alerts for errors

## Monitoring and Logs

### Azure App Services

```bash
# View live logs
az webapp log tail \
  --resource-group salai-rg \
  --name salai-chat

# Download logs
az webapp log download \
  --resource-group salai-rg \
  --name salai-chat \
  --log-file logs.zip
```

### Environment Variables

Update .env file or use platform-specific configuration:

```bash
# Azure
az webapp config appsettings set \
  --resource-group salai-rg \
  --name salai-chat \
  --settings LOG_LEVEL="DEBUG"

# Vercel
vercel env add LOG_LEVEL
```

## Scaling Considerations

### Azure App Service

For production, upgrade from Free (F1) tier:

```bash
# Upgrade to Standard tier (S1)
az appservice plan update \
  --name salai-plan \
  --resource-group salai-rg \
  --sku S1
```

### Vercel

Vercel automatically scales serverless functions, but note:
- Each concurrent request = new function instance
- Each instance has isolated state
- No shared storage between instances

## Troubleshooting

### Issue: Module not found on Azure

```bash
# Verify requirements are installed
az webapp ssh --resource-group salai-rg --name salai-chat
pip list
```

### Issue: WebSocket connection fails

- Ensure WebSocket is enabled: `az webapp config set --web-sockets-enabled true`
- Check firewall rules aren't blocking WebSocket upgrade

### Issue: Knowledge base not persisting

- Verify `VECTORSTORE_DIR` path is writable
- Check storage quota on Azure App Service

### Issue: Slow file uploads

- Increase `MAX_UPLOAD_BYTES` in environment
- Consider upgrading App Service tier
- Use Azure CDN for static assets

## Cleanup

```bash
# Delete Azure resources
az group delete --name salai-rg --yes
```

## Support

For more information:
- [Azure App Service Documentation](https://docs.microsoft.com/azure/app-service/)
- [Vercel Deployment Documentation](https://vercel.com/docs)
- [Flask Deployment Guide](https://flask.palletsprojects.com/deployment/)
