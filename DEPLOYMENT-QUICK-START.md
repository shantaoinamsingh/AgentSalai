# Deployment Quick Start Guide

Quick reference for deploying Salai to Azure App Services or Vercel.

## TL;DR

### Azure App Services (Recommended) ⭐

```bash
# 1. Create Azure resources
az group create --name salai-rg --location eastus
az appservice plan create --name salai-plan --resource-group salai-rg --sku F1 --is-linux
az webapp create --resource-group salai-rg --plan salai-plan --name salai-chat --runtime "PYTHON|3.11"

# 2. Set secrets
RAND_KEY=$(openssl rand -hex 32)
az webapp config appsettings set --resource-group salai-rg --name salai-chat \
  --settings SECRET_KEY="$RAND_KEY" ADMIN_API_KEY="$RAND_KEY" FLASK_ENV="production"

# 3. Deploy
git push azure main

# 4. Enable features
az webapp config set --resource-group salai-rg --name salai-chat --web-sockets-enabled true
az webapp config set --resource-group salai-rg --name salai-chat --startup-file "startup.sh"

# 5. Verify
curl https://salai-chat.azurewebsites.net/
```

### Vercel (Limited) ⚠️

```bash
# 1. Install CLI
npm install -g vercel

# 2. Deploy
vercel

# 3. Set environment (in dashboard or CLI)
vercel env add SECRET_KEY
vercel env add ADMIN_API_KEY
vercel env add FLASK_ENV=production

# 4. Redeploy
vercel --prod
```

## Platform Comparison

| Feature | Azure App Services | Vercel | Docker |
|---------|-------------------|--------|--------|
| **Ease of Setup** | Medium | Easy | Medium |
| **Cost** | Free (F1) | Free tier | Pay-as-you-go |
| **WebSocket Support** | ✅ Yes | ⚠️ Limited | ✅ Yes |
| **Persistent Storage** | ✅ Yes | ❌ No | ✅ Yes |
| **Long-running Tasks** | ✅ Yes | ❌ 60s timeout | ✅ Yes |
| **Knowledge Base** | ✅ Works | ❌ Resets | ✅ Works |
| **Best For** | Production | Static/API | Development |

## Deployment Methods

### Method 1: Git Push (Recommended for Azure)

```bash
# Add Azure remote
git remote add azure <your-git-url-from-azure>

# Push to deploy
git push azure main
```

**Pros**: Automatic deploys on push, version control, rollback easy

### Method 2: Azure CLI (Azure Only)

```bash
az webapp up --name salai-chat --resource-group salai-rg
```

**Pros**: One command, simple

### Method 3: Docker

```bash
# Build locally
docker build -t salai:latest .

# Test locally
docker-compose up

# Push to registry
docker tag salai:latest your-registry.azurecr.io/salai:latest
az acr build --registry your-registry --image salai:latest .

# Deploy to Azure Container Instances
az container create --resource-group salai-rg --name salai --image your-registry.azurecr.io/salai:latest
```

**Pros**: Works everywhere, consistent environment

### Method 4: Vercel CLI

```bash
vercel --prod
```

**Pros**: Simple, automatic HTTPS

## Environment Variables

### Required (all platforms)

```env
SECRET_KEY=<generate with: openssl rand -hex 32>
ADMIN_API_KEY=<generate with: openssl rand -hex 32>
FLASK_ENV=production
```

### Optional (recommended for production)

```env
LOG_LEVEL=INFO
SESSION_COOKIE_SECURE=true  # Only with HTTPS
```

### Platform-specific (Azure)

```env
HOST=0.0.0.0
PORT=8000
VECTORSTORE_DIR=/home/site/wwwroot/vectorstore
UPLOADS_DIR=/home/site/wwwroot/uploads
```

## Troubleshooting

### 502 Bad Gateway

```bash
# Check logs
az webapp log tail --resource-group salai-rg --name salai-chat

# Restart app
az webapp restart --resource-group salai-rg --name salai-chat
```

### WebSocket Connection Failed

```bash
# Enable WebSockets
az webapp config set --resource-group salai-rg --name salai-chat --web-sockets-enabled true
```

### Module Not Found

```bash
# Check Python version
az webapp ssh --resource-group salai-rg --name salai-chat
python --version

# Reinstall dependencies
pip install -r requirements-azure.txt
```

### Knowledge Base Empty

```bash
# SSH into Azure
az webapp ssh --resource-group salai-rg --name salai-chat

# Check directories
ls -la /home/site/wwwroot/vectorstore
ls -la /home/site/wwwroot/uploads

# Reindex if needed
cd /home/site/wwwroot
python manage_kb.py reindex
```

## Monitoring

### Azure App Services

```bash
# View metrics
az monitor metrics list --resource /subscriptions/{id}/resourceGroups/salai-rg/providers/Microsoft.Web/sites/salai-chat \
  --start-time 2024-01-01T00:00:00Z \
  --interval PT1H \
  --metric "Requests"

# Stream logs
az webapp log tail --resource-group salai-rg --name salai-chat --follow
```

### Vercel

View in dashboard: https://vercel.com/dashboard

## Security Checklist

- [ ] Set strong `SECRET_KEY` and `ADMIN_API_KEY`
- [ ] Enable HTTPS (automatic on both platforms)
- [ ] Set `FLASK_ENV=production`
- [ ] Set `SESSION_COOKIE_SECURE=true` (Azure with HTTPS)
- [ ] Keep dependencies updated: `pip install --upgrade -r requirements.txt`
- [ ] Use environment variables for all secrets
- [ ] Never commit `.env` file
- [ ] Review logs regularly

## Rollback

### Azure

```bash
# View deployment history
az webapp deployment list --resource-group salai-rg --name salai-chat

# Rollback to previous version
az webapp deployment slot swap --resource-group salai-rg --name salai-chat --slot staging
```

### Vercel

```bash
# Rollback in dashboard or via CLI
vercel rollback
```

## Performance Tips

### Azure

1. **Upgrade tier** from F1 to B1 or S1 for better performance
2. **Enable Application Insights** for monitoring
3. **Use CDN** for static assets
4. **Scale horizontally** by increasing instances

### General

1. Keep knowledge base size reasonable
2. Optimize document chunking with `KB_CHUNK_SIZE`
3. Limit concurrent uploads with `MAX_UPLOAD_JOBS`
4. Monitor response times and optimize slow queries

## Getting Help

- **DEPLOYMENT.md** - Detailed deployment guide
- **Azure Docs**: https://docs.microsoft.com/azure/app-service/
- **Vercel Docs**: https://vercel.com/docs
- **Flask Docs**: https://flask.palletsprojects.com/

## Next Steps

1. Choose your platform (Azure App Services recommended)
2. Generate strong secrets
3. Follow the appropriate deployment guide
4. Test all features in your deployment
5. Set up monitoring and logging
6. Configure backups
