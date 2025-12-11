# Backend Deployment Guide

## Deployment Overview
- **Application**: [Application Name]
- **Technology Stack**: FastAPI + PostgreSQL + Redis
- **Deployment Method**: Docker Containers
- **Orchestration**: Docker Compose / Kubernetes
- **Cloud Provider**: AWS / GCP / Azure

## Prerequisites

### System Requirements
- Docker 20.10+
- Docker Compose 2.0+
- Python 3.10+
- PostgreSQL 14+
- Redis 6.2+
- Nginx 1.20+

### Environment Setup
```bash
# Clone repository
git clone https://github.com/organization/backend-service.git
cd backend-service

# Create environment file
cp .env.example .env

# Install dependencies
pip install -r requirements.txt
```

## Environment Configuration

### Environment Variables
```env
# Application
APP_NAME=backend-service
APP_ENV=production
APP_PORT=8000
APP_HOST=0.0.0.0
DEBUG=false

# Database
DATABASE_URL=postgresql://user:password@db:5432/dbname
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=40

# Redis
REDIS_URL=redis://redis:6379/0
REDIS_MAX_CONNECTIONS=50

# Security
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=your-jwt-secret
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# External Services
AWS_ACCESS_KEY_ID=xxx
AWS_SECRET_ACCESS_KEY=xxx
AWS_REGION=us-east-1
S3_BUCKET_NAME=app-uploads

# Monitoring
SENTRY_DSN=https://xxx@sentry.io/xxx
LOG_LEVEL=INFO
```

## Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose
```yaml
version: '3.8'

services:
  app:
    build: .
    container_name: backend-app
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/appdb
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    volumes:
      - ./logs:/app/logs
    networks:
      - app-network
    restart: unless-stopped

  db:
    image: postgres:14-alpine
    container_name: backend-db
    environment:
      POSTGRES_DB: appdb
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - postgres-data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    networks:
      - app-network
    restart: unless-stopped

  redis:
    image: redis:6.2-alpine
    container_name: backend-redis
    command: redis-server --appendonly yes
    volumes:
      - redis-data:/data
    ports:
      - "6379:6379"
    networks:
      - app-network
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    container_name: backend-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - app
    networks:
      - app-network
    restart: unless-stopped

volumes:
  postgres-data:
  redis-data:

networks:
  app-network:
    driver: bridge
```

## Kubernetes Deployment

### Deployment Manifest
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend-api
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: backend-api
  template:
    metadata:
      labels:
        app: backend-api
    spec:
      containers:
      - name: api
        image: registry.example.com/backend-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: backend-secrets
              key: database-url
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Service Configuration
```yaml
apiVersion: v1
kind: Service
metadata:
  name: backend-api-service
  namespace: production
spec:
  selector:
    app: backend-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Database Migration

### Initial Setup
```bash
# Create database
psql -U postgres -c "CREATE DATABASE appdb;"

# Run migrations
alembic upgrade head

# Seed initial data (optional)
python scripts/seed_database.py
```

### Migration Commands
```bash
# Create new migration
alembic revision --autogenerate -m "Add user table"

# Apply migrations
alembic upgrade head

# Rollback one version
alembic downgrade -1

# Check current version
alembic current
```

## Nginx Configuration

### nginx.conf
```nginx
upstream backend {
    server app:8000;
}

server {
    listen 80;
    server_name api.example.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.example.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;

    client_max_body_size 10M;

    location / {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    location /static {
        alias /app/static;
        expires 30d;
    }
}
```

## Deployment Process

### Production Deployment
```bash
#!/bin/bash
# deploy.sh

# 1. Build and tag image
docker build -t backend-api:$VERSION .
docker tag backend-api:$VERSION registry.example.com/backend-api:$VERSION
docker tag backend-api:$VERSION registry.example.com/backend-api:latest

# 2. Push to registry
docker push registry.example.com/backend-api:$VERSION
docker push registry.example.com/backend-api:latest

# 3. Run database migrations
kubectl exec -it deploy/backend-api -- alembic upgrade head

# 4. Update deployment
kubectl set image deployment/backend-api api=registry.example.com/backend-api:$VERSION

# 5. Wait for rollout
kubectl rollout status deployment/backend-api

# 6. Health check
curl https://api.example.com/health
```

### Rollback Procedure
```bash
# Rollback deployment
kubectl rollout undo deployment/backend-api

# Rollback database (if needed)
kubectl exec -it deploy/backend-api -- alembic downgrade -1
```

## Monitoring Setup

### Health Check Endpoints
```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": settings.APP_VERSION
    }

@app.get("/ready")
async def readiness_check(db: Session = Depends(get_db)):
    try:
        # Check database connection
        db.execute("SELECT 1")
        # Check Redis connection
        redis_client.ping()
        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(status_code=503, detail="Service not ready")
```

### Logging Configuration
```python
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            'logs/app.log',
            maxBytes=10485760,  # 10MB
            backupCount=10
        ),
        logging.StreamHandler()
    ]
)
```

### Metrics Collection
```python
from prometheus_client import Counter, Histogram, generate_latest

# Define metrics
request_count = Counter('app_requests_total', 'Total requests')
request_duration = Histogram('app_request_duration_seconds', 'Request duration')

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain")
```

## Security Hardening

### SSL/TLS Configuration
```bash
# Generate self-signed certificate (development)
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Production: Use Let's Encrypt
certbot certonly --webroot -w /var/www/html -d api.example.com
```

### Security Headers
```python
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://app.example.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000"
    return response
```

## Backup and Recovery

### Database Backup
```bash
# Backup script
#!/bin/bash
BACKUP_DIR="/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DB_NAME="appdb"

# Create backup
pg_dump -h localhost -U postgres -d $DB_NAME > $BACKUP_DIR/backup_$TIMESTAMP.sql

# Compress backup
gzip $BACKUP_DIR/backup_$TIMESTAMP.sql

# Upload to S3
aws s3 cp $BACKUP_DIR/backup_$TIMESTAMP.sql.gz s3://backups-bucket/

# Clean old backups (keep last 30 days)
find $BACKUP_DIR -name "backup_*.sql.gz" -mtime +30 -delete
```

### Restore Procedure
```bash
# Download backup from S3
aws s3 cp s3://backups-bucket/backup_20240101_120000.sql.gz .

# Decompress
gunzip backup_20240101_120000.sql.gz

# Restore database
psql -h localhost -U postgres -d appdb < backup_20240101_120000.sql
```

## Troubleshooting

### Common Issues

#### Application Won't Start
```bash
# Check logs
docker logs backend-app

# Check environment variables
docker exec backend-app env

# Test database connection
docker exec backend-app python -c "from src.core.database import engine; engine.connect()"
```

#### Database Connection Issues
```bash
# Test connection
psql -h localhost -U postgres -d appdb

# Check PostgreSQL logs
docker logs backend-db

# Verify network
docker network inspect app-network
```

#### High Memory Usage
```bash
# Check memory usage
docker stats backend-app

# Analyze Python memory
docker exec backend-app python -m memory_profiler src/main.py
```

## Performance Tuning

### Application Optimization
```python
# Connection pooling
engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True,
    pool_recycle=3600
)

# Response caching
from fastapi_cache import FastAPICache
from fastapi_cache.backend.redis import RedisBackend

@app.on_event("startup")
async def startup():
    redis = aioredis.from_url("redis://localhost")
    FastAPICache.init(RedisBackend(redis), prefix="cache")
```

### Database Optimization
```sql
-- Analyze query performance
EXPLAIN ANALYZE SELECT * FROM users WHERE email = 'user@example.com';

-- Update statistics
ANALYZE users;

-- Vacuum tables
VACUUM ANALYZE;
```

## Maintenance

### Scheduled Tasks
```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler

scheduler = AsyncIOScheduler()

@scheduler.scheduled_job('cron', hour=2)
async def daily_cleanup():
    """Clean up old sessions and logs"""
    await cleanup_expired_sessions()
    await rotate_logs()

scheduler.start()
```

### Update Procedure
1. Announce maintenance window
2. Backup database
3. Deploy new version to staging
4. Run tests on staging
5. Deploy to production
6. Monitor for issues
7. Rollback if necessary

## Documentation
- API Documentation: https://api.example.com/docs
- Monitoring Dashboard: https://monitoring.example.com
- Log Analysis: https://logs.example.com