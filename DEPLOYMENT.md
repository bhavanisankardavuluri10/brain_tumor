# Deployment Guide

## Production Deployment Options

This guide covers deploying your Brain Tumor Detection System to production.

---

## 🌐 Option 1: Docker Deployment (Recommended)

### Create Dockerfile for Backend

```dockerfile
FROM python:3.8-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create necessary directories
RUN mkdir -p models uploads logs results

EXPOSE 5000

CMD ["python", "app.py"]
```

### Create docker-compose.yml

```yaml
version: "3.8"

services:
  backend:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./models:/app/models
      - ./uploads:/app/uploads
    environment:
      - FLASK_ENV=production
    restart: unless-stopped

  frontend:
    build: ./frontend
    ports:
      - "80:80"
    depends_on:
      - backend
    restart: unless-stopped
```

### Deploy with Docker

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## ☁️ Option 2: Cloud Deployment

### AWS Deployment

#### Using AWS Elastic Beanstalk:

1. Install AWS CLI and EB CLI:

```bash
pip install awscli awsebcli
```

2. Initialize EB:

```bash
eb init -p python-3.8 brain-tumor-detection
```

3. Create environment:

```bash
eb create production-env
```

4. Deploy:

```bash
eb deploy
```

#### Using AWS EC2:

1. Launch EC2 instance (t2.large or larger)
2. SSH into instance
3. Install dependencies:

```bash
sudo apt-get update
sudo apt-get install python3-pip nginx
```

4. Clone and setup:

```bash
git clone <your-repo>
cd <project-dir>
pip3 install -r requirements.txt
```

5. Setup Nginx as reverse proxy
6. Use PM2 or systemd for process management

### Google Cloud Platform

#### Using Cloud Run:

1. Build container:

```bash
gcloud builds submit --tag gcr.io/PROJECT-ID/brain-tumor-detection
```

2. Deploy:

```bash
gcloud run deploy --image gcr.io/PROJECT-ID/brain-tumor-detection --platform managed
```

### Azure

#### Using Azure App Service:

1. Create App Service
2. Configure Python runtime
3. Deploy via Git or ZIP:

```bash
az webapp up --name brain-tumor-app --resource-group myResourceGroup
```

---

## 🚀 Option 3: Heroku Deployment

### Setup Heroku

1. Create `Procfile`:

```
web: gunicorn app:app
```

2. Create `runtime.txt`:

```
python-3.8.10
```

3. Update requirements.txt:

```bash
echo "gunicorn==20.1.0" >> requirements.txt
```

4. Deploy:

```bash
heroku create brain-tumor-detection
git push heroku main
```

---

## 🔧 Production Configuration

### Update config.yaml for Production

```yaml
api:
  host: "0.0.0.0"
  port: 5000
  debug: false # Important!

# Add production settings
production:
  max_file_size: 10485760 # 10MB
  rate_limiting: true
  https_only: true
```

### Security Best Practices

1. **Use HTTPS**:

   - Get SSL certificate (Let's Encrypt)
   - Configure reverse proxy (Nginx/Apache)

2. **Environment Variables**:

```python
import os
SECRET_KEY = os.environ.get('SECRET_KEY')
API_KEY = os.environ.get('API_KEY')
```

3. **Rate Limiting**:

```python
from flask_limiter import Limiter
limiter = Limiter(app, key_func=get_remote_address)

@app.route('/api/predict')
@limiter.limit("10 per minute")
def predict():
    ...
```

4. **Input Validation**:

   - Already implemented in app.py
   - File size limits
   - File type validation

5. **CORS Configuration**:

```python
CORS(app, resources={
    r"/api/*": {
        "origins": ["https://yourdomain.com"]
    }
})
```

---

## 📊 Performance Optimization

### 1. Model Optimization

Convert to ONNX for faster inference:

```python
import tf2onnx

# Convert model
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
output_path = "models/model.onnx"

model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=13,
    output_path=output_path
)
```

### 2. Caching

```python
from flask_caching import Cache

cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@app.route('/api/model/info')
@cache.cached(timeout=300)
def get_model_info():
    ...
```

### 3. Load Balancing

Use Nginx for load balancing:

```nginx
upstream backend {
    server 127.0.0.1:5000;
    server 127.0.0.1:5001;
    server 127.0.0.1:5002;
}

server {
    listen 80;

    location /api {
        proxy_pass http://backend;
    }
}
```

### 4. Database for Statistics

Add PostgreSQL for persistent statistics:

```python
from flask_sqlalchemy import SQLAlchemy

app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
db = SQLAlchemy(app)
```

---

## 🔍 Monitoring & Logging

### 1. Setup Logging

```python
import logging
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)
```

### 2. Error Tracking

Use Sentry:

```python
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration

sentry_sdk.init(
    dsn="your-sentry-dsn",
    integrations=[FlaskIntegration()]
)
```

### 3. Performance Monitoring

Use New Relic or DataDog:

```bash
pip install newrelic
newrelic-admin run-program gunicorn app:app
```

---

## 🧪 Pre-Deployment Checklist

- [ ] Model trained and tested
- [ ] Model file < 100MB (or use model splitting)
- [ ] All tests passing
- [ ] Debug mode disabled
- [ ] HTTPS configured
- [ ] Environment variables set
- [ ] CORS properly configured
- [ ] Rate limiting enabled
- [ ] Logging configured
- [ ] Error tracking setup
- [ ] Database configured (if needed)
- [ ] Backup strategy in place
- [ ] Monitoring setup
- [ ] Load testing completed
- [ ] Documentation updated

---

## 📱 Frontend Production Build

### Build React App

```bash
cd frontend
npm run build
```

### Serve with Nginx

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    root /path/to/frontend/build;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api {
        proxy_pass http://localhost:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

---

## 🔐 Environment Variables

Create `.env` file:

```bash
# Flask
FLASK_ENV=production
SECRET_KEY=your-secret-key

# API
API_PORT=5000
API_HOST=0.0.0.0

# Model
MODEL_PATH=/app/models/best_model.h5

# Database (optional)
DATABASE_URL=postgresql://user:pass@localhost/dbname

# Monitoring
SENTRY_DSN=your-sentry-dsn
NEW_RELIC_LICENSE_KEY=your-key
```

Load in Python:

```python
from dotenv import load_dotenv
load_dotenv()

import os
SECRET_KEY = os.getenv('SECRET_KEY')
```

---

## 🚦 Health Checks

Implement comprehensive health checks:

```python
@app.route('/health')
def health_check():
    health = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'checks': {
            'model': model is not None,
            'disk_space': get_disk_usage() < 90,
            'memory': get_memory_usage() < 90,
        }
    }

    status_code = 200 if all(health['checks'].values()) else 503
    return jsonify(health), status_code
```

---

## 📊 Load Testing

Use Apache Bench or Locust:

```bash
# Apache Bench
ab -n 1000 -c 10 http://localhost:5000/api/health

# Locust
pip install locust
locust -f loadtest.py
```

`loadtest.py`:

```python
from locust import HttpUser, task

class BrainTumorUser(HttpUser):
    @task
    def predict(self):
        files = {'file': open('test_image.jpg', 'rb')}
        self.client.post('/api/predict', files=files)
```

---

## 🎯 Scaling Strategies

### Horizontal Scaling

- Multiple API instances behind load balancer
- Stateless design (no local session storage)
- Shared model storage (S3, NFS)

### Vertical Scaling

- Larger instance types
- More CPU/RAM
- GPU instances for inference

### Auto-scaling

- Configure based on CPU/memory
- Queue-based scaling for batch jobs
- Kubernetes for orchestration

---

## 🔄 Continuous Deployment

### GitHub Actions

`.github/workflows/deploy.yml`:

```yaml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v2

      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.8

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run tests
        run: pytest

      - name: Deploy to production
        run: |
          # Your deployment commands
```

---

## 📞 Support & Maintenance

### Backup Strategy

- Daily model backups
- Database backups (if applicable)
- Configuration backups
- Automated backup verification

### Update Strategy

- Zero-downtime deployments
- Rolling updates
- Canary deployments
- Rollback procedures

### Monitoring

- Uptime monitoring
- Performance metrics
- Error rates
- User analytics

---

## 🎉 Conclusion

Your Brain Tumor Detection System is now production-ready! Choose the deployment option that best fits your needs and follow the security and optimization best practices.

For questions or issues, refer to the main README.md or open an issue.
