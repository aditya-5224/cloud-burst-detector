# Production Deployment Guide

## Overview

This guide covers deploying the Cloud Burst Prediction System with all production features:
- **Model Retraining Pipeline**: Automated retraining with version control
- **Data Quality Monitoring**: Real-time validation and anomaly detection
- **Redis Caching**: High-performance API response caching
- **A/B Testing**: Controlled model comparison experiments

## Prerequisites

### Required Software
- Python 3.8+
- Redis Server 6.0+
- PostgreSQL 12+ (optional, for production data storage)

### System Requirements
- **Memory**: Minimum 4GB RAM (8GB+ recommended)
- **Storage**: 10GB+ free space
- **CPU**: 2+ cores recommended

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install and Configure Redis

#### Windows:
Download from: https://github.com/microsoftarchive/redis/releases
```powershell
# Start Redis server
redis-server
```

#### Linux/Mac:
```bash
# Install Redis
sudo apt-get install redis-server  # Ubuntu/Debian
brew install redis  # macOS

# Start Redis
sudo systemctl start redis  # Linux
brew services start redis  # macOS
```

### 3. Configure Environment Variables

Create `.env` file:
```env
# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=  # Leave empty for no password

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Model Configuration
MODEL_RETRAIN_INTERVAL_DAYS=7
MIN_ACCURACY_THRESHOLD=0.75
MIN_SAMPLES_FOR_RETRAINING=100

# Weather API
OPENMETEO_API_KEY=your_key_here
```

## Component Setup

### 1. Model Retraining Pipeline

#### Automatic Scheduled Retraining

```python
# Start retraining scheduler
python -m src.models.retraining_pipeline
```

This will:
- Check for new training data every 7 days
- Train and evaluate new models
- Compare with current production model
- Auto-deploy if performance improves

#### Manual Retraining

```python
from src.models.retraining_pipeline import ModelRetrainingPipeline

pipeline = ModelRetrainingPipeline()
result = pipeline.run_retraining_pipeline(
    model_type='random_forest',
    days_back=30
)
print(result)
```

#### Via API

```bash
curl -X POST "http://localhost:8000/admin/retrain?model_type=random_forest&days_back=30"
```

### 2. Data Quality Monitoring

The data quality middleware automatically:
- Validates all incoming data against schema
- Detects anomalies using statistical methods
- Checks logical consistency
- Tracks quality metrics

#### View Quality Report

```bash
curl "http://localhost:8000/monitoring/data-quality/report"
```

#### Configure Quality Thresholds

Edit `src/data/quality_middleware.py`:
```python
self.anomaly_threshold = 3.0  # Z-score threshold
self.expected_ranges = {
    'temperature_2m': (-40, 50),
    'relative_humidity_2m': (0, 100),
    # ... customize ranges
}
```

### 3. Redis Caching

#### Verify Redis Connection

```python
from src.data.cache_manager import get_cache_manager

cache = get_cache_manager()
stats = cache.get_stats()
print(f"Cache backend: {stats['backend']}")
print(f"Hit rate: {stats['hit_rate']:.2%}")
```

#### Cache Configuration

Default TTL values (in seconds):
- Weather data: 300 (5 minutes)
- Predictions: 600 (10 minutes)
- Satellite images: 1800 (30 minutes)
- Historical data: 3600 (1 hour)

Customize in `src/data/cache_manager.py`:
```python
self.default_ttls = {
    'weather_data': 300,
    'predictions': 600,
    # ... modify as needed
}
```

#### Cache Management

Clear all cache:
```bash
curl -X POST "http://localhost:8000/monitoring/cache/clear"
```

View cache statistics:
```bash
curl "http://localhost:8000/monitoring/cache/stats"
```

### 4. A/B Testing Framework

#### Create Experiment

```python
from src.models.ab_testing import ABTestingFramework, ModelVariant, TrafficSplitStrategy

framework = ABTestingFramework()

variants = [
    ModelVariant(
        variant_id='control',
        model_path='models/trained/random_forest_model.pkl',
        model_type='random_forest',
        version='v1',
        traffic_percentage=50.0
    ),
    ModelVariant(
        variant_id='treatment',
        model_path='models/versions/random_forest_v20250101.pkl',
        model_type='random_forest',
        version='v2',
        traffic_percentage=50.0
    )
]

experiment = framework.create_experiment(
    experiment_id='rf_v1_vs_v2',
    variants=variants,
    strategy=TrafficSplitStrategy.PERCENTAGE
)
```

#### Analyze Results

```python
analysis = framework.analyze_experiment('rf_v1_vs_v2')
print(f"Winner: {analysis['winner']['winner_variant']}")
print(f"Confidence: {analysis['winner']['confidence']}")
```

#### Stop Experiment

```python
framework.stop_experiment('rf_v1_vs_v2')
```

## Running the System

### Development Mode

```bash
# Terminal 1: Start Redis
redis-server

# Terminal 2: Start API server
python src/api/main.py

# Terminal 3: Start dashboard
streamlit run src/dashboard/app.py

# Terminal 4: Start retraining scheduler (optional)
python -m src.models.retraining_pipeline
```

### Production Mode

#### Using Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
  
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_HOST=redis
      - REDIS_PORT=6379
    depends_on:
      - redis
    command: uvicorn src.api.main:app --host 0.0.0.0 --port 8000
  
  dashboard:
    build: .
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8000
    depends_on:
      - api
    command: streamlit run src/dashboard/app.py
  
  retraining:
    build: .
    environment:
      - REDIS_HOST=redis
    depends_on:
      - redis
    command: python -m src.models.retraining_pipeline

volumes:
  redis_data:
```

Start services:
```bash
docker-compose up -d
```

## Monitoring and Maintenance

### Health Checks

```bash
# API health
curl "http://localhost:8000/health"

# Cache status
curl "http://localhost:8000/monitoring/cache/stats"

# Data quality
curl "http://localhost:8000/monitoring/data-quality/report"
```

### View Logs

```bash
# API logs
tail -f logs/api.log

# Retraining logs
tail -f logs/retraining.log

# Data quality logs
tail -f logs/quality.log
```

### Model Version Management

View all model versions:
```bash
curl "http://localhost:8000/admin/model/history"
```

List model files:
```bash
# Production models
ls models/trained/

# All versions
ls models/versions/

# Metrics
ls models/metrics/
```

### Performance Tuning

#### Redis Optimization

Edit `redis.conf`:
```conf
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

#### API Rate Limiting

The system includes built-in rate limiting. Configure in `src/data/quality_middleware.py`:
```python
@rate_limit(max_requests=100, window=60)  # 100 requests per minute
async def your_endpoint():
    ...
```

## Troubleshooting

### Redis Connection Issues

```python
# Test Redis connection
import redis
r = redis.Redis(host='localhost', port=6379)
r.ping()  # Should return True
```

If Redis is not available, the system automatically falls back to in-memory caching.

### Model Retraining Failures

Check logs in `models/metrics/`:
```bash
# View recent retraining attempts
cat models/metrics/random_forest_v*_metrics.json
```

Common issues:
- Insufficient training data (< 100 samples)
- Model accuracy below threshold (< 0.75)
- Feature mismatch

### Cache Performance

Monitor cache hit rate:
```python
from src.data.cache_manager import get_cache_manager

cache = get_cache_manager()
stats = cache.get_stats()

# Hit rate should be > 50% for good performance
print(f"Hit rate: {stats['hit_rate']:.2%}")
```

## Security Recommendations

### Production Deployment

1. **Enable Redis Authentication**:
   ```bash
   # redis.conf
   requirepass your_strong_password
   ```

2. **Use HTTPS** for API:
   ```python
   # Use nginx or traefik as reverse proxy
   # with SSL/TLS certificates
   ```

3. **API Key Authentication**:
   Add API key middleware to FastAPI

4. **Rate Limiting**:
   Already included in quality middleware

5. **Input Sanitization**:
   Pydantic models handle validation

## Backup and Recovery

### Backup Redis Data

```bash
# Create backup
redis-cli BGSAVE

# Backup file location
cp /var/lib/redis/dump.rdb ./backups/redis_backup_$(date +%Y%m%d).rdb
```

### Backup Models

```bash
# Backup all models
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/
```

### Restore from Backup

```bash
# Restore Redis
sudo systemctl stop redis
cp backup/dump.rdb /var/lib/redis/
sudo systemctl start redis

# Restore models
tar -xzf models_backup_20250101.tar.gz
```

## Performance Metrics

Expected performance with optimizations:

| Metric | Without Cache | With Cache |
|--------|--------------|------------|
| Weather API Response | 500-1000ms | 10-50ms |
| Prediction Response | 100-200ms | 20-50ms |
| Data Quality Check | 50ms | 50ms |
| Total Request Time | 650-1250ms | 80-150ms |

## Support and Maintenance

For issues or questions:
1. Check logs in `logs/` directory
2. Review monitoring endpoints
3. Check Redis status: `redis-cli ping`
4. Verify model files in `models/trained/`

## Next Steps

1. ✅ Set up production database (PostgreSQL)
2. ✅ Configure automated backups
3. ✅ Set up monitoring dashboard (Grafana)
4. ✅ Implement user authentication
5. ✅ Deploy to cloud (AWS/GCP/Azure)
