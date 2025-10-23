# Development Guide

## Getting Started

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your API keys
# - OPENWEATHERMAP_API_KEY
# - GOOGLE_EARTH_ENGINE_SERVICE_ACCOUNT_KEY
```

### 3. Run Tests

```bash
# Run basic tests
python -m pytest tests/ -v

# Or run with unittest
python -m unittest tests.test_basic -v
```

## Development Workflow

### Phase 1 Implementation Tasks

#### Week 1-2: Foundation
- [ ] Set up Google Earth Engine authentication
- [ ] Test weather API connections
- [ ] Verify data collection pipelines

#### Week 3-4: Processing
- [ ] Implement advanced image processing features
- [ ] Optimize feature engineering pipeline
- [ ] Add data validation and quality checks

#### Week 5-6: Models & API
- [ ] Hyperparameter tuning for baseline models
- [ ] API endpoint testing and optimization
- [ ] Model deployment pipeline

#### Week 7-8: Dashboard & Validation
- [ ] Dashboard UI improvements
- [ ] Real-time data integration
- [ ] Performance optimization

### Running Individual Components

#### Data Collection
```bash
# Collect weather data
python run_pipeline.py --step weather

# Collect satellite data
python run_pipeline.py --step satellite --days-back 7
```

#### Model Training
```bash
# Train all models
python run_pipeline.py --step models

# Or run individual model training
python src/models/baseline_models.py
```

#### API Server
```bash
# Start API server
python src/api/main.py

# Or with uvicorn
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

#### Dashboard
```bash
# Start dashboard
streamlit run src/dashboard/app.py
```

#### Full Pipeline
```bash
# Run complete pipeline
python run_pipeline.py --region default --days-back 7
```

## Project Structure Explained

```
cloud-burst-predictor/
├── src/                    # Source code
│   ├── data/              # Data ingestion modules
│   ├── preprocessing/     # Data processing
│   ├── features/          # Feature engineering
│   ├── models/            # ML models
│   ├── api/               # REST API
│   └── dashboard/         # Streamlit dashboard
├── config/                # Configuration files
├── notebooks/             # Jupyter notebooks
├── tests/                 # Unit tests
├── data/                  # Data storage (gitignored)
├── models/                # Trained models (gitignored)
├── logs/                  # Log files (gitignored)
├── reports/               # Training reports
└── docs/                  # Documentation
```

## API Endpoints

- `GET /health` - Health check
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions
- `GET /features` - Feature information
- `GET /models` - Model information
- `GET /config` - Configuration

## Development Notes

### Adding New Features
1. Create feature extraction function in `src/features/feature_engineering.py`
2. Update configuration in `config/config.yaml`
3. Add tests in `tests/`
4. Update documentation

### Adding New Models
1. Implement model class in `src/models/`
2. Update `baseline_models.py` to include new model
3. Add model-specific configuration
4. Update API to handle new model

### Data Sources
- **Weather APIs**: Free tier limits apply
- **Satellite Data**: Requires Google Earth Engine authentication
- **Mock Data**: Available for development/testing

### Performance Considerations
- Use data caching for repeated API calls
- Implement batch processing for large datasets
- Monitor memory usage with large image datasets
- Use asynchronous processing for real-time predictions

### Debugging
- Check logs in `logs/` directory
- Use `--debug` flag for verbose output
- Verify configuration in `config/config.yaml`
- Test individual components before full pipeline

### Deployment
- Docker containers (add Dockerfile if needed)
- Environment variables for sensitive data
- Health checks for monitoring
- Load balancing for high availability