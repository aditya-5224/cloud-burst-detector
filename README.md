# Cloud Burst Prediction System - Phase 1

A machine learning system for predicting cloud burst events using meteorological data and satellite imagery analysis.

## Project Overview

This MVP system ingests meteorological and satellite cloud imagery data, performs feature extraction, and runs baseline ML models to predict cloud burst events with 70-80% accuracy, exposed via a web dashboard and API.

## Features

- **Data Ingestion**: Automated collection from weather APIs (Open-Meteo, WeatherAPI.com) and satellite imagery (Google Earth Engine)
- **Live Weather Integration**: Real-time weather data with location name resolution and multi-source fallback
- **Historical Data**: Database of 10+ known cloud burst events (2016-2023) with validation system
- **Image Processing**: Cloud detection and feature extraction from satellite imagery using OpenCV
- **Feature Engineering**: Meteorological feature computation including CAPE and Lifted Index
- **ML Models**: Baseline models including Random Forest, SVM, and LSTM
- **Model Validation**: Historical validation against real disasters with accuracy metrics
- **Web Dashboard**: Interactive Streamlit dashboard with live predictions, historical analysis, and maps
- **REST API**: FastAPI endpoints for predictions and feature access

## Tech Stack

- **Language**: Python 3.8+
- **ML Frameworks**: scikit-learn, TensorFlow
- **Web Frameworks**: FastAPI, Streamlit
- **Image Processing**: OpenCV
- **Data Processing**: pandas, numpy
- **Deployment**: Google Colab, Hugging Face Spaces

## Project Structure

```
cloud-burst-predictor/
├── src/
│   ├── data/           # Data ingestion and storage
│   ├── preprocessing/  # Data cleaning and transformation
│   ├── features/       # Feature engineering
│   ├── models/         # ML model implementations
│   ├── api/           # FastAPI endpoints
│   └── dashboard/     # Streamlit dashboard
├── config/            # Configuration files
├── notebooks/         # Jupyter notebooks
├── tests/            # Unit tests
├── data/             # Data storage
├── models/           # Trained model artifacts
└── docs/             # Documentation
```

## Quick Start

1. **Environment Setup**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configuration**:
   ```bash
   cp config/config.example.yaml config/config.yaml
   # Edit config.yaml with your API keys
   ```

3. **Run Data Pipeline**:
   ```bash
   python src/data/weather_api.py
   python src/data/satellite_imagery.py
   ```

4. **Train Models**:
   ```bash
   python src/models/train_baseline.py
   ```

5. **Launch Dashboard**:
   ```bash
   streamlit run src/dashboard/app.py
   ```

6. **Start API Server**:
   ```bash
   uvicorn src.api.main:app --reload
   ```

## Historical Data Integration

**New Feature:** Validate your model against real cloud burst disasters!

### Quick Start with Historical Data:

1. **Test the Integration**:
   ```bash
   python test_historical_integration.py
   ```

2. **Build Historical Dataset** (10 known events, 2016-2023):
   ```bash
   python src/data/historical_weather.py
   ```

3. **Validate Model Performance**:
   ```bash
   python src/models/historical_validation.py
   ```

4. **View Results in Dashboard**:
   - Start dashboard: `streamlit run src/dashboard/app.py`
   - Click "📊 Historical Analysis" in sidebar
   - Explore events, validation metrics, and patterns

### Features:
- ✅ Database of 10+ documented cloud burst events
- ✅ Automated historical weather data collection
- ✅ Model validation with accuracy metrics
- ✅ Pattern analysis (cloud burst vs normal conditions)
- ✅ Interactive dashboard with maps and visualizations
- ✅ Custom date range analysis for any location

### Documentation:
- **Full Guide**: See `HISTORICAL_DATA_GUIDE.md` (15,000+ words)
- **Quick Summary**: See `HISTORICAL_DATA_SUMMARY.md`
- **Test Script**: Run `test_historical_integration.py`

### Expected Results:
- Model Accuracy: 70-80% on real events
- Warning Time: 2-3 hours before cloud burst
- Validation against disasters from Uttarakhand, Himachal Pradesh, J&K

## Phase 1 Goals

- ✅ Ingest and standardize free weather API and satellite image data
- ✅ Implement image-based cloud detection and integrate numeric weather features
- ✅ Train baseline ML models (Random Forest, SVM, LSTM)
- ✅ Expose predictions via a basic dashboard and REST API
- ✅ Achieve ≥70% event-level prediction accuracy on historical data

## Acceptance Criteria

- API success rate: ≥95% hourly pulls
- Cloud mask accuracy: ≥85% manual match
- Model F1-score: ≥70%
- Dashboard features: Interactive map and real-time charts
- API uptime: 99% under light load

## Data Sources

- **Weather APIs**: Open-Meteo, OpenWeatherMap
- **Satellite APIs**: Google Earth Engine (Sentinel-2)

## Development Phases

### Week 1-2: Foundation
- Environment setup and weather API connectors
- Earth Engine integration and image ingestion

### Week 3-4: Processing
- Image processing and feature extraction pipeline
- Numeric feature engineering and index calculation

### Week 5-6: Models & API
- Baseline model training and metrics report
- REST API implementation and model deployment

### Week 7-8: Dashboard & Validation
- Streamlit dashboard with map overlay
- Validation on test set and tuning

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions and support, please open an issue in the repository.