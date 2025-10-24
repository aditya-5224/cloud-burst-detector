from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import pandas as pd
from datetime import datetime
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.api.prediction_service import get_prediction_service
from src.data.live_weather import live_weather_collector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Cloud Burst Prediction API", version="1.0.0", docs_url="/docs")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

prediction_service = get_prediction_service()

class PredictionRequest(BaseModel):
    features: Dict[str, float]
    model: Optional[str] = 'random_forest'

class PredictionResponse(BaseModel):
    success: bool
    prediction: Optional[int] = None
    probability: Optional[float] = None
    risk_level: Optional[str] = None
    model: Optional[str] = None
    timestamp: Optional[str] = None
    error: Optional[str] = None

class WeatherRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate")
    force_refresh: Optional[bool] = Field(False, description="Skip cache and fetch fresh data")

class LivePredictionRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate")
    model: Optional[str] = Field('random_forest', description="Model to use for prediction")

@app.get("/")
async def root():
    return {"name": "Cloud Burst Prediction API", "version": "1.0.0", "status": "operational"}

@app.get("/health")
async def health():
    info = prediction_service.get_model_info()
    return {"status": "healthy" if info['model_ready'] else "degraded", "timestamp": datetime.now().isoformat(), "model_loaded": info['model_ready'], "models_available": info['models_loaded']}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        features_df = pd.DataFrame([request.features])
        result = prediction_service.predict(features_df, model_name=request.model)
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def model_info():
    info = prediction_service.get_model_info()
    importance = prediction_service.get_feature_importance(top_n=20)
    info['top_features'] = importance
    return info

@app.get("/model/features")
async def get_features():
    if not prediction_service.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"feature_count": len(prediction_service.feature_names), "features": prediction_service.feature_names}

# ============================================================================
# Live Weather Endpoints
# ============================================================================

@app.post("/weather/live")
async def get_live_weather(request: WeatherRequest):
    """Get current weather data for a location"""
    try:
        weather = live_weather_collector.get_live_weather(
            request.latitude, 
            request.longitude,
            force_refresh=request.force_refresh
        )
        
        if not weather:
            raise HTTPException(status_code=503, detail="Weather data unavailable from all sources")
        
        return {
            "success": True,
            "weather": weather,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Weather fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/live")
async def predict_live(request: LivePredictionRequest):
    """Make prediction using live weather data for a location"""
    try:
        # Fetch live weather
        weather = live_weather_collector.get_weather_for_prediction(
            request.latitude,
            request.longitude
        )
        
        if not weather:
            raise HTTPException(status_code=503, detail="Unable to fetch weather data")
        
        # Prepare features for prediction
        # Get the expected feature names from the model
        expected_features = prediction_service.feature_names
        
        # Create a dictionary with all expected features initialized to 0
        features = {feature: 0.0 for feature in expected_features}
        
        # Map the available weather data to the expected features
        feature_mapping = {
            'temperature_2m': weather['temperature'],
            'relative_humidity_2m': weather['humidity'],
            'pressure_msl': weather['pressure'],
            'precipitation': weather['precipitation'],
            'wind_speed_10m': weather['wind_speed'] / 3.6,  # Convert km/h to m/s
            'wind_direction_10m': weather.get('wind_direction', 0),
            'cloud_cover_total': weather['cloud_cover'],
        }
        
        # Estimate atmospheric stability indices from basic weather data
        # These are simplified approximations - ideally would calculate from upper air data
        temp_c = weather['temperature']
        humidity = weather['humidity']
        pressure = weather['pressure']
        
        # CAPE approximation (Convective Available Potential Energy)
        # Higher temps + humidity can indicate potential CAPE
        cape_estimate = max(0, (temp_c - 15) * (humidity / 100) * 100)
        
        # Lifted Index approximation (negative = unstable)
        # Simplified: lower pressure + higher temp = more unstable
        lifted_index = (pressure - 1000) / 10 - (temp_c - 20) / 5
        
        # K-Index (measure of thunderstorm potential)
        # Higher humidity + moderate temps = higher K
        k_index = (temp_c - 15) + (humidity - 40) / 10
        
        # Total Totals Index
        total_totals = temp_c + humidity / 5 - 40
        
        # Showalter Index (stability indicator)
        showalter_index = lifted_index * 0.8
        
        # Update features with weather data
        for feature_name, value in feature_mapping.items():
            if feature_name in features:
                features[feature_name] = value
        
        # Add atmospheric indices (handling duplicates properly)
        # Note: Model has 'cape' listed twice, so we need to set both
        cape_value = cape_estimate
        lifted_index_value = lifted_index
        k_index_value = k_index
        total_totals_value = total_totals
        showalter_index_value = showalter_index
        
        # Create features dataframe with correct column order matching model
        feature_values = []
        for feature_name in expected_features:
            if feature_name == 'cape':
                feature_values.append(cape_value)
            elif feature_name == 'lifted_index':
                feature_values.append(lifted_index_value)
            elif feature_name == 'k_index':
                feature_values.append(k_index_value)
            elif feature_name == 'total_totals':
                feature_values.append(total_totals_value)
            elif feature_name == 'showalter_index':
                feature_values.append(showalter_index_value)
            elif feature_name in features:
                feature_values.append(features[feature_name])
            else:
                feature_values.append(0.0)
        
        # Make prediction - ensure column names are in exact order
        # Use the EXACT same feature list from the model - don't convert/copy
        features_df = pd.DataFrame([feature_values], columns=expected_features)
        
        # Log for debugging
        logger.info(f"Created features DataFrame with shape: {features_df.shape}")
        logger.info(f"Feature columns: {list(features_df.columns)}")
        logger.info(f"Expected features: {expected_features}")
        logger.info(f"Match: {list(features_df.columns) == list(expected_features)}")
        
        result = prediction_service.predict(features_df, model_name=request.model)
        
        # Add weather data to response
        result['weather_data'] = weather
        result['location'] = weather['location']
        
        return result
        
    except Exception as e:
        logger.error(f"Live prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/weather/cache/stats")
async def get_cache_stats():
    """Get weather cache statistics"""
    try:
        stats = live_weather_collector.get_cache_stats()
        return {
            "success": True,
            "cache_stats": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Cache stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/weather/cache/clear")
async def clear_weather_cache():
    """Clear the weather cache"""
    try:
        live_weather_collector.clear_cache()
        return {
            "success": True,
            "message": "Weather cache cleared",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup():
    logger.info("="*80)
    logger.info("Cloud Burst Prediction API Starting...")
    info = prediction_service.get_model_info()
    logger.info(f"Model ready: {info['model_ready']}")
    logger.info("="*80)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
