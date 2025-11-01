"""
Data Quality and Validation Middleware

Provides comprehensive data validation, anomaly detection, and quality metrics.
"""
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pydantic import BaseModel, validator, Field
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict, deque
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic Models for Validation
class WeatherDataInput(BaseModel):
    """Validated weather data input schema"""
    
    temperature_2m: float = Field(..., ge=-50, le=60, description="Temperature in Celsius")
    relative_humidity_2m: float = Field(..., ge=0, le=100, description="Relative humidity percentage")
    pressure_msl: float = Field(..., ge=870, le=1085, description="Mean sea level pressure in hPa")
    wind_speed_10m: float = Field(..., ge=0, le=100, description="Wind speed in m/s")
    wind_direction_10m: float = Field(..., ge=0, le=360, description="Wind direction in degrees")
    cloud_cover: float = Field(..., ge=0, le=100, description="Cloud cover percentage")
    precipitation: float = Field(..., ge=0, le=500, description="Precipitation in mm")
    
    # Optional advanced features
    dewpoint_2m: Optional[float] = Field(None, ge=-60, le=40)
    surface_pressure: Optional[float] = Field(None, ge=800, le=1100)
    
    @validator('*', pre=True)
    def check_not_nan(cls, v):
        """Ensure no NaN values"""
        if isinstance(v, (float, int)) and (np.isnan(v) or np.isinf(v)):
            raise ValueError('Value cannot be NaN or Inf')
        return v
    
    class Config:
        extra = 'allow'  # Allow additional fields


class LocationInput(BaseModel):
    """Validated location input"""
    
    latitude: float = Field(..., ge=-90, le=90, description="Latitude in degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in degrees")
    
    @validator('latitude', 'longitude')
    def check_valid_coordinates(cls, v):
        """Ensure valid coordinate values"""
        if np.isnan(v) or np.isinf(v):
            raise ValueError('Coordinate cannot be NaN or Inf')
        return v


class DataQualityMetrics(BaseModel):
    """Data quality metrics"""
    
    completeness: float = Field(..., ge=0, le=1)
    accuracy: float = Field(..., ge=0, le=1)
    consistency: float = Field(..., ge=0, le=1)
    timeliness: float = Field(..., ge=0, le=1)
    anomaly_score: float = Field(..., ge=0, le=1)
    overall_quality: float = Field(..., ge=0, le=1)
    
    timestamp: datetime = Field(default_factory=datetime.now)
    details: Dict[str, Any] = Field(default_factory=dict)


class DataQualityMiddleware:
    """Middleware for data validation and quality monitoring"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Historical data for anomaly detection
        self.history_window = 1000
        self.data_history = defaultdict(lambda: deque(maxlen=self.history_window))
        
        # Quality metrics storage
        self.metrics_dir = Path("data/quality_metrics")
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Anomaly thresholds (z-score)
        self.anomaly_threshold = 3.0
        
        # Define expected ranges for weather variables
        self.expected_ranges = {
            'temperature_2m': (-40, 50),
            'relative_humidity_2m': (0, 100),
            'pressure_msl': (900, 1050),
            'wind_speed_10m': (0, 50),
            'wind_direction_10m': (0, 360),
            'cloud_cover': (0, 100),
            'precipitation': (0, 300)
        }
        
        # Quality metrics cache
        self.quality_cache = deque(maxlen=100)
    
    def validate_input(self, data: Dict) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        Validate input data against schema
        
        Args:
            data: Input data dictionary
            
        Returns:
            Tuple of (is_valid, error_message, validated_data)
        """
        try:
            # Validate weather data
            validated = WeatherDataInput(**data)
            return True, None, validated.dict()
        
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False, str(e), None
    
    def detect_anomalies(self, data: Dict) -> Tuple[bool, List[str], Dict[str, float]]:
        """
        Detect anomalies in input data using statistical methods
        
        Args:
            data: Input data dictionary
            
        Returns:
            Tuple of (has_anomalies, anomaly_descriptions, anomaly_scores)
        """
        anomalies = []
        scores = {}
        
        for field, value in data.items():
            if not isinstance(value, (int, float)):
                continue
            
            # Check against expected ranges
            if field in self.expected_ranges:
                min_val, max_val = self.expected_ranges[field]
                if value < min_val or value > max_val:
                    anomalies.append(f"{field} value {value} outside expected range [{min_val}, {max_val}]")
                    scores[field] = 1.0
                    continue
            
            # Statistical anomaly detection using historical data
            if field in self.data_history and len(self.data_history[field]) >= 30:
                history = list(self.data_history[field])
                mean = np.mean(history)
                std = np.std(history)
                
                if std > 0:
                    z_score = abs((value - mean) / std)
                    scores[field] = min(z_score / self.anomaly_threshold, 1.0)
                    
                    if z_score > self.anomaly_threshold:
                        anomalies.append(
                            f"{field} value {value:.2f} is {z_score:.2f} std devs from mean {mean:.2f}"
                        )
            
            # Update history
            self.data_history[field].append(value)
        
        has_anomalies = len(anomalies) > 0
        
        return has_anomalies, anomalies, scores
    
    def check_data_consistency(self, data: Dict) -> Tuple[bool, List[str]]:
        """
        Check for logical consistency in weather data
        
        Args:
            data: Input data dictionary
            
        Returns:
            Tuple of (is_consistent, inconsistency_messages)
        """
        issues = []
        
        # Check dewpoint vs temperature
        if 'dewpoint_2m' in data and 'temperature_2m' in data:
            if data['dewpoint_2m'] > data['temperature_2m']:
                issues.append("Dewpoint cannot exceed temperature")
        
        # Check humidity vs precipitation
        if 'relative_humidity_2m' in data and 'precipitation' in data:
            if data['precipitation'] > 0 and data['relative_humidity_2m'] < 60:
                issues.append("High precipitation with low humidity is unusual")
        
        # Check cloud cover vs precipitation
        if 'cloud_cover' in data and 'precipitation' in data:
            if data['precipitation'] > 10 and data['cloud_cover'] < 50:
                issues.append("Heavy precipitation usually requires high cloud cover")
        
        # Check wind speed consistency
        if 'wind_speed_10m' in data:
            if data['wind_speed_10m'] > 30:
                # Extreme wind speed - check if other conditions support it
                if data.get('pressure_msl', 1013) > 1020:
                    issues.append("High wind speed with high pressure is unusual")
        
        is_consistent = len(issues) == 0
        
        return is_consistent, issues
    
    def calculate_completeness(self, data: Dict, required_fields: List[str]) -> float:
        """
        Calculate data completeness score
        
        Args:
            data: Input data dictionary
            required_fields: List of required field names
            
        Returns:
            Completeness score (0-1)
        """
        if not required_fields:
            return 1.0
        
        present_fields = sum(1 for field in required_fields if field in data and data[field] is not None)
        return present_fields / len(required_fields)
    
    def calculate_quality_metrics(self, data: Dict) -> DataQualityMetrics:
        """
        Calculate comprehensive data quality metrics
        
        Args:
            data: Input data dictionary
            
        Returns:
            DataQualityMetrics object
        """
        # Required fields for weather prediction
        required_fields = [
            'temperature_2m', 'relative_humidity_2m', 'pressure_msl',
            'wind_speed_10m', 'wind_direction_10m', 'cloud_cover', 'precipitation'
        ]
        
        # 1. Completeness
        completeness = self.calculate_completeness(data, required_fields)
        
        # 2. Accuracy (based on validation)
        is_valid, _, _ = self.validate_input(data)
        accuracy = 1.0 if is_valid else 0.5
        
        # 3. Consistency
        is_consistent, consistency_issues = self.check_data_consistency(data)
        consistency = 1.0 if is_consistent else max(0.5, 1.0 - len(consistency_issues) * 0.1)
        
        # 4. Timeliness (assume real-time data is timely)
        timeliness = 1.0
        
        # 5. Anomaly detection
        has_anomalies, anomaly_list, anomaly_scores = self.detect_anomalies(data)
        avg_anomaly_score = np.mean(list(anomaly_scores.values())) if anomaly_scores else 0.0
        anomaly_metric = 1.0 - avg_anomaly_score
        
        # Overall quality (weighted average)
        overall_quality = (
            completeness * 0.25 +
            accuracy * 0.25 +
            consistency * 0.20 +
            timeliness * 0.10 +
            anomaly_metric * 0.20
        )
        
        # Create metrics object
        metrics = DataQualityMetrics(
            completeness=completeness,
            accuracy=accuracy,
            consistency=consistency,
            timeliness=timeliness,
            anomaly_score=avg_anomaly_score,
            overall_quality=overall_quality,
            details={
                'has_anomalies': has_anomalies,
                'anomaly_list': anomaly_list,
                'consistency_issues': consistency_issues,
                'anomaly_scores': anomaly_scores
            }
        )
        
        # Cache metrics
        self.quality_cache.append(metrics.dict())
        
        return metrics
    
    def process_and_validate(self, data: Dict) -> Dict:
        """
        Main middleware function: validate, check quality, detect anomalies
        
        Args:
            data: Input data dictionary
            
        Returns:
            Dictionary with validation results and quality metrics
        """
        result = {
            'data': data,
            'validation': {},
            'quality_metrics': {},
            'anomalies': {},
            'passed': False,
            'timestamp': datetime.now().isoformat()
        }
        
        # 1. Schema validation
        is_valid, error_msg, validated_data = self.validate_input(data)
        result['validation'] = {
            'passed': is_valid,
            'error': error_msg,
            'validated_data': validated_data
        }
        
        if not is_valid:
            logger.warning(f"Validation failed: {error_msg}")
            return result
        
        # 2. Quality metrics
        quality_metrics = self.calculate_quality_metrics(data)
        result['quality_metrics'] = quality_metrics.dict()
        
        # 3. Anomaly detection
        has_anomalies, anomaly_list, anomaly_scores = self.detect_anomalies(data)
        result['anomalies'] = {
            'detected': has_anomalies,
            'descriptions': anomaly_list,
            'scores': anomaly_scores
        }
        
        # 4. Overall pass/fail
        result['passed'] = (
            is_valid and
            quality_metrics.overall_quality >= 0.6 and
            not (has_anomalies and quality_metrics.anomaly_score > 0.7)
        )
        
        # Log quality issues
        if not result['passed']:
            logger.warning(f"Data quality check failed: Quality={quality_metrics.overall_quality:.2f}")
            if has_anomalies:
                logger.warning(f"Anomalies detected: {anomaly_list}")
        
        # Save metrics periodically
        if len(self.quality_cache) >= 100:
            self.save_quality_metrics()
        
        return result
    
    def save_quality_metrics(self):
        """Save accumulated quality metrics to disk"""
        if not self.quality_cache:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.metrics_dir / f"quality_metrics_{timestamp}.json"
        
        with open(filepath, 'w') as f:
            json.dump(list(self.quality_cache), f, indent=2)
        
        logger.info(f"Saved {len(self.quality_cache)} quality metrics to {filepath}")
    
    def get_quality_report(self, last_n: int = 100) -> Dict:
        """
        Generate quality report from recent data
        
        Args:
            last_n: Number of recent metrics to analyze
            
        Returns:
            Dictionary with quality statistics
        """
        if not self.quality_cache:
            return {'message': 'No quality data available'}
        
        recent = list(self.quality_cache)[-last_n:]
        
        df = pd.DataFrame(recent)
        
        report = {
            'total_samples': len(recent),
            'average_quality': {
                'completeness': df['completeness'].mean(),
                'accuracy': df['accuracy'].mean(),
                'consistency': df['consistency'].mean(),
                'overall': df['overall_quality'].mean()
            },
            'quality_distribution': {
                'high (>0.8)': int((df['overall_quality'] > 0.8).sum()),
                'medium (0.6-0.8)': int(((df['overall_quality'] >= 0.6) & (df['overall_quality'] <= 0.8)).sum()),
                'low (<0.6)': int((df['overall_quality'] < 0.6).sum())
            },
            'anomaly_rate': df['anomaly_score'].mean(),
            'timestamp': datetime.now().isoformat()
        }
        
        return report


# FastAPI Middleware Integration
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class DataQualityHTTPMiddleware(BaseHTTPMiddleware):
    """FastAPI HTTP middleware for data quality checking"""
    
    def __init__(self, app, validator: DataQualityMiddleware):
        super().__init__(app)
        self.validator = validator
    
    async def dispatch(self, request: Request, call_next):
        # Only validate POST requests with JSON body
        if request.method == "POST" and "application/json" in request.headers.get("content-type", ""):
            try:
                # Read and parse body
                body = await request.body()
                import json
                data = json.loads(body)
                
                # Extract weather data if present
                weather_data = data.get('weather', data)
                
                # Validate and check quality
                result = self.validator.process_and_validate(weather_data)
                
                # If quality check fails, return warning (but don't block)
                if not result['passed']:
                    logger.warning(f"Data quality issues: {result}")
                    # Add warning header
                    response = await call_next(request)
                    response.headers['X-Data-Quality-Warning'] = 'true'
                    return response
                
            except Exception as e:
                logger.error(f"Middleware error: {e}")
        
        # Continue with normal request processing
        response = await call_next(request)
        return response


if __name__ == "__main__":
    # Example usage
    middleware = DataQualityMiddleware()
    
    # Test data
    test_data = {
        'temperature_2m': 25.5,
        'relative_humidity_2m': 75.0,
        'pressure_msl': 1013.25,
        'wind_speed_10m': 5.2,
        'wind_direction_10m': 180.0,
        'cloud_cover': 60.0,
        'precipitation': 2.5
    }
    
    result = middleware.process_and_validate(test_data)
    print(json.dumps(result, indent=2, default=str))
