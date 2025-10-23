"""
Cloud Burst Prediction System

A comprehensive machine learning system for predicting cloud burst events
using meteorological data and satellite imagery analysis.
"""

__version__ = "1.0.0"
__author__ = "Cloud Burst Prediction Team"
__email__ = "team@cloudburst-prediction.com"

# Don't import modules at package level to avoid circular dependencies
# Import them in your code when needed: from src.data.weather_api import WeatherDataCollector
__all__ = [
    'WeatherDataCollector',
    'SatelliteImageryClient', 
    'ImageProcessor',
    'WeatherFeatureEngineer',
    'BaselineModels'
]