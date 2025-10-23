"""
Weather API Data Ingestion Module

Handles data collection from Open-Meteo and OpenWeatherMap APIs
for meteorological data used in cloud burst prediction.
"""

import requests
import pandas as pd
import yaml
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import time
import os
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeatherAPIClient:
    """Base class for weather API clients"""
    
    def __init__(self, config_path: str = "./config/config.yaml"):
        """Initialize with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_path = Path(self.config['data']['weather_data_path'])
        self.data_path.mkdir(parents=True, exist_ok=True)
    
    def save_data(self, data: pd.DataFrame, filename: str) -> None:
        """Save data to CSV file"""
        filepath = self.data_path / filename
        data.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")


class OpenMeteoClient(WeatherAPIClient):
    """Client for Open-Meteo API"""
    
    def __init__(self, config_path: str = "./config/config.yaml"):
        super().__init__(config_path)
        self.api_config = self.config['weather_apis']['open_meteo']
        self.base_url = self.api_config['base_url']
        
    def fetch_forecast_data(self, 
                           latitude: float, 
                           longitude: float,
                           hours: int = 168) -> pd.DataFrame:
        """
        Fetch forecast data from Open-Meteo API
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            hours: Number of hours to forecast (default: 168 = 7 days)
            
        Returns:
            DataFrame with weather data
        """
        
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'hourly': ','.join(self.api_config['hourly_params']),
            'timezone': 'UTC',
            'forecast_days': min(16, hours // 24 + 1)  # API limit is 16 days
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to DataFrame
            hourly_data = data['hourly']
            df = pd.DataFrame(hourly_data)
            df['datetime'] = pd.to_datetime(df['time'])
            df['latitude'] = latitude
            df['longitude'] = longitude
            df['source'] = 'open_meteo'
            
            # Drop the original 'time' column
            df = df.drop('time', axis=1)
            
            logger.info(f"Fetched {len(df)} records from Open-Meteo API")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Open-Meteo data: {e}")
            return pd.DataFrame()


class OpenWeatherMapClient(WeatherAPIClient):
    """Client for OpenWeatherMap API"""
    
    def __init__(self, config_path: str = "./config/config.yaml"):
        super().__init__(config_path)
        self.api_config = self.config['weather_apis']['openweathermap']
        self.base_url = self.api_config['base_url']
        self.api_key = os.getenv('OPENWEATHERMAP_API_KEY', 
                                self.api_config.get('api_key', ''))
        
        if not self.api_key or self.api_key == 'YOUR_OPENWEATHERMAP_API_KEY':
            logger.warning("OpenWeatherMap API key not configured")
    
    def fetch_current_weather(self, 
                             latitude: float, 
                             longitude: float) -> pd.DataFrame:
        """
        Fetch current weather data from OpenWeatherMap API
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            DataFrame with current weather data
        """
        
        if not self.api_key or self.api_key == 'YOUR_OPENWEATHERMAP_API_KEY':
            logger.error("OpenWeatherMap API key not configured")
            return pd.DataFrame()
        
        params = {
            'lat': latitude,
            'lon': longitude,
            'appid': self.api_key,
            'units': self.api_config.get('units', 'metric')
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract relevant data
            weather_data = {
                'datetime': [datetime.utcnow()],
                'latitude': [latitude],
                'longitude': [longitude],
                'temperature_2m': [data['main']['temp']],
                'relative_humidity_2m': [data['main']['humidity']],
                'pressure_msl': [data['main']['pressure']],
                'wind_speed_10m': [data['wind']['speed']],
                'wind_direction_10m': [data['wind'].get('deg', 0)],
                'cloud_cover': [data['clouds']['all']],
                'precipitation': [data.get('rain', {}).get('1h', 0)],
                'source': ['openweathermap']
            }
            
            df = pd.DataFrame(weather_data)
            logger.info("Fetched current weather from OpenWeatherMap API")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OpenWeatherMap data: {e}")
            return pd.DataFrame()


class WeatherDataCollector:
    """Main class for collecting weather data from multiple sources"""
    
    def __init__(self, config_path: str = "./config/config.yaml"):
        """Initialize data collector"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.open_meteo = OpenMeteoClient(config_path)
        self.openweathermap = OpenWeatherMapClient(config_path)
        self.regions = self.config['regions']
        
    def collect_all_data(self, region_name: str = 'default') -> pd.DataFrame:
        """
        Collect data from all sources for a specific region
        
        Args:
            region_name: Name of the region to collect data for
            
        Returns:
            Combined DataFrame with all weather data
        """
        
        if region_name not in self.regions:
            logger.error(f"Region {region_name} not found in configuration")
            return pd.DataFrame()
        
        region = self.regions[region_name]
        lat, lon = region['latitude'], region['longitude']
        
        # Collect data from all sources
        dataframes = []
        
        # Open-Meteo forecast data
        meteo_data = self.open_meteo.fetch_forecast_data(lat, lon)
        if not meteo_data.empty:
            dataframes.append(meteo_data)
        
        # OpenWeatherMap current data
        owm_data = self.openweathermap.fetch_current_weather(lat, lon)
        if not owm_data.empty:
            dataframes.append(owm_data)
        
        # Combine all data
        if dataframes:
            combined_data = pd.concat(dataframes, ignore_index=True)
            combined_data = combined_data.sort_values('datetime')
            
            # Save combined data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"weather_data_{region_name}_{timestamp}.csv"
            self.open_meteo.save_data(combined_data, filename)
            
            logger.info(f"Collected {len(combined_data)} total weather records")
            return combined_data
        
        logger.warning("No weather data collected")
        return pd.DataFrame()
    
    def scheduled_collection(self, 
                           region_name: str = 'default',
                           interval_minutes: int = 60) -> None:
        """
        Run scheduled data collection
        
        Args:
            region_name: Region to collect data for
            interval_minutes: Collection interval in minutes
        """
        
        logger.info(f"Starting scheduled collection every {interval_minutes} minutes")
        
        while True:
            try:
                logger.info("Starting scheduled weather data collection...")
                self.collect_all_data(region_name)
                logger.info(f"Sleeping for {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("Scheduled collection stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in scheduled collection: {e}")
                time.sleep(60)  # Wait 1 minute before retrying


def main():
    """Main function for running weather data collection"""
    collector = WeatherDataCollector()
    
    # Collect data once
    data = collector.collect_all_data('default')
    print(f"Collected {len(data)} weather records")
    
    # For continuous collection, uncomment the following line:
    # collector.scheduled_collection('default', interval_minutes=60)


if __name__ == "__main__":
    main()