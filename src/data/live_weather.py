"""
Live Weather Data Collection Module

Fetches real-time weather data from multiple weather APIs:
- Open-Meteo (free, no API key required)
- OpenWeatherMap (backup, requires API key)

Provides current weather conditions including temperature, humidity, pressure,
precipitation, wind speed, and cloud cover for cloud burst prediction.
"""

import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import yaml
from pathlib import Path
import time
from threading import Lock

logger = logging.getLogger(__name__)


class LiveWeatherCollector:
    """Collects real-time weather data from multiple weather APIs"""
    
    def __init__(self, config_path: str = "./config/config.yaml"):
        """
        Initialize the live weather collector
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.cache = {}
        self.cache_lock = Lock()
        self.cache_ttl = 300  # Cache for 5 minutes
        logger.info("Live weather collector initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load config: {e}. Using defaults.")
            return {}
    
    def _get_cache_key(self, latitude: float, longitude: float) -> str:
        """Generate cache key for location"""
        return f"{latitude:.4f},{longitude:.4f}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        with self.cache_lock:
            if cache_key not in self.cache:
                return False
            
            cached_time = self.cache[cache_key].get('timestamp')
            if not cached_time:
                return False
            
            age = (datetime.now() - cached_time).total_seconds()
            return age < self.cache_ttl
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Get data from cache if valid"""
        if self._is_cache_valid(cache_key):
            with self.cache_lock:
                logger.info(f"Using cached weather data for {cache_key}")
                return self.cache[cache_key]['data']
        return None
    
    def _save_to_cache(self, cache_key: str, data: Dict):
        """Save data to cache"""
        with self.cache_lock:
            self.cache[cache_key] = {
                'data': data,
                'timestamp': datetime.now()
            }
    
    def get_location_name(self, latitude: float, longitude: float) -> Dict[str, str]:
        """
        Get location name from coordinates using reverse geocoding
        
        Uses multiple services with fallback:
        1. Nominatim (OpenStreetMap) - Free, no API key
        2. WeatherAPI.com location data - Already have the key
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            Dictionary with location information (city, state, country)
        """
        location_info = {
            'city': 'Unknown',
            'state': '',
            'country': 'Unknown',
            'display_name': f"{latitude:.4f}, {longitude:.4f}"
        }
        
        try:
            # Try Nominatim (OpenStreetMap) first - free, no API key needed
            url = "https://nominatim.openstreetmap.org/reverse"
            params = {
                'lat': latitude,
                'lon': longitude,
                'format': 'json',
                'zoom': 10,
                'addressdetails': 1
            }
            headers = {
                'User-Agent': 'CloudBurstPredictor/1.0'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            address = data.get('address', {})
            
            # Extract location components
            city = (address.get('city') or 
                   address.get('town') or 
                   address.get('village') or 
                   address.get('municipality') or
                   address.get('county') or
                   'Unknown')
            
            state = (address.get('state') or 
                    address.get('state_district') or 
                    address.get('region') or '')
            
            country = address.get('country', 'Unknown')
            
            # Create display name
            display_parts = [city]
            if state and state != city:
                display_parts.append(state)
            if country:
                display_parts.append(country)
            
            location_info = {
                'city': city,
                'state': state,
                'country': country,
                'display_name': ', '.join(display_parts)
            }
            
            logger.info(f"✅ Resolved location: {location_info['display_name']}")
            return location_info
            
        except Exception as e:
            logger.warning(f"Nominatim geocoding failed: {e}")
            
            # Fallback: Try to get location from WeatherAPI.com if available
            try:
                api_config = self.config.get('weather_apis', {}).get('weatherapi', {})
                api_key = api_config.get('api_key')
                
                if api_key and api_key != 'YOUR_WEATHERAPI_KEY':
                    url = "https://api.weatherapi.com/v1/current.json"
                    params = {
                        'key': api_key,
                        'q': f"{latitude},{longitude}",
                        'aqi': 'no'
                    }
                    
                    response = requests.get(url, params=params, timeout=5)
                    response.raise_for_status()
                    
                    data = response.json()
                    loc = data.get('location', {})
                    
                    location_info = {
                        'city': loc.get('name', 'Unknown'),
                        'state': loc.get('region', ''),
                        'country': loc.get('country', 'Unknown'),
                        'display_name': f"{loc.get('name', 'Unknown')}, {loc.get('country', 'Unknown')}"
                    }
                    
                    logger.info(f"✅ Resolved location via WeatherAPI: {location_info['display_name']}")
                    return location_info
                    
            except Exception as e2:
                logger.warning(f"WeatherAPI geocoding fallback failed: {e2}")
        
        # Return default with coordinates if all geocoding fails
        logger.warning(f"Could not resolve location name for {latitude}, {longitude}")
        return location_info
    
    def fetch_open_meteo(self, latitude: float, longitude: float) -> Optional[Dict]:
        """
        Fetch weather data from Open-Meteo API (free, no API key needed)
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            Dictionary with current weather data or None if failed
        """
        try:
            base_url = "https://api.open-meteo.com/v1/forecast"
            
            params = {
                'latitude': latitude,
                'longitude': longitude,
                'current': [
                    'temperature_2m',
                    'relative_humidity_2m',
                    'precipitation',
                    'rain',
                    'pressure_msl',
                    'cloud_cover',
                    'wind_speed_10m',
                    'wind_direction_10m',
                    'weather_code'
                ],
                'timezone': 'auto'
            }
            
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            current = data.get('current', {})
            
            # Convert to our standard format
            weather_data = {
                'source': 'open-meteo',
                'timestamp': datetime.now().isoformat(),
                'location': {
                    'latitude': latitude,
                    'longitude': longitude
                },
                'temperature': current.get('temperature_2m'),  # °C
                'humidity': current.get('relative_humidity_2m'),  # %
                'pressure': current.get('pressure_msl'),  # hPa
                'precipitation': current.get('precipitation', 0),  # mm
                'rain': current.get('rain', 0),  # mm
                'cloud_cover': current.get('cloud_cover'),  # %
                'wind_speed': current.get('wind_speed_10m'),  # km/h
                'wind_direction': current.get('wind_direction_10m'),  # degrees
                'weather_code': current.get('weather_code')
            }
            
            logger.info(f"✅ Fetched weather from Open-Meteo for ({latitude}, {longitude})")
            return weather_data
            
        except Exception as e:
            logger.error(f"❌ Open-Meteo API error: {e}")
            return None
    
    def fetch_weatherapi(self, latitude: float, longitude: float) -> Optional[Dict]:
        """
        Fetch weather data from WeatherAPI.com (free tier available, more accurate)
        
        This source provides more accurate real-time data from weather stations
        compared to Open-Meteo's model-based forecasts.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            Dictionary with current weather data or None if failed
        """
        try:
            # WeatherAPI.com - Free tier: 1M calls/month, real-time station data
            # Get API key from config if available
            api_config = self.config.get('weather_apis', {}).get('weatherapi', {})
            api_key = api_config.get('api_key', 'demo')
            
            # If no key configured, skip to next source
            if not api_key or api_key == 'YOUR_WEATHERAPI_KEY':
                logger.info("WeatherAPI.com API key not configured, skipping...")
                return None
            
            base_url = "https://api.weatherapi.com/v1/current.json"
            
            params = {
                'key': api_key,
                'q': f"{latitude},{longitude}",
                'aqi': 'no'
            }
            
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            current = data.get('current', {})
            
            # Convert to our standard format
            weather_data = {
                'source': 'weatherapi.com',
                'timestamp': datetime.now().isoformat(),
                'location': {
                    'latitude': latitude,
                    'longitude': longitude
                },
                'temperature': current.get('temp_c'),  # °C
                'humidity': current.get('humidity'),  # %
                'pressure': current.get('pressure_mb'),  # hPa/mb
                'precipitation': current.get('precip_mm', 0),  # mm
                'rain': current.get('precip_mm', 0),  # mm
                'cloud_cover': current.get('cloud'),  # %
                'wind_speed': current.get('wind_kph'),  # km/h
                'wind_direction': current.get('wind_degree'),  # degrees
                'weather_code': current.get('condition', {}).get('code'),
                'feels_like': current.get('feelslike_c'),  # °C
                'uv_index': current.get('uv'),
                'visibility': current.get('vis_km')
            }
            
            logger.info(f"✅ Fetched weather from WeatherAPI.com for ({latitude}, {longitude})")
            return weather_data
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"WeatherAPI.com request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ WeatherAPI.com error: {e}")
            return None
    
    def fetch_openweathermap(self, latitude: float, longitude: float) -> Optional[Dict]:
        """
        Fetch weather data from OpenWeatherMap API (requires API key)
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            Dictionary with current weather data or None if failed
        """
        try:
            api_config = self.config.get('weather_apis', {}).get('openweathermap', {})
            api_key = api_config.get('api_key')
            
            if not api_key or api_key == 'YOUR_OPENWEATHERMAP_API_KEY':
                logger.warning("OpenWeatherMap API key not configured")
                return None
            
            base_url = api_config.get('base_url', 'https://api.openweathermap.org/data/2.5/weather')
            
            params = {
                'lat': latitude,
                'lon': longitude,
                'appid': api_key,
                'units': 'metric'
            }
            
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            main = data.get('main', {})
            wind = data.get('wind', {})
            clouds = data.get('clouds', {})
            rain = data.get('rain', {})
            
            # Convert to our standard format
            weather_data = {
                'source': 'openweathermap',
                'timestamp': datetime.now().isoformat(),
                'location': {
                    'latitude': latitude,
                    'longitude': longitude
                },
                'temperature': main.get('temp'),  # °C
                'humidity': main.get('humidity'),  # %
                'pressure': main.get('pressure'),  # hPa
                'precipitation': rain.get('1h', 0),  # mm in last hour
                'rain': rain.get('1h', 0),  # mm in last hour
                'cloud_cover': clouds.get('all'),  # %
                'wind_speed': wind.get('speed', 0) * 3.6,  # Convert m/s to km/h
                'wind_direction': wind.get('deg'),  # degrees
                'weather_code': data.get('weather', [{}])[0].get('id')
            }
            
            logger.info(f"✅ Fetched weather from OpenWeatherMap for ({latitude}, {longitude})")
            return weather_data
            
        except Exception as e:
            logger.error(f"❌ OpenWeatherMap API error: {e}")
            return None
    
    def get_live_weather(self, latitude: float, longitude: float, 
                        force_refresh: bool = False) -> Optional[Dict]:
        """
        Get current weather data for a location with automatic fallback
        
        Priority order:
        1. WeatherAPI.com - Most accurate real-time station data
        2. Open-Meteo - Free, reliable, but model-based (15-30 min lag)
        3. OpenWeatherMap - Requires API key
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            force_refresh: Skip cache and fetch fresh data
            
        Returns:
            Dictionary with current weather data or None if all sources fail
        """
        cache_key = self._get_cache_key(latitude, longitude)
        
        # Try cache first
        if not force_refresh:
            cached_data = self._get_from_cache(cache_key)
            if cached_data:
                return cached_data
        
        # Try WeatherAPI first (most accurate real-time data)
        weather_data = self.fetch_weatherapi(latitude, longitude)
        
        # Fallback to Open-Meteo (free, no API key needed)
        if not weather_data:
            logger.info("WeatherAPI failed, falling back to Open-Meteo...")
            weather_data = self.fetch_open_meteo(latitude, longitude)
        
        # Fallback to OpenWeatherMap as last resort
        if not weather_data:
            logger.info("Open-Meteo failed, falling back to OpenWeatherMap...")
            weather_data = self.fetch_openweathermap(latitude, longitude)
        
        # Add location name to weather data
        if weather_data:
            location_info = self.get_location_name(latitude, longitude)
            weather_data['location_name'] = location_info['display_name']
            weather_data['location_details'] = {
                'city': location_info['city'],
                'state': location_info['state'],
                'country': location_info['country']
            }
        
        # Cache the data if successful
        if weather_data:
            self._save_to_cache(cache_key, weather_data)
            return weather_data
        
        logger.error(f"❌ All weather APIs failed for ({latitude}, {longitude})")
        return None
    
    def get_weather_for_prediction(self, latitude: float, longitude: float) -> Optional[Dict]:
        """
        Get weather data formatted for prediction model input
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            Dictionary with weather features for model prediction
        """
        weather = self.get_live_weather(latitude, longitude)
        
        if not weather:
            return None
        
        # Format for prediction model
        prediction_data = {
            'temperature': weather.get('temperature', 25.0),
            'humidity': weather.get('humidity', 70.0),
            'pressure': weather.get('pressure', 1013.0),
            'precipitation': weather.get('precipitation', 0.0),
            'wind_speed': weather.get('wind_speed', 10.0),
            'cloud_cover': weather.get('cloud_cover', 50.0),
            'timestamp': weather.get('timestamp'),
            'location': weather.get('location'),
            'location_name': weather.get('location_name', 'Unknown Location'),
            'location_details': weather.get('location_details', {}),
            'source': weather.get('source', 'unknown')
        }
        
        return prediction_data
    
    def get_weather_summary(self, latitude: float, longitude: float) -> str:
        """
        Get a human-readable weather summary
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            Formatted weather summary string
        """
        weather = self.get_live_weather(latitude, longitude)
        
        if not weather:
            return "Weather data unavailable"
        
        summary = f"""
🌡️ Temperature: {weather.get('temperature', 'N/A')}°C
💧 Humidity: {weather.get('humidity', 'N/A')}%
🌧️ Precipitation: {weather.get('precipitation', 'N/A')} mm/h
☁️ Cloud Cover: {weather.get('cloud_cover', 'N/A')}%
🌪️ Wind Speed: {weather.get('wind_speed', 'N/A')} km/h
⏱️ Pressure: {weather.get('pressure', 'N/A')} hPa
📍 Source: {weather.get('source', 'N/A')}
        """.strip()
        
        return summary
    
    def clear_cache(self):
        """Clear all cached weather data"""
        with self.cache_lock:
            self.cache.clear()
            logger.info("Weather cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        with self.cache_lock:
            valid_entries = sum(1 for key in self.cache.keys() 
                              if self._is_cache_valid(key))
            return {
                'total_entries': len(self.cache),
                'valid_entries': valid_entries,
                'cache_ttl_seconds': self.cache_ttl
            }


# Global instance for easy access
live_weather_collector = LiveWeatherCollector()


def get_current_weather(latitude: float, longitude: float) -> Optional[Dict]:
    """
    Convenience function to get current weather
    
    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        
    Returns:
        Dictionary with current weather data
    """
    return live_weather_collector.get_live_weather(latitude, longitude)


def get_weather_for_prediction(latitude: float, longitude: float) -> Optional[Dict]:
    """
    Convenience function to get weather data for prediction
    
    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        
    Returns:
        Dictionary with weather features for model prediction
    """
    return live_weather_collector.get_weather_for_prediction(latitude, longitude)


if __name__ == "__main__":
    # Test the live weather collector
    logging.basicConfig(level=logging.INFO)
    
    # Test locations
    locations = [
        (19.0760, 72.8777, "Mumbai"),
        (28.6139, 77.2090, "Delhi"),
        (12.9716, 77.5946, "Bangalore")
    ]
    
    collector = LiveWeatherCollector()
    
    print("=" * 60)
    print("Live Weather Data Collection Test")
    print("=" * 60)
    
    for lat, lon, city in locations:
        print(f"\n📍 {city} ({lat}, {lon})")
        print("-" * 60)
        
        weather = collector.get_live_weather(lat, lon)
        if weather:
            print(collector.get_weather_summary(lat, lon))
        else:
            print("❌ Failed to fetch weather data")
        
        time.sleep(1)  # Rate limiting
    
    print("\n" + "=" * 60)
    print("Cache Statistics:")
    print(collector.get_cache_stats())
    print("=" * 60)
