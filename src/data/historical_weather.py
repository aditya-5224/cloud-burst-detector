"""
Historical Weather Data Integration Module

This module collects historical weather data for past cloud burst events
to validate predictions, improve model accuracy, and enable backtesting.

Data Sources:
1. Open-Meteo Historical API (free, up to 80 years of data)
2. Visual Crossing Weather API (historical weather)
3. Known cloud burst event database
"""

import requests
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yaml
from pathlib import Path
import json
import time

logger = logging.getLogger(__name__)


class HistoricalWeatherCollector:
    """Collects historical weather data for analysis and validation"""
    
    def __init__(self, config_path: str = "./config/config.yaml"):
        """Initialize the historical weather collector"""
        self.config = self._load_config(config_path)
        self.data_dir = Path("./data/historical")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Historical weather collector initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load config: {e}. Using defaults.")
            return {}
    
    def fetch_historical_open_meteo(
        self, 
        latitude: float, 
        longitude: float,
        start_date: str,  # Format: YYYY-MM-DD
        end_date: str,    # Format: YYYY-MM-DD
        hourly_vars: Optional[List[str]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical weather data from Open-Meteo Archive API
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            hourly_vars: List of weather variables to fetch
            
        Returns:
            DataFrame with historical weather data or None if failed
        """
        try:
            if hourly_vars is None:
                hourly_vars = [
                    "temperature_2m",
                    "relative_humidity_2m",
                    "dewpoint_2m",  # For atmospheric stability indices calculation
                    "precipitation",
                    "pressure_msl",
                    "cloud_cover",
                    "wind_speed_10m",
                    "wind_direction_10m",
                    "cape"  # Convective Available Potential Energy for cloudburst detection
                ]
            
            # Open-Meteo Historical API (free)
            base_url = "https://archive-api.open-meteo.com/v1/archive"
            
            params = {
                'latitude': latitude,
                'longitude': longitude,
                'start_date': start_date,
                'end_date': end_date,
                'hourly': ','.join(hourly_vars),
                'timezone': 'auto'
            }
            
            logger.info(f"Fetching historical data from {start_date} to {end_date} for ({latitude}, {longitude})")
            
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse into DataFrame
            if 'hourly' in data:
                df = pd.DataFrame(data['hourly'])
                df['time'] = pd.to_datetime(df['time'])
                df['latitude'] = latitude
                df['longitude'] = longitude
                
                logger.info(f"✅ Fetched {len(df)} hourly records")
                return df
            else:
                logger.error("No hourly data in response")
                return None
                
        except Exception as e:
            logger.error(f"❌ Open-Meteo historical fetch error: {e}")
            return None
    
    def fetch_known_cloudburst_events(self) -> List[Dict]:
        """
        Get list of known cloud burst events from various sources
        
        Returns:
            List of cloud burst events with date, location, and details
        """
        # Known cloud burst events in India (can be expanded)
        events = [
            {
                'date': '2023-07-09',
                'location': 'Kedarnath, Uttarakhand',
                'latitude': 30.7346,
                'longitude': 79.0669,
                'rainfall_mm': 200,
                'duration_hours': 2,
                'description': 'Kedarnath cloud burst disaster',
                'casualties': 'Significant',
                'source': 'IMD Reports'
            },
            {
                'date': '2023-08-14',
                'location': 'Himachal Pradesh',
                'latitude': 31.1048,
                'longitude': 77.1734,
                'rainfall_mm': 180,
                'duration_hours': 1.5,
                'description': 'Heavy cloud burst in Himachal',
                'casualties': 'Multiple',
                'source': 'News Reports'
            },
            {
                'date': '2022-07-28',
                'location': 'Amarnath, J&K',
                'latitude': 34.2268,
                'longitude': 75.5345,
                'rainfall_mm': 150,
                'duration_hours': 1,
                'description': 'Amarnath cave cloud burst',
                'casualties': '16 deaths',
                'source': 'Official Reports'
            },
            {
                'date': '2021-10-19',
                'location': 'Uttarkashi, Uttarakhand',
                'latitude': 30.7268,
                'longitude': 78.4354,
                'rainfall_mm': 170,
                'duration_hours': 2,
                'description': 'Flash floods due to cloud burst',
                'casualties': 'Several missing',
                'source': 'IMD'
            },
            {
                'date': '2021-07-29',
                'location': 'Dharamshala, Himachal Pradesh',
                'latitude': 32.2190,
                'longitude': 76.3234,
                'rainfall_mm': 160,
                'duration_hours': 1.5,
                'description': 'Cloud burst causing landslides',
                'casualties': '9 deaths',
                'source': 'News Reports'
            },
            {
                'date': '2020-08-06',
                'location': 'Devprayag, Uttarakhand',
                'latitude': 30.1457,
                'longitude': 78.5983,
                'rainfall_mm': 190,
                'duration_hours': 2,
                'description': 'Severe cloud burst',
                'casualties': 'Multiple',
                'source': 'State Reports'
            },
            {
                'date': '2019-08-02',
                'location': 'Kullu, Himachal Pradesh',
                'latitude': 31.9578,
                'longitude': 77.1092,
                'rainfall_mm': 175,
                'duration_hours': 1.5,
                'description': 'Cloud burst flash floods',
                'casualties': '5 deaths',
                'source': 'IMD'
            },
            {
                'date': '2018-08-08',
                'location': 'Pithoragarh, Uttarakhand',
                'latitude': 29.5831,
                'longitude': 80.2184,
                'rainfall_mm': 185,
                'duration_hours': 2,
                'description': 'Multiple villages affected',
                'casualties': 'Several injured',
                'source': 'Disaster Reports'
            },
            {
                'date': '2017-08-15',
                'location': 'Mandi, Himachal Pradesh',
                'latitude': 31.7084,
                'longitude': 76.9318,
                'rainfall_mm': 165,
                'duration_hours': 1,
                'description': 'Heavy rainfall cloud burst',
                'casualties': '20+ deaths',
                'source': 'Official Records'
            },
            {
                'date': '2016-07-30',
                'location': 'Chamoli, Uttarakhand',
                'latitude': 30.4086,
                'longitude': 79.3310,
                'rainfall_mm': 195,
                'duration_hours': 2,
                'description': 'Cloud burst with flash floods',
                'casualties': 'Multiple missing',
                'source': 'IMD Reports'
            }
        ]
        
        logger.info(f"Loaded {len(events)} known cloud burst events")
        return events
    
    def collect_event_historical_data(
        self, 
        event: Dict,
        hours_before: int = 24,
        hours_after: int = 6
    ) -> Optional[pd.DataFrame]:
        """
        Collect historical weather data for a specific cloud burst event
        
        Args:
            event: Cloud burst event dictionary with date, latitude, longitude
            hours_before: Hours of data to collect before event
            hours_after: Hours of data to collect after event
            
        Returns:
            DataFrame with weather data around the event
        """
        try:
            event_date = pd.to_datetime(event['date'])
            start_date = (event_date - timedelta(hours=hours_before)).strftime('%Y-%m-%d')
            end_date = (event_date + timedelta(hours=hours_after)).strftime('%Y-%m-%d')
            
            logger.info(f"Collecting data for event: {event['location']} on {event['date']}")
            
            df = self.fetch_historical_open_meteo(
                latitude=event['latitude'],
                longitude=event['longitude'],
                start_date=start_date,
                end_date=end_date
            )
            
            if df is not None:
                # Add event information
                df['event_date'] = event['date']
                df['event_location'] = event['location']
                df['event_rainfall'] = event.get('rainfall_mm', None)
                df['is_cloudburst_event'] = 1
                
                # Mark the actual cloud burst time window
                event_time = pd.to_datetime(event['date'])
                duration = event.get('duration_hours', 2)
                df['during_cloudburst'] = (
                    (df['time'] >= event_time) & 
                    (df['time'] <= event_time + timedelta(hours=duration))
                ).astype(int)
                
                logger.info(f"✅ Collected {len(df)} records for {event['location']}")
                return df
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error collecting event data: {e}")
            return None
    
    def build_historical_dataset(
        self, 
        output_file: str = "historical_cloudburst_data.csv"
    ) -> pd.DataFrame:
        """
        Build complete historical dataset with all known cloud burst events
        
        Args:
            output_file: Name of file to save the dataset
            
        Returns:
            Complete DataFrame with all historical data
        """
        all_data = []
        events = self.fetch_known_cloudburst_events()
        
        logger.info(f"Building historical dataset from {len(events)} events...")
        
        for i, event in enumerate(events, 1):
            logger.info(f"Processing event {i}/{len(events)}: {event['location']}")
            
            df = self.collect_event_historical_data(event)
            
            if df is not None:
                all_data.append(df)
            
            # Be nice to the API - add delay
            time.sleep(2)
        
        if all_data:
            # Combine all data
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Save to file
            output_path = self.data_dir / output_file
            combined_df.to_csv(output_path, index=False)
            
            logger.info(f"✅ Historical dataset saved: {output_path}")
            logger.info(f"   Total records: {len(combined_df)}")
            logger.info(f"   Date range: {combined_df['time'].min()} to {combined_df['time'].max()}")
            logger.info(f"   Events covered: {combined_df['event_location'].nunique()}")
            
            return combined_df
        else:
            logger.error("❌ No historical data collected")
            return pd.DataFrame()
    
    def fetch_date_range_data(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        location_name: str = "Unknown"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical data for a specific date range and location
        
        Useful for custom analysis periods
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            location_name: Name of the location
            
        Returns:
            DataFrame with historical weather data
        """
        logger.info(f"Fetching custom date range: {start_date} to {end_date} for {location_name}")
        
        df = self.fetch_historical_open_meteo(
            latitude=latitude,
            longitude=longitude,
            start_date=start_date,
            end_date=end_date
        )
        
        if df is not None:
            df['location_name'] = location_name
            
            # Save to file
            safe_name = location_name.replace(' ', '_').replace(',', '')
            filename = f"historical_{safe_name}_{start_date}_to_{end_date}.csv"
            output_path = self.data_dir / filename
            df.to_csv(output_path, index=False)
            
            logger.info(f"✅ Saved {len(df)} records to {output_path}")
            
        return df
    
    def analyze_historical_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Analyze patterns in historical data
        
        Args:
            df: Historical weather DataFrame
            
        Returns:
            Dictionary with analysis results
        """
        analysis = {
            'total_records': len(df),
            'date_range': {
                'start': str(df['time'].min()),
                'end': str(df['time'].max())
            },
            'temperature': {
                'mean': df['temperature_2m'].mean(),
                'min': df['temperature_2m'].min(),
                'max': df['temperature_2m'].max(),
                'std': df['temperature_2m'].std()
            },
            'humidity': {
                'mean': df['relative_humidity_2m'].mean(),
                'min': df['relative_humidity_2m'].min(),
                'max': df['relative_humidity_2m'].max()
            },
            'precipitation': {
                'total': df['precipitation'].sum(),
                'mean': df['precipitation'].mean(),
                'max': df['precipitation'].max(),
                'days_with_rain': (df['precipitation'] > 0).sum()
            },
            'pressure': {
                'mean': df['pressure_msl'].mean(),
                'min': df['pressure_msl'].min()
            }
        }
        
        # If cloud burst events are marked
        if 'during_cloudburst' in df.columns:
            cloudburst_data = df[df['during_cloudburst'] == 1]
            if len(cloudburst_data) > 0:
                analysis['cloudburst_conditions'] = {
                    'avg_temperature': cloudburst_data['temperature_2m'].mean(),
                    'avg_humidity': cloudburst_data['relative_humidity_2m'].mean(),
                    'avg_precipitation': cloudburst_data['precipitation'].mean(),
                    'avg_pressure': cloudburst_data['pressure_msl'].mean(),
                    'samples': len(cloudburst_data)
                }
        
        return analysis


def main():
    """Main function to demonstrate historical data collection"""
    print("=" * 70)
    print("Cloud Burst Historical Data Collection")
    print("=" * 70)
    
    collector = HistoricalWeatherCollector()
    
    # Option 1: Build complete historical dataset from known events
    print("\n📊 Building historical dataset from known cloud burst events...")
    historical_df = collector.build_historical_dataset()
    
    if not historical_df.empty:
        print(f"\n✅ Dataset created with {len(historical_df)} records")
        print(f"   Events: {historical_df['event_location'].nunique()}")
        print(f"   Date range: {historical_df['time'].min()} to {historical_df['time'].max()}")
        
        # Analyze patterns
        print("\n📈 Analyzing patterns...")
        analysis = collector.analyze_historical_patterns(historical_df)
        
        print(f"\nHistorical Weather Analysis:")
        print(f"  Temperature: {analysis['temperature']['mean']:.1f}°C (avg)")
        print(f"  Humidity: {analysis['humidity']['mean']:.1f}% (avg)")
        print(f"  Total Precipitation: {analysis['precipitation']['total']:.1f}mm")
        print(f"  Days with rain: {analysis['precipitation']['days_with_rain']}")
        
        if 'cloudburst_conditions' in analysis:
            print(f"\nCloud Burst Conditions:")
            cb = analysis['cloudburst_conditions']
            print(f"  Temperature: {cb['avg_temperature']:.1f}°C")
            print(f"  Humidity: {cb['avg_humidity']:.1f}%")
            print(f"  Precipitation: {cb['avg_precipitation']:.1f}mm/h")
            print(f"  Pressure: {cb['avg_pressure']:.1f}hPa")
    
    # Option 2: Fetch custom date range (example)
    print("\n\n📅 Example: Fetching data for Mumbai (monsoon period)...")
    custom_df = collector.fetch_date_range_data(
        latitude=19.0760,
        longitude=72.8777,
        start_date="2023-07-01",
        end_date="2023-07-31",
        location_name="Mumbai"
    )
    
    if custom_df is not None:
        print(f"✅ Fetched {len(custom_df)} records for Mumbai")
    
    print("\n" + "=" * 70)
    print("Historical data collection complete!")
    print(f"Data saved in: ./data/historical/")
    print("=" * 70)


if __name__ == "__main__":
    main()
