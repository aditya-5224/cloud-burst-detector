"""
Historical Weather Data Collection

Collects and stores historical weather data for model training.
Includes data validation, gap filling, and quality checks.
"""

import pandas as pd
import numpy as np
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import time
import yaml
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.database import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HistoricalDataCollector:
    """Collects historical weather data for training"""
    
    def __init__(self, config_path: str = "./config/config.yaml"):
        """Initialize historical data collector"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.db = DatabaseManager()
        self.regions = self.config['regions']
        
    def collect_historical_open_meteo(self, 
                                     latitude: float,
                                     longitude: float,
                                     start_date: datetime,
                                     end_date: datetime,
                                     region_name: str = 'default') -> pd.DataFrame:
        """
        Collect historical data from Open-Meteo Archive API
        
        Open-Meteo provides free historical weather data from 1940 onwards
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            start_date: Start date for historical data
            end_date: End date for historical data
            region_name: Name of the region
            
        Returns:
            DataFrame with historical weather data
        """
        
        logger.info(f"Collecting historical data from {start_date} to {end_date}")
        
        # Open-Meteo Archive API endpoint
        url = "https://archive-api.open-meteo.com/v1/archive"
        
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'hourly': [
                'temperature_2m',
                'relative_humidity_2m',
                'precipitation',
                'pressure_msl',
                'surface_pressure',
                'cloud_cover',
                'wind_speed_10m',
                'wind_direction_10m',
                'wind_gusts_10m'
            ],
            'timezone': 'UTC'
        }
        
        all_data = []
        current_start = start_date
        
        # API limit is 1 year per request, so we need to chunk
        while current_start < end_date:
            current_end = min(current_start + timedelta(days=365), end_date)
            
            params['start_date'] = current_start.strftime('%Y-%m-%d')
            params['end_date'] = current_end.strftime('%Y-%m-%d')
            
            try:
                logger.info(f"Fetching data from {params['start_date']} to {params['end_date']}")
                response = requests.get(url, params=params, timeout=60)
                response.raise_for_status()
                
                data = response.json()
                
                if 'hourly' in data:
                    hourly_data = data['hourly']
                    df = pd.DataFrame(hourly_data)
                    df['datetime'] = pd.to_datetime(df['time'])
                    df = df.drop('time', axis=1)
                    df['latitude'] = latitude
                    df['longitude'] = longitude
                    df['region'] = region_name
                    df['source'] = 'open_meteo_archive'
                    
                    all_data.append(df)
                    logger.info(f"Collected {len(df)} records")
                else:
                    logger.warning(f"No data returned for period {params['start_date']} to {params['end_date']}")
                
                # Respect API rate limits
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error fetching data for period {params['start_date']} to {params['end_date']}: {e}")
                
            current_start = current_end + timedelta(days=1)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"Total historical records collected: {len(combined_df)}")
            return combined_df
        else:
            logger.warning("No historical data collected")
            return pd.DataFrame()
    
    def validate_data_quality(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Validate and assess data quality
        
        Args:
            df: DataFrame with weather data
            
        Returns:
            Tuple of (cleaned DataFrame, quality metrics)
        """
        
        initial_count = len(df)
        quality_metrics = {
            'initial_records': initial_count,
            'missing_values': {},
            'outliers_removed': 0,
            'duplicates_removed': 0,
            'final_records': 0
        }
        
        # Check for duplicates
        duplicates = df.duplicated(subset=['datetime', 'latitude', 'longitude'])
        df = df[~duplicates]
        quality_metrics['duplicates_removed'] = duplicates.sum()
        
        # Check missing values
        for col in df.columns:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            if missing_pct > 0:
                quality_metrics['missing_values'][col] = f"{missing_pct:.2f}%"
        
        # Fill missing values with interpolation
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().any():
                df[col] = df[col].interpolate(method='linear', limit_direction='both')
        
        # Remove outliers (using IQR method)
        outlier_columns = ['temperature_2m', 'relative_humidity_2m', 'pressure_msl', 'wind_speed_10m']
        for col in outlier_columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                quality_metrics['outliers_removed'] += outliers.sum()
                df = df[~outliers]
        
        quality_metrics['final_records'] = len(df)
        quality_metrics['data_quality_score'] = (quality_metrics['final_records'] / initial_count) * 100
        
        logger.info(f"Data quality: {quality_metrics['data_quality_score']:.2f}%")
        logger.info(f"Removed {quality_metrics['duplicates_removed']} duplicates")
        logger.info(f"Removed {quality_metrics['outliers_removed']} outliers")
        
        return df, quality_metrics
    
    def collect_and_store_historical_data(self,
                                         region_name: str = 'default',
                                         months_back: int = 6) -> Dict:
        """
        Collect historical data and store in database
        
        Args:
            region_name: Name of the region
            months_back: Number of months of historical data to collect
            
        Returns:
            Dictionary with collection statistics
        """
        
        if region_name not in self.regions:
            logger.error(f"Region {region_name} not found")
            return {}
        
        region = self.regions[region_name]
        latitude = region['latitude']
        longitude = region['longitude']
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=months_back * 30)
        
        logger.info(f"Collecting {months_back} months of historical data for {region_name}")
        logger.info(f"Period: {start_date.date()} to {end_date.date()}")
        
        # Collect data
        df = self.collect_historical_open_meteo(
            latitude, longitude, start_date, end_date, region_name
        )
        
        if df.empty:
            logger.error("No data collected")
            return {'success': False, 'records_collected': 0}
        
        # Validate data quality
        df_cleaned, quality_metrics = self.validate_data_quality(df)
        
        # Store in database
        try:
            records_inserted = self.db.insert_weather_data(df_cleaned)
            
            # Log collection
            self.db.log_data_collection({
                'collection_datetime': datetime.now(),
                'data_type': 'historical_weather',
                'source': 'open_meteo_archive',
                'records_collected': records_inserted,
                'success': True
            })
            
            logger.info(f"✓ Successfully stored {records_inserted} historical records")
            
            return {
                'success': True,
                'region': region_name,
                'start_date': start_date,
                'end_date': end_date,
                'records_collected': records_inserted,
                'quality_metrics': quality_metrics
            }
            
        except Exception as e:
            logger.error(f"Error storing data: {e}")
            self.db.log_data_collection({
                'collection_datetime': datetime.now(),
                'data_type': 'historical_weather',
                'source': 'open_meteo_archive',
                'records_collected': 0,
                'success': False,
                'error_message': str(e)
            })
            
            return {'success': False, 'error': str(e)}
    
    def generate_data_summary_report(self) -> str:
        """Generate a summary report of collected data"""
        
        # Get data range from database
        conn = self.db.get_connection()
        
        summary_query = """
            SELECT 
                region,
                source,
                COUNT(*) as record_count,
                MIN(datetime) as earliest_date,
                MAX(datetime) as latest_date,
                AVG(temperature_2m) as avg_temp,
                AVG(relative_humidity_2m) as avg_humidity,
                AVG(precipitation) as avg_precipitation
            FROM weather_data
            GROUP BY region, source
        """
        
        df_summary = pd.read_sql_query(summary_query, conn)
        
        report = "=" * 80 + "\n"
        report += "HISTORICAL WEATHER DATA COLLECTION SUMMARY\n"
        report += "=" * 80 + "\n\n"
        
        if df_summary.empty:
            report += "No data collected yet.\n"
        else:
            for _, row in df_summary.iterrows():
                report += f"Region: {row['region']} | Source: {row['source']}\n"
                report += f"  Records: {row['record_count']:,}\n"
                report += f"  Period: {row['earliest_date']} to {row['latest_date']}\n"
                report += f"  Avg Temperature: {row['avg_temp']:.1f}°C\n"
                report += f"  Avg Humidity: {row['avg_humidity']:.1f}%\n"
                report += f"  Avg Precipitation: {row['avg_precipitation']:.2f} mm\n"
                report += "\n"
        
        report += "=" * 80 + "\n"
        
        return report


def main():
    """Main function for collecting historical data"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Historical Weather Data Collection')
    parser.add_argument('--region', default='default', help='Region name')
    parser.add_argument('--months', type=int, default=6, help='Months of data to collect')
    parser.add_argument('--summary', action='store_true', help='Show data summary')
    
    args = parser.parse_args()
    
    collector = HistoricalDataCollector()
    
    if args.summary:
        report = collector.generate_data_summary_report()
        print(report)
    else:
        result = collector.collect_and_store_historical_data(
            region_name=args.region,
            months_back=args.months
        )
        
        if result.get('success'):
            print(f"\n✓ Successfully collected {result['records_collected']} records")
            print(f"  Region: {result['region']}")
            print(f"  Period: {result['start_date'].date()} to {result['end_date'].date()}")
            print(f"  Data Quality Score: {result['quality_metrics']['data_quality_score']:.2f}%")
            
            # Show summary
            report = collector.generate_data_summary_report()
            print(f"\n{report}")
        else:
            print(f"\n✗ Data collection failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()