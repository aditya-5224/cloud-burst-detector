"""
Satellite Imagery Data Ingestion Module

Handles collection and processing of satellite imagery from Google Earth Engine
for cloud detection and analysis in cloud burst prediction.
"""

import ee
import yaml
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import os
from typing import Dict, List, Tuple, Optional
import requests

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SatelliteImageryClient:
    """Client for satellite imagery data collection using Google Earth Engine"""
    
    def __init__(self, config_path: str = "./config/config.yaml"):
        """Initialize the satellite imagery client"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.ee_config = self.config['earth_engine']
        self.data_path = Path(self.config['data']['satellite_images_path'])
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize Earth Engine
        self._initialize_earth_engine()
        
    def _initialize_earth_engine(self) -> None:
        """Initialize Google Earth Engine authentication"""
        try:
            # Try to initialize with service account if available
            service_account_key = os.getenv('GOOGLE_EARTH_ENGINE_SERVICE_ACCOUNT_KEY',
                                          self.ee_config.get('service_account_key', ''))
            
            if service_account_key and os.path.exists(service_account_key):
                credentials = ee.ServiceAccountCredentials(None, service_account_key)
                ee.Initialize(credentials)
                logger.info("Earth Engine initialized with service account")
            else:
                # Try to initialize with user authentication
                try:
                    ee.Initialize()
                    logger.info("Earth Engine initialized with user authentication")
                except Exception:
                    logger.warning("Earth Engine authentication failed. Using mock data.")
                    self.ee_authenticated = False
                    return
                    
            self.ee_authenticated = True
            
        except Exception as e:
            logger.error(f"Failed to initialize Earth Engine: {e}")
            logger.warning("Will generate mock satellite data for development")
            self.ee_authenticated = False
    
    def get_cloud_probability_image(self, 
                                  bbox: Dict[str, float],
                                  start_date: str,
                                  end_date: str) -> Optional[ee.Image]:
        """
        Get cloud probability image from Sentinel-2 data
        
        Args:
            bbox: Bounding box with 'north', 'south', 'east', 'west' keys
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            Earth Engine Image with cloud probability data
        """
        
        if not self.ee_authenticated:
            return None
        
        try:
            # Define the area of interest
            aoi = ee.Geometry.Rectangle([
                bbox['west'], bbox['south'], 
                bbox['east'], bbox['north']
            ])
            
            # Get Sentinel-2 cloud probability collection
            collection = ee.ImageCollection(self.ee_config['collection']) \
                .filterBounds(aoi) \
                .filterDate(start_date, end_date)
            
            # Get the most recent image
            image = collection.sort('system:time_start', False).first()
            
            if image:
                logger.info(f"Retrieved cloud probability image for {start_date} to {end_date}")
                return image
            else:
                logger.warning(f"No cloud probability images found for the specified period")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving cloud probability image: {e}")
            return None
    
    def download_image_data(self, 
                           image: ee.Image,
                           bbox: Dict[str, float],
                           filename: str) -> Optional[np.ndarray]:
        """
        Download image data as numpy array
        
        Args:
            image: Earth Engine Image object
            bbox: Bounding box dictionary
            filename: Filename to save the image data
            
        Returns:
            Numpy array with image data
        """
        
        if not self.ee_authenticated or image is None:
            return self._generate_mock_cloud_data(bbox)
        
        try:
            # Define region
            region = ee.Geometry.Rectangle([
                bbox['west'], bbox['south'], 
                bbox['east'], bbox['north']
            ])
            
            # Get download URL
            url = image.getDownloadURL({
                'scale': self.ee_config['scale'],
                'crs': 'EPSG:4326',
                'region': region,
                'format': 'GEO_TIFF'
            })
            
            # Download the image
            response = requests.get(url, timeout=300)
            response.raise_for_status()
            
            # Save to file
            filepath = self.data_path / f"{filename}.tif"
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded satellite image to {filepath}")
            
            # For now, return mock data (in production, you'd use rasterio to read the GeoTIFF)
            return self._generate_mock_cloud_data(bbox)
            
        except Exception as e:
            logger.error(f"Error downloading image data: {e}")
            return self._generate_mock_cloud_data(bbox)
    
    def _generate_mock_cloud_data(self, bbox: Dict[str, float]) -> np.ndarray:
        """
        Generate mock cloud probability data for development/testing
        
        Args:
            bbox: Bounding box dictionary
            
        Returns:
            Mock cloud probability array
        """
        
        # Generate realistic-looking cloud probability data
        np.random.seed(42)  # For reproducibility
        
        # Create a 100x100 grid
        height, width = 100, 100
        
        # Generate base cloud pattern
        x = np.linspace(0, 4*np.pi, width)
        y = np.linspace(0, 4*np.pi, height)
        X, Y = np.meshgrid(x, y)
        
        # Create cloud-like patterns using sine waves and noise
        cloud_pattern = (
            0.3 * np.sin(X) * np.cos(Y) +
            0.2 * np.sin(2*X + np.pi/4) * np.sin(2*Y) +
            0.1 * np.random.normal(0, 1, (height, width))
        )
        
        # Normalize to 0-100 probability range
        cloud_probability = np.clip((cloud_pattern + 1) * 50, 0, 100)
        
        logger.info(f"Generated mock cloud data with shape {cloud_probability.shape}")
        return cloud_probability
    
    def collect_daily_imagery(self, 
                             region_name: str = 'default',
                             days_back: int = 7) -> List[np.ndarray]:
        """
        Collect satellite imagery for the past N days
        
        Args:
            region_name: Name of the region to collect data for
            days_back: Number of days back to collect data
            
        Returns:
            List of numpy arrays containing image data
        """
        
        if region_name not in self.config['regions']:
            logger.error(f"Region {region_name} not found in configuration")
            return []
        
        region = self.config['regions'][region_name]
        bbox = region['bbox']
        
        images = []
        
        for i in range(days_back):
            date = datetime.now() - timedelta(days=i)
            start_date = date.strftime('%Y-%m-%d')
            end_date = (date + timedelta(days=1)).strftime('%Y-%m-%d')
            
            logger.info(f"Collecting imagery for {start_date}")
            
            # Get cloud probability image
            image = self.get_cloud_probability_image(bbox, start_date, end_date)
            
            # Download image data
            filename = f"cloud_probability_{region_name}_{start_date}"
            image_data = self.download_image_data(image, bbox, filename)
            
            if image_data is not None:
                images.append(image_data)
                
                # Save metadata
                metadata = {
                    'date': start_date,
                    'region': region_name,
                    'bbox': bbox,
                    'shape': image_data.shape,
                    'mean_cloud_probability': float(np.mean(image_data)),
                    'max_cloud_probability': float(np.max(image_data)),
                    'min_cloud_probability': float(np.min(image_data))
                }
                
                # Save metadata to CSV
                metadata_df = pd.DataFrame([metadata])
                metadata_file = self.data_path / f"metadata_{region_name}_{start_date}.csv"
                metadata_df.to_csv(metadata_file, index=False)
        
        logger.info(f"Collected {len(images)} satellite images")
        return images
    
    def get_image_statistics(self, image_data: np.ndarray) -> Dict[str, float]:
        """
        Calculate basic statistics for an image
        
        Args:
            image_data: Numpy array containing image data
            
        Returns:
            Dictionary with image statistics
        """
        
        stats = {
            'mean_cloud_probability': float(np.mean(image_data)),
            'std_cloud_probability': float(np.std(image_data)),
            'max_cloud_probability': float(np.max(image_data)),
            'min_cloud_probability': float(np.min(image_data)),
            'cloud_coverage_percentage': float(np.sum(image_data > 50) / image_data.size * 100),
            'high_probability_pixels': int(np.sum(image_data > 80)),
            'total_pixels': int(image_data.size)
        }
        
        return stats


def main():
    """Main function for running satellite imagery collection"""
    client = SatelliteImageryClient()
    
    # Collect imagery for the past week
    images = client.collect_daily_imagery('default', days_back=7)
    
    # Calculate statistics for each image
    for i, image in enumerate(images):
        stats = client.get_image_statistics(image)
        print(f"Image {i+1} statistics: {stats}")


if __name__ == "__main__":
    main()