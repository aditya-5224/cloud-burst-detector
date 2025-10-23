"""
Main Pipeline Script for Cloud Burst Prediction System

This script orchestrates the complete ML pipeline including data collection,
preprocessing, feature engineering, model training, and evaluation.
"""

import argparse
import logging
import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import os

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.data.weather_api import WeatherDataCollector
from src.data.satellite_imagery import SatelliteImageryClient
from src.preprocessing.image_processing import ImageProcessor
from src.features.feature_engineering import WeatherFeatureEngineer
from src.models.baseline_models import BaselineModels

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CloudBurstPipeline:
    """Main pipeline class for cloud burst prediction system"""
    
    def __init__(self, config_path: str = "./config/config.yaml"):
        """Initialize the pipeline"""
        self.config_path = config_path
        self.load_config()
        
        # Initialize components
        self.weather_collector = WeatherDataCollector(config_path)
        self.satellite_client = SatelliteImageryClient(config_path)
        self.image_processor = ImageProcessor(config_path)
        self.feature_engineer = WeatherFeatureEngineer(config_path)
        self.model_trainer = BaselineModels(config_path)
        
        logger.info("Pipeline components initialized")
    
    def load_config(self):
        """Load configuration"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def collect_weather_data(self, region: str = 'default') -> str:
        """
        Step 1: Collect weather data from APIs
        
        Args:
            region: Region name to collect data for
            
        Returns:
            Path to collected data file
        """
        logger.info(f"Starting weather data collection for region: {region}")
        
        try:
            weather_data = self.weather_collector.collect_all_data(region)
            
            if weather_data.empty:
                logger.warning("No weather data collected")
                return None
            
            # Save data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"weather_data_{region}_{timestamp}.csv"
            data_path = Path(self.config['data']['weather_data_path'])
            data_path.mkdir(parents=True, exist_ok=True)
            filepath = data_path / filename
            weather_data.to_csv(filepath, index=False)
            
            logger.info(f"Weather data collection completed: {len(weather_data)} records saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error in weather data collection: {e}")
            raise
    
    def collect_satellite_data(self, region: str = 'default', days_back: int = 7) -> list:
        """
        Step 2: Collect satellite imagery data
        
        Args:
            region: Region name to collect data for
            days_back: Number of days to collect data for
            
        Returns:
            List of paths to image files
        """
        logger.info(f"Starting satellite data collection for region: {region}")
        
        try:
            satellite_images = self.satellite_client.collect_daily_imagery(region, days_back)
            
            if not satellite_images:
                logger.warning("No satellite images collected")
                return []
            
            logger.info(f"Satellite data collection completed: {len(satellite_images)} images collected")
            return satellite_images
            
        except Exception as e:
            logger.error(f"Error in satellite data collection: {e}")
            raise
    
    def process_images(self, satellite_images: list, region: str = 'default') -> str:
        """
        Step 3: Process satellite images and extract features
        
        Args:
            satellite_images: List of satellite image arrays
            region: Region name
            
        Returns:
            Path to processed image features file
        """
        logger.info("Starting image processing and feature extraction")
        
        try:
            if not satellite_images:
                logger.warning("No satellite images to process")
                return None
            
            # Process images and extract features
            image_features = self.image_processor.process_image_batch(satellite_images, region)
            
            if image_features.empty:
                logger.warning("No image features extracted")
                return None
            
            # The process_image_batch method already saves the file
            # Get the most recent image features file
            processed_path = Path(self.config['data']['processed_data_path'])
            feature_files = list(processed_path.glob(f"image_features_{region}_*.csv"))
            
            if feature_files:
                latest_file = max(feature_files, key=lambda x: x.stat().st_mtime)
                logger.info(f"Image processing completed: {len(image_features)} features saved to {latest_file}")
                return str(latest_file)
            else:
                logger.warning("Image features file not found after processing")
                return None
                
        except Exception as e:
            logger.error(f"Error in image processing: {e}")
            raise
    
    def engineer_features(self, weather_data_path: str) -> str:
        """
        Step 4: Engineer features from weather data
        
        Args:
            weather_data_path: Path to weather data CSV file
            
        Returns:
            Path to engineered features file
        """
        logger.info("Starting feature engineering")
        
        try:
            if not weather_data_path or not Path(weather_data_path).exists():
                logger.error("Weather data file not found")
                return None
            
            # Load weather data
            weather_data = pd.read_csv(weather_data_path)
            
            # Engineer features
            engineered_data = self.feature_engineer.engineer_all_features(weather_data)
            
            if engineered_data.empty:
                logger.warning("No features engineered")
                return None
            
            # Save engineered features
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"engineered_features_{timestamp}.csv"
            features_path = self.feature_engineer.save_engineered_features(
                engineered_data, filename
            )
            
            logger.info(f"Feature engineering completed: {len(engineered_data)} rows, "
                       f"{len(engineered_data.columns)} features saved to {features_path}")
            return features_path
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            raise
    
    def train_models(self, features_data_path: str) -> dict:
        """
        Step 5: Train baseline ML models
        
        Args:
            features_data_path: Path to engineered features CSV file
            
        Returns:
            Dictionary with model evaluation results
        """
        logger.info("Starting model training")
        
        try:
            if not features_data_path or not Path(features_data_path).exists():
                logger.error("Features data file not found")
                return {}
            
            # Load features data
            import pandas as pd
            features_data = pd.read_csv(features_data_path)
            
            # Train all baseline models
            results = self.model_trainer.train_all_models(features_data)
            
            # Generate and save report
            report = self.model_trainer.generate_model_report(results)
            
            # Save report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = Path(f"./reports/training_report_{timestamp}.txt")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            logger.info(f"Model training completed. Report saved to {report_path}")
            print(report)  # Also print to console
            
            return results
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            raise
    
    def run_full_pipeline(self, region: str = 'default', days_back: int = 7) -> dict:
        """
        Run the complete ML pipeline
        
        Args:
            region: Region name to process
            days_back: Number of days of satellite data to collect
            
        Returns:
            Dictionary with pipeline results
        """
        logger.info("=" * 60)
        logger.info("STARTING FULL CLOUD BURST PREDICTION PIPELINE")
        logger.info("=" * 60)
        
        pipeline_results = {
            'start_time': datetime.now(),
            'region': region,
            'days_back': days_back,
            'weather_data_path': None,
            'satellite_images_count': 0,
            'image_features_path': None,
            'engineered_features_path': None,
            'model_results': {},
            'success': False,
            'error': None
        }
        
        try:
            # Step 1: Collect weather data
            logger.info("STEP 1: Collecting weather data...")
            weather_data_path = self.collect_weather_data(region)
            pipeline_results['weather_data_path'] = weather_data_path
            
            # Step 2: Collect satellite data
            logger.info("STEP 2: Collecting satellite imagery...")
            satellite_images = self.collect_satellite_data(region, days_back)
            pipeline_results['satellite_images_count'] = len(satellite_images)
            
            # Step 3: Process images
            logger.info("STEP 3: Processing satellite images...")
            image_features_path = self.process_images(satellite_images, region)
            pipeline_results['image_features_path'] = image_features_path
            
            # Step 4: Engineer features
            logger.info("STEP 4: Engineering features...")
            if weather_data_path:
                engineered_features_path = self.engineer_features(weather_data_path)
                pipeline_results['engineered_features_path'] = engineered_features_path
                
                # Step 5: Train models
                logger.info("STEP 5: Training models...")
                model_results = self.train_models(engineered_features_path)
                pipeline_results['model_results'] = model_results
            else:
                logger.warning("Skipping feature engineering and model training due to missing weather data")
            
            pipeline_results['success'] = True
            pipeline_results['end_time'] = datetime.now()
            pipeline_results['duration'] = pipeline_results['end_time'] - pipeline_results['start_time']
            
            logger.info("=" * 60)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"Duration: {pipeline_results['duration']}")
            logger.info("=" * 60)
            
        except Exception as e:
            pipeline_results['error'] = str(e)
            pipeline_results['end_time'] = datetime.now()
            pipeline_results['duration'] = pipeline_results['end_time'] - pipeline_results['start_time']
            
            logger.error("=" * 60)
            logger.error("PIPELINE FAILED")
            logger.error(f"Error: {e}")
            logger.error(f"Duration: {pipeline_results['duration']}")
            logger.error("=" * 60)
            
            raise
        
        return pipeline_results


def main():
    """Main function for command-line execution"""
    parser = argparse.ArgumentParser(description='Cloud Burst Prediction Pipeline')
    parser.add_argument('--config', default='./config/config.yaml', 
                       help='Path to configuration file')
    parser.add_argument('--region', default='default', 
                       help='Region name to process')
    parser.add_argument('--days-back', type=int, default=7, 
                       help='Number of days of satellite data to collect')
    parser.add_argument('--step', choices=['weather', 'satellite', 'images', 'features', 'models', 'full'],
                       default='full', help='Which pipeline step to run')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = CloudBurstPipeline(args.config)
    
    try:
        if args.step == 'full':
            results = pipeline.run_full_pipeline(args.region, args.days_back)
            print(f"Pipeline completed. Success: {results['success']}")
            
        elif args.step == 'weather':
            path = pipeline.collect_weather_data(args.region)
            print(f"Weather data collected: {path}")
            
        elif args.step == 'satellite':
            images = pipeline.collect_satellite_data(args.region, args.days_back)
            print(f"Satellite images collected: {len(images)}")
            
        elif args.step == 'images':
            # This would require existing satellite data
            print("Image processing step requires satellite data from previous step")
            
        elif args.step == 'features':
            # This would require existing weather data
            print("Feature engineering step requires weather data from previous step")
            
        elif args.step == 'models':
            # This would require existing features data
            print("Model training step requires engineered features from previous step")
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()