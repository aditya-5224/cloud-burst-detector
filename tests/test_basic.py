"""
Basic tests for the Cloud Burst Prediction System

Simple unit tests to verify core functionality of the system components.
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data.weather_api import WeatherDataCollector, OpenMeteoClient
from src.preprocessing.image_processing import ImageProcessor
from src.features.feature_engineering import WeatherFeatureEngineer
from src.models.baseline_models import BaselineModels


class TestWeatherAPI(unittest.TestCase):
    """Test weather API functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.yaml"
        
        # Create minimal test config
        test_config = """
weather_apis:
  open_meteo:
    base_url: "https://api.open-meteo.com/v1/forecast"
    hourly_params: ["temperature_2m", "relative_humidity_2m", "pressure_msl"]
data:
  weather_data_path: "{}"
""".format(self.temp_dir)
        
        with open(self.config_path, 'w') as f:
            f.write(test_config)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_weather_collector_init(self):
        """Test weather collector initialization"""
        collector = WeatherDataCollector(str(self.config_path))
        self.assertIsNotNone(collector)
        self.assertIsInstance(collector.open_meteo, OpenMeteoClient)


class TestImageProcessing(unittest.TestCase):
    """Test image processing functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = ImageProcessor()
        self.sample_image = np.random.uniform(0, 100, (50, 50))
    
    def test_cloud_mask_generation(self):
        """Test cloud mask generation"""
        mask = self.processor.generate_cloud_mask(self.sample_image, threshold=50.0)
        self.assertEqual(mask.shape, self.sample_image.shape)
        self.assertTrue(np.all((mask == 0) | (mask == 1)))
    
    def test_cloud_coverage_calculation(self):
        """Test cloud coverage calculation"""
        # Create test mask with known coverage
        test_mask = np.zeros((10, 10))
        test_mask[:5, :5] = 1  # 25% coverage
        
        coverage = self.processor.calculate_cloud_coverage(test_mask)
        self.assertEqual(coverage, 25.0)
    
    def test_feature_extraction(self):
        """Test feature extraction from image"""
        features = self.processor.extract_all_features(self.sample_image)
        self.assertIsInstance(features, dict)
        self.assertIn('cloud_coverage_percentage', features)
        self.assertIn('glcm_contrast', features)


class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engineer = WeatherFeatureEngineer()
        
        # Create sample weather data
        dates = pd.date_range(start='2023-01-01', end='2023-01-03', freq='H')
        self.sample_data = pd.DataFrame({
            'datetime': dates,
            'temperature_2m': np.random.normal(25, 5, len(dates)),
            'relative_humidity_2m': np.random.uniform(40, 90, len(dates)),
            'pressure_msl': np.random.normal(1013, 10, len(dates)),
            'wind_speed_10m': np.random.exponential(3, len(dates)),
            'cloud_cover': np.random.uniform(0, 100, len(dates)),
            'precipitation': np.random.exponential(0.5, len(dates))
        })
    
    def test_rolling_features(self):
        """Test rolling feature creation"""
        columns = ['temperature_2m', 'relative_humidity_2m']
        result = self.engineer.create_rolling_features(self.sample_data, columns)
        
        # Check that rolling features were created
        for col in columns:
            for window in self.engineer.rolling_windows:
                self.assertIn(f'{col}_rolling_{window}h_mean', result.columns)
    
    def test_time_features(self):
        """Test time feature creation"""
        result = self.engineer.create_time_features(self.sample_data)
        
        expected_features = ['hour', 'day_of_week', 'month', 'hour_sin', 'hour_cos']
        for feature in expected_features:
            self.assertIn(feature, result.columns)
    
    def test_cape_calculation(self):
        """Test CAPE calculation"""
        temp = np.array([25, 30, 35])
        pressure = np.array([1013, 1010, 1005])
        humidity = np.array([70, 80, 90])
        
        cape = self.engineer.calculate_cape(temp, pressure, humidity)
        
        self.assertEqual(len(cape), 3)
        self.assertTrue(np.all(cape >= 0))  # CAPE should be non-negative


class TestBaselineModels(unittest.TestCase):
    """Test baseline model functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.trainer = BaselineModels()
        
        # Create sample training data
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        
        self.sample_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
        self.sample_df['cloud_burst_event'] = y
    
    def test_data_preparation(self):
        """Test data preparation for training"""
        X_train, X_val, X_test, y_train, y_val, y_test = self.trainer.prepare_data(
            self.sample_df, 'cloud_burst_event'
        )
        
        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_test), 0)
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))
    
    def test_random_forest_training(self):
        """Test Random Forest model training"""
        X_train, X_val, X_test, y_train, y_val, y_test = self.trainer.prepare_data(
            self.sample_df, 'cloud_burst_event'
        )
        
        model = self.trainer.train_random_forest(X_train, y_train)
        self.assertIsNotNone(model)
        
        # Test prediction
        predictions = model.predict(X_test)
        self.assertEqual(len(predictions), len(X_test))
    
    def test_synthetic_target_creation(self):
        """Test synthetic target creation"""
        df_without_target = self.sample_df.drop('cloud_burst_event', axis=1)
        df_with_target = self.trainer._create_synthetic_target(df_without_target, 'test_target')
        
        self.assertIn('test_target', df_with_target.columns)
        self.assertTrue(df_with_target['test_target'].isin([0, 1]).all())


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline"""
    
    def test_pipeline_components_integration(self):
        """Test that all pipeline components can work together"""
        
        # Test that we can create sample data and process it through the pipeline
        np.random.seed(42)
        
        # 1. Generate sample weather data
        dates = pd.date_range(start='2023-01-01', end='2023-01-02', freq='H')
        weather_data = pd.DataFrame({
            'datetime': dates,
            'temperature_2m': np.random.normal(25, 5, len(dates)),
            'relative_humidity_2m': np.random.uniform(40, 90, len(dates)),
            'pressure_msl': np.random.normal(1013, 10, len(dates)),
            'wind_speed_10m': np.random.exponential(3, len(dates)),
            'cloud_cover': np.random.uniform(0, 100, len(dates)),
            'precipitation': np.random.exponential(0.5, len(dates))
        })
        
        # 2. Engineer features
        engineer = WeatherFeatureEngineer()
        engineered_data = engineer.engineer_all_features(weather_data)
        
        # 3. Train a simple model
        trainer = BaselineModels()
        results = trainer.train_all_models(engineered_data)
        
        # Verify we got some results
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)


if __name__ == '__main__':
    unittest.main()