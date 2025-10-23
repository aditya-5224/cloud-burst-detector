"""
Feature Engineering Module

Handles feature engineering for meteorological data including rolling averages,
atmospheric indices calculation (CAPE, Lifted Index), and data transformation
for cloud burst prediction models.
"""

import pandas as pd
import numpy as np
import yaml
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeatherFeatureEngineer:
    """Class for engineering weather-based features"""
    
    def __init__(self, config_path: str = "./config/config.yaml"):
        """Initialize the feature engineer"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.feature_config = self.config['features']
        self.rolling_windows = self.feature_config['rolling_window_hours']
        self.processed_data_path = Path(self.config['data']['processed_data_path'])
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
    def create_rolling_features(self, df: pd.DataFrame, 
                               columns: List[str]) -> pd.DataFrame:
        """
        Create rolling average and difference features
        
        Args:
            df: DataFrame with weather data
            columns: List of columns to create rolling features for
            
        Returns:
            DataFrame with additional rolling features
        """
        
        df_features = df.copy()
        
        # Ensure datetime column is datetime type
        if 'datetime' in df_features.columns:
            df_features['datetime'] = pd.to_datetime(df_features['datetime'])
            df_features = df_features.sort_values('datetime')
        
        for col in columns:
            if col not in df_features.columns:
                logger.warning(f"Column {col} not found in data")
                continue
            
            for window in self.rolling_windows:
                # Rolling mean
                df_features[f'{col}_rolling_{window}h_mean'] = (
                    df_features[col].rolling(window=window, min_periods=1).mean()
                )
                
                # Rolling standard deviation
                df_features[f'{col}_rolling_{window}h_std'] = (
                    df_features[col].rolling(window=window, min_periods=1).std()
                )
                
                # Rolling min and max
                df_features[f'{col}_rolling_{window}h_min'] = (
                    df_features[col].rolling(window=window, min_periods=1).min()
                )
                
                df_features[f'{col}_rolling_{window}h_max'] = (
                    df_features[col].rolling(window=window, min_periods=1).max()
                )
                
                # Change from rolling mean
                rolling_mean = df_features[f'{col}_rolling_{window}h_mean']
                df_features[f'{col}_change_from_{window}h_mean'] = (
                    df_features[col] - rolling_mean
                )
        
        logger.info(f"Created rolling features for {len(columns)} columns with {len(self.rolling_windows)} windows")
        return df_features
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features
        
        Args:
            df: DataFrame with datetime column
            
        Returns:
            DataFrame with additional time features
        """
        
        df_features = df.copy()
        
        if 'datetime' not in df_features.columns:
            logger.error("No datetime column found")
            return df_features
        
        dt = pd.to_datetime(df_features['datetime'])
        
        # Basic time features
        df_features['hour'] = dt.dt.hour
        df_features['day_of_week'] = dt.dt.dayofweek
        df_features['month'] = dt.dt.month
        df_features['day_of_year'] = dt.dt.dayofyear
        
        # Cyclical encoding for time features
        df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
        df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
        
        df_features['day_of_week_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
        df_features['day_of_week_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
        
        df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
        df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
        
        # Season indicator
        df_features['is_summer'] = ((df_features['month'] >= 6) & 
                                   (df_features['month'] <= 8)).astype(int)
        df_features['is_monsoon'] = ((df_features['month'] >= 6) & 
                                    (df_features['month'] <= 9)).astype(int)
        
        logger.info("Created time-based features")
        return df_features
    
    def calculate_cape(self, 
                      temperature: np.ndarray,
                      pressure: np.ndarray,
                      humidity: np.ndarray) -> np.ndarray:
        """
        Calculate Convective Available Potential Energy (CAPE)
        Simplified calculation for surface-based CAPE
        
        Args:
            temperature: Temperature in Celsius
            pressure: Pressure in hPa
            humidity: Relative humidity in %
            
        Returns:
            CAPE values in J/kg
        """
        
        # Constants
        Rd = 287.04  # Specific gas constant for dry air (J/kg/K)
        Rv = 461.5   # Specific gas constant for water vapor (J/kg/K)
        Cp = 1005.0  # Specific heat at constant pressure (J/kg/K)
        L = 2.5e6    # Latent heat of vaporization (J/kg)
        
        # Convert temperature to Kelvin
        T_kelvin = temperature + 273.15
        
        # Calculate saturation vapor pressure (Tetens formula)
        es = 6.112 * np.exp(17.67 * temperature / (temperature + 243.5))
        
        # Calculate actual vapor pressure
        e = humidity * es / 100.0
        
        # Calculate mixing ratio
        w = 0.622 * e / (pressure - e)
        
        # Calculate potential temperature
        theta = T_kelvin * (1000.0 / pressure) ** (Rd / Cp)
        
        # Calculate equivalent potential temperature (simplified)
        theta_e = theta * np.exp(L * w / (Cp * T_kelvin))
        
        # Simplified CAPE calculation
        # This is a very simplified version - real CAPE requires vertical profile
        cape = np.maximum(0, (theta_e - theta) * 10)  # Scaling factor for realistic values
        
        return cape
    
    def calculate_lifted_index(self, 
                              temperature: np.ndarray,
                              pressure: np.ndarray,
                              humidity: np.ndarray) -> np.ndarray:
        """
        Calculate Lifted Index (simplified)
        
        Args:
            temperature: Temperature in Celsius
            pressure: Pressure in hPa
            humidity: Relative humidity in %
            
        Returns:
            Lifted Index values
        """
        
        # Simplified calculation based on surface conditions
        # Real LI requires 500 hPa temperature
        
        # Calculate dew point temperature
        def calculate_dewpoint(temp, rh):
            a = 17.27
            b = 237.7
            alpha = ((a * temp) / (b + temp)) + np.log(rh / 100.0)
            return (b * alpha) / (a - alpha)
        
        dewpoint = calculate_dewpoint(temperature, humidity)
        
        # Simplified lifted index calculation
        # LI = T500 - Tparcel_at_500
        # Using approximation based on surface conditions
        lifted_index = temperature - dewpoint - 8  # Simplified calculation
        
        return lifted_index
    
    def calculate_k_index(self, 
                         temperature: np.ndarray,
                         humidity: np.ndarray) -> np.ndarray:
        """
        Calculate K-Index (simplified)
        
        Args:
            temperature: Temperature in Celsius
            humidity: Relative humidity in %
            
        Returns:
            K-Index values
        """
        
        # Simplified K-Index calculation
        # Real K-Index: (T850 - T500) + Td850 - (T700 - Td700)
        # Using surface approximation
        
        dewpoint = self._calculate_dewpoint_simple(temperature, humidity)
        k_index = temperature + dewpoint - 25  # Simplified calculation
        
        return k_index
    
    def calculate_total_totals_index(self, 
                                   temperature: np.ndarray,
                                   humidity: np.ndarray) -> np.ndarray:
        """
        Calculate Total Totals Index (simplified)
        
        Args:
            temperature: Temperature in Celsius
            humidity: Relative humidity in %
            
        Returns:
            Total Totals Index values
        """
        
        dewpoint = self._calculate_dewpoint_simple(temperature, humidity)
        
        # Simplified Total Totals calculation
        # Real TT: (T850 + Td850) - 2*T500
        # Using surface approximation
        total_totals = temperature + dewpoint - 50  # Simplified calculation
        
        return total_totals
    
    def _calculate_dewpoint_simple(self, temperature: np.ndarray, 
                                  humidity: np.ndarray) -> np.ndarray:
        """Simple dewpoint calculation using Magnus formula"""
        a = 17.27
        b = 237.7
        alpha = ((a * temperature) / (b + temperature)) + np.log(humidity / 100.0)
        return (b * alpha) / (a - alpha)
    
    def calculate_atmospheric_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all atmospheric stability indices
        
        Args:
            df: DataFrame with weather data
            
        Returns:
            DataFrame with additional atmospheric indices
        """
        
        df_features = df.copy()
        
        required_cols = ['temperature_2m', 'pressure_msl', 'relative_humidity_2m']
        missing_cols = [col for col in required_cols if col not in df_features.columns]
        
        if missing_cols:
            logger.warning(f"Missing columns for atmospheric indices: {missing_cols}")
            return df_features
        
        temp = df_features['temperature_2m'].values
        pressure = df_features['pressure_msl'].values
        humidity = df_features['relative_humidity_2m'].values
        
        # Calculate indices
        df_features['cape'] = self.calculate_cape(temp, pressure, humidity)
        df_features['lifted_index'] = self.calculate_lifted_index(temp, pressure, humidity)
        df_features['k_index'] = self.calculate_k_index(temp, humidity)
        df_features['total_totals_index'] = self.calculate_total_totals_index(temp, humidity)
        
        logger.info("Calculated atmospheric stability indices")
        return df_features
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between different variables
        
        Args:
            df: DataFrame with weather data
            
        Returns:
            DataFrame with interaction features
        """
        
        df_features = df.copy()
        
        # Temperature-humidity interactions
        if all(col in df_features.columns for col in ['temperature_2m', 'relative_humidity_2m']):
            df_features['temp_humidity_interaction'] = (
                df_features['temperature_2m'] * df_features['relative_humidity_2m']
            )
            
            # Heat index calculation
            df_features['heat_index'] = self._calculate_heat_index(
                df_features['temperature_2m'], 
                df_features['relative_humidity_2m']
            )
        
        # Wind-pressure interactions
        if all(col in df_features.columns for col in ['wind_speed_10m', 'pressure_msl']):
            df_features['wind_pressure_interaction'] = (
                df_features['wind_speed_10m'] * df_features['pressure_msl']
            )
        
        # Cloud-precipitation interactions
        if all(col in df_features.columns for col in ['cloud_cover', 'precipitation']):
            df_features['cloud_precip_interaction'] = (
                df_features['cloud_cover'] * df_features['precipitation']
            )
        
        logger.info("Created interaction features")
        return df_features
    
    def _calculate_heat_index(self, temperature: np.ndarray, 
                             humidity: np.ndarray) -> np.ndarray:
        """Calculate heat index"""
        # Convert to Fahrenheit for calculation
        T = temperature * 9/5 + 32
        RH = humidity
        
        # Rothfusz regression
        HI = 0.5 * (T + 61.0 + ((T - 68.0) * 1.2) + (RH * 0.094))
        
        # Convert back to Celsius
        return (HI - 32) * 5/9
    
    def create_lag_features(self, df: pd.DataFrame, 
                           columns: List[str],
                           lags: List[int] = [1, 3, 6, 12]) -> pd.DataFrame:
        """
        Create lagged features
        
        Args:
            df: DataFrame with time series data
            columns: Columns to create lags for
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features
        """
        
        df_features = df.copy()
        
        for col in columns:
            if col not in df_features.columns:
                continue
                
            for lag in lags:
                df_features[f'{col}_lag_{lag}'] = df_features[col].shift(lag)
        
        logger.info(f"Created lag features for {len(columns)} columns")
        return df_features
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering steps
        
        Args:
            df: Raw weather DataFrame
            
        Returns:
            DataFrame with all engineered features
        """
        
        logger.info("Starting comprehensive feature engineering")
        
        # Make a copy to avoid modifying original data
        df_features = df.copy()
        
        # Ensure datetime is properly formatted
        if 'datetime' in df_features.columns:
            df_features['datetime'] = pd.to_datetime(df_features['datetime'])
            df_features = df_features.sort_values('datetime')
        
        # Define numeric columns for feature engineering
        numeric_columns = [
            'temperature_2m', 'relative_humidity_2m', 'pressure_msl',
            'wind_speed_10m', 'wind_direction_10m', 'cloud_cover', 'precipitation'
        ]
        
        # Filter to only existing columns
        existing_numeric_columns = [col for col in numeric_columns 
                                   if col in df_features.columns]
        
        # Apply feature engineering steps
        df_features = self.create_time_features(df_features)
        df_features = self.create_rolling_features(df_features, existing_numeric_columns)
        df_features = self.create_lag_features(df_features, existing_numeric_columns)
        df_features = self.calculate_atmospheric_indices(df_features)
        df_features = self.create_interaction_features(df_features)
        
        # Remove rows with too many missing values (from rolling/lag features)
        initial_rows = len(df_features)
        df_features = df_features.dropna(thresh=len(df_features.columns) * 0.7)
        final_rows = len(df_features)
        
        logger.info(f"Feature engineering complete: {initial_rows} -> {final_rows} rows, "
                   f"{len(df_features.columns)} features")
        
        return df_features
    
    def save_engineered_features(self, df: pd.DataFrame, 
                                filename: str = None) -> str:
        """
        Save engineered features to file
        
        Args:
            df: DataFrame with engineered features
            filename: Optional filename
            
        Returns:
            Path to saved file
        """
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"engineered_features_{timestamp}.csv"
        
        filepath = self.processed_data_path / filename
        df.to_csv(filepath, index=False)
        
        logger.info(f"Saved engineered features to {filepath}")
        return str(filepath)


def main():
    """Main function for running feature engineering"""
    engineer = WeatherFeatureEngineer()
    
    # Generate sample weather data for testing
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-01-07', freq='H')
    
    sample_data = pd.DataFrame({
        'datetime': dates,
        'temperature_2m': np.random.normal(25, 5, len(dates)),
        'relative_humidity_2m': np.random.uniform(40, 90, len(dates)),
        'pressure_msl': np.random.normal(1013, 10, len(dates)),
        'wind_speed_10m': np.random.exponential(3, len(dates)),
        'wind_direction_10m': np.random.uniform(0, 360, len(dates)),
        'cloud_cover': np.random.uniform(0, 100, len(dates)),
        'precipitation': np.random.exponential(0.5, len(dates))
    })
    
    # Engineer features
    engineered_data = engineer.engineer_all_features(sample_data)
    
    print(f"Original data shape: {sample_data.shape}")
    print(f"Engineered data shape: {engineered_data.shape}")
    print(f"New features: {set(engineered_data.columns) - set(sample_data.columns)}")
    
    # Save the engineered features
    filepath = engineer.save_engineered_features(engineered_data, "sample_engineered_features.csv")
    print(f"Features saved to: {filepath}")


if __name__ == "__main__":
    main()