"""
Time-Series Feature Engineering

Advanced time-series features for cloud burst prediction:
- Rolling window statistics
- Rate of change features
- Lag features
- Trend features
- Seasonal decomposition
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesFeatures:
    """Generate time-series features from weather data"""
    
    def __init__(self, datetime_col: str = 'datetime'):
        """
        Initialize time-series feature engineer
        
        Args:
            datetime_col: Name of datetime column
        """
        self.datetime_col = datetime_col
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic temporal features (hour, day, month, etc.)
        
        Args:
            df: DataFrame with datetime column
            
        Returns:
            DataFrame with added temporal features
        """
        df = df.copy()
        
        # Ensure datetime column is datetime type
        if not pd.api.types.is_datetime64_any_dtype(df[self.datetime_col]):
            df[self.datetime_col] = pd.to_datetime(df[self.datetime_col])
        
        # Extract temporal components
        df['hour'] = df[self.datetime_col].dt.hour
        df['day_of_week'] = df[self.datetime_col].dt.dayofweek
        df['day_of_month'] = df[self.datetime_col].dt.day
        df['day_of_year'] = df[self.datetime_col].dt.dayofyear
        df['week_of_year'] = df[self.datetime_col].dt.isocalendar().week
        df['month'] = df[self.datetime_col].dt.month
        df['quarter'] = df[self.datetime_col].dt.quarter
        df['year'] = df[self.datetime_col].dt.year
        
        # Cyclical encoding for periodic features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Season encoding (0=Winter, 1=Spring, 2=Summer, 3=Autumn)
        df['season'] = (df['month'] % 12 // 3)
        
        # Weekend/weekday
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        logger.info(f"✓ Added 19 temporal features")
        
        return df
    
    def add_rolling_statistics(self, df: pd.DataFrame, 
                               columns: List[str],
                               windows: List[int] = [3, 6, 12, 24]) -> pd.DataFrame:
        """
        Add rolling window statistics
        
        Args:
            df: DataFrame with time-series data
            columns: Columns to calculate rolling stats for
            windows: Window sizes in hours
            
        Returns:
            DataFrame with rolling statistics
        """
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping")
                continue
            
            for window in windows:
                # Rolling mean
                df[f'{col}_rolling_mean_{window}h'] = \
                    df[col].rolling(window=window, min_periods=1).mean()
                
                # Rolling std
                df[f'{col}_rolling_std_{window}h'] = \
                    df[col].rolling(window=window, min_periods=1).std()
                
                # Rolling min/max
                df[f'{col}_rolling_min_{window}h'] = \
                    df[col].rolling(window=window, min_periods=1).min()
                df[f'{col}_rolling_max_{window}h'] = \
                    df[col].rolling(window=window, min_periods=1).max()
                
                # Rolling range
                df[f'{col}_rolling_range_{window}h'] = \
                    df[f'{col}_rolling_max_{window}h'] - df[f'{col}_rolling_min_{window}h']
        
        features_added = len(columns) * len(windows) * 5
        logger.info(f"✓ Added {features_added} rolling statistics features")
        
        return df
    
    def add_rate_of_change(self, df: pd.DataFrame,
                          columns: List[str],
                          periods: List[int] = [1, 3, 6, 12]) -> pd.DataFrame:
        """
        Add rate of change features (derivatives)
        
        Args:
            df: DataFrame with time-series data
            columns: Columns to calculate rate of change for
            periods: Time periods for rate calculation
            
        Returns:
            DataFrame with rate of change features
        """
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping")
                continue
            
            for period in periods:
                # Absolute change
                df[f'{col}_change_{period}h'] = df[col].diff(periods=period)
                
                # Percentage change
                df[f'{col}_pct_change_{period}h'] = df[col].pct_change(periods=period) * 100
                
                # Acceleration (second derivative)
                if period == 1:
                    df[f'{col}_acceleration'] = df[f'{col}_change_{period}h'].diff()
        
        features_added = len(columns) * (len(periods) * 2 + 1)
        logger.info(f"✓ Added {features_added} rate of change features")
        
        return df
    
    def add_lag_features(self, df: pd.DataFrame,
                        columns: List[str],
                        lags: List[int] = [1, 3, 6, 12, 24]) -> pd.DataFrame:
        """
        Add lagged features
        
        Args:
            df: DataFrame with time-series data
            columns: Columns to create lags for
            lags: Lag periods in hours
            
        Returns:
            DataFrame with lag features
        """
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping")
                continue
            
            for lag in lags:
                df[f'{col}_lag_{lag}h'] = df[col].shift(lag)
        
        features_added = len(columns) * len(lags)
        logger.info(f"✓ Added {features_added} lag features")
        
        return df
    
    def add_trend_features(self, df: pd.DataFrame,
                          columns: List[str],
                          window: int = 24) -> pd.DataFrame:
        """
        Add trend features
        
        Args:
            df: DataFrame with time-series data
            columns: Columns to calculate trends for
            window: Window size for trend calculation
            
        Returns:
            DataFrame with trend features
        """
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping")
                continue
            
            # Linear trend using rolling regression
            def calculate_trend(series):
                if len(series) < 2:
                    return 0
                x = np.arange(len(series))
                try:
                    slope = np.polyfit(x, series, 1)[0]
                    return slope
                except:
                    return 0
            
            df[f'{col}_trend_{window}h'] = \
                df[col].rolling(window=window, min_periods=2).apply(calculate_trend, raw=True)
        
        logger.info(f"✓ Added {len(columns)} trend features")
        
        return df
    
    def add_interaction_features(self, df: pd.DataFrame,
                                 column_pairs: List[tuple]) -> pd.DataFrame:
        """
        Add interaction features between variables
        
        Args:
            df: DataFrame with features
            column_pairs: List of column pairs to create interactions
            
        Returns:
            DataFrame with interaction features
        """
        df = df.copy()
        
        for col1, col2 in column_pairs:
            if col1 in df.columns and col2 in df.columns:
                # Multiplication
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                
                # Ratio (avoid division by zero)
                df[f'{col1}_div_{col2}'] = df[col1] / (df[col2].replace(0, np.nan))
        
        features_added = len(column_pairs) * 2
        logger.info(f"✓ Added {features_added} interaction features")
        
        return df
    
    def add_statistical_features(self, df: pd.DataFrame,
                                 columns: List[str],
                                 window: int = 24) -> pd.DataFrame:
        """
        Add statistical features (skewness, kurtosis, etc.)
        
        Args:
            df: DataFrame with time-series data
            columns: Columns to calculate statistics for
            window: Window size
            
        Returns:
            DataFrame with statistical features
        """
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping")
                continue
            
            # Skewness
            df[f'{col}_skew_{window}h'] = \
                df[col].rolling(window=window, min_periods=3).skew()
            
            # Kurtosis
            df[f'{col}_kurt_{window}h'] = \
                df[col].rolling(window=window, min_periods=4).kurt()
            
            # Coefficient of variation
            rolling_mean = df[col].rolling(window=window, min_periods=1).mean()
            rolling_std = df[col].rolling(window=window, min_periods=1).std()
            df[f'{col}_cv_{window}h'] = (rolling_std / rolling_mean.replace(0, np.nan)) * 100
        
        features_added = len(columns) * 3
        logger.info(f"✓ Added {features_added} statistical features")
        
        return df
    
    def add_all_timeseries_features(self, df: pd.DataFrame,
                                    numeric_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Add all time-series features
        
        Args:
            df: DataFrame with time-series weather data
            numeric_columns: List of numeric columns to process (auto-detect if None)
            
        Returns:
            DataFrame with all time-series features
        """
        logger.info("="*80)
        logger.info("ADDING TIME-SERIES FEATURES")
        logger.info("="*80)
        
        # Auto-detect numeric columns if not provided
        if numeric_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude ID and year columns
            numeric_columns = [col for col in numeric_columns 
                             if col not in ['id', 'year', 'month', 'day', 'hour']]
        
        logger.info(f"\nProcessing {len(numeric_columns)} numeric columns")
        
        # Add temporal features
        df = self.add_temporal_features(df)
        
        # Add rolling statistics
        df = self.add_rolling_statistics(df, numeric_columns, windows=[3, 6, 12, 24])
        
        # Add rate of change
        df = self.add_rate_of_change(df, numeric_columns, periods=[1, 3, 6, 12])
        
        # Add lag features
        df = self.add_lag_features(df, numeric_columns, lags=[1, 3, 6, 12, 24])
        
        # Add trend features
        df = self.add_trend_features(df, numeric_columns, window=24)
        
        # Add statistical features
        df = self.add_statistical_features(df, numeric_columns, window=24)
        
        # Add important interactions
        if 'temperature_2m' in df.columns and 'relative_humidity_2m' in df.columns:
            interaction_pairs = [
                ('temperature_2m', 'relative_humidity_2m'),
                ('temperature_2m', 'pressure_msl'),
                ('precipitation', 'wind_speed_10m'),
            ]
            df = self.add_interaction_features(df, interaction_pairs)
        
        # Fill NaN values created by rolling/lag operations
        # Forward fill first, then backward fill, then fill with 0
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        logger.info(f"\n✓ Total features: {len(df.columns)}")
        logger.info("="*80)
        
        return df


def create_sequences_for_lstm(df: pd.DataFrame, 
                               feature_columns: List[str],
                               target_column: str,
                               sequence_length: int = 24,
                               forecast_horizon: int = 6) -> tuple:
    """
    Create sequences for LSTM model
    
    Args:
        df: DataFrame with features
        feature_columns: List of feature column names
        target_column: Target column name
        sequence_length: Length of input sequence (hours)
        forecast_horizon: Hours ahead to predict
        
    Returns:
        Tuple of (X, y) where X is 3D array (samples, timesteps, features)
    """
    logger.info(f"Creating LSTM sequences: length={sequence_length}, horizon={forecast_horizon}")
    
    # Extract features and target
    X_data = df[feature_columns].values
    y_data = df[target_column].values
    
    X_sequences = []
    y_sequences = []
    
    for i in range(len(df) - sequence_length - forecast_horizon):
        # Input sequence
        X_seq = X_data[i:i + sequence_length]
        
        # Target (forecast_horizon hours ahead)
        y_seq = y_data[i + sequence_length + forecast_horizon]
        
        X_sequences.append(X_seq)
        y_sequences.append(y_seq)
    
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    
    logger.info(f"✓ Created {len(X_sequences)} sequences")
    logger.info(f"  X shape: {X_sequences.shape}")
    logger.info(f"  y shape: {y_sequences.shape}")
    
    return X_sequences, y_sequences


if __name__ == "__main__":
    # Test time-series features
    print("="*80)
    print("TIME-SERIES FEATURES TEST")
    print("="*80)
    
    # Create sample data
    dates = pd.date_range('2025-01-01', periods=168, freq='1H')  # 1 week
    df_test = pd.DataFrame({
        'datetime': dates,
        'temperature_2m': 25 + 5 * np.sin(np.arange(168) * 2 * np.pi / 24) + np.random.randn(168),
        'precipitation': np.abs(np.random.randn(168) * 2),
        'relative_humidity_2m': 70 + 10 * np.random.randn(168)
    })
    
    print(f"\nSample data shape: {df_test.shape}")
    print(f"Columns: {df_test.columns.tolist()}")
    
    # Generate features
    ts_engineer = TimeSeriesFeatures()
    df_features = ts_engineer.add_all_timeseries_features(df_test)
    
    print(f"\nFinal shape: {df_features.shape}")
    print(f"Features added: {len(df_features.columns) - len(df_test.columns)}")
    print(f"\nSample features: {df_features.columns.tolist()[:10]}")
    
    # Test LSTM sequence creation
    feature_cols = [col for col in df_features.columns if col != 'datetime']
    df_features['cloud_burst'] = (df_features['precipitation'] > 3).astype(int)
    
    X, y = create_sequences_for_lstm(
        df_features,
        feature_cols[:10],  # Use first 10 features for demo
        'cloud_burst',
        sequence_length=12,
        forecast_horizon=3
    )
    
    print("\n" + "="*80)
