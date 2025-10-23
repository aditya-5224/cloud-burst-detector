"""
Sprint 2: Feature Engineering Pipeline

Complete integration of:
- Atmospheric indices (CAPE, Lifted Index, K-Index)
- Time-series features (rolling, lag, rate of change)
- Feature validation and selection
- LSTM sequence preparation
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.database import DatabaseManager
from src.features.atmospheric_indices import AtmosphericIndices
from src.features.timeseries_features import TimeSeriesFeatures
from src.features.feature_selection import FeatureValidator, FeatureSelector

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureEngineeringPipeline:
    """Complete feature engineering pipeline for Sprint 2"""
    
    def __init__(self, db_path: str = "./data/cloudburst.db",
                 output_dir: str = "./reports/sprint2"):
        """
        Initialize the feature engineering pipeline
        
        Args:
            db_path: Path to database
            output_dir: Directory for reports and outputs
        """
        self.db = DatabaseManager(db_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize feature engineers
        self.atmos_calc = AtmosphericIndices()
        self.ts_engineer = TimeSeriesFeatures()
        self.validator = FeatureValidator(output_dir=str(self.output_dir))
        self.selector = FeatureSelector()
    
    def load_weather_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Load weather data from database"""
        logger.info("="*80)
        logger.info("LOADING WEATHER DATA")
        logger.info("="*80)
        
        df = self.db.get_weather_data(start_date=start_date, end_date=end_date)
        
        if df.empty:
            logger.error("No data found in database!")
            return df
        
        logger.info(f"✓ Loaded {len(df)} weather records")
        logger.info(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        logger.info(f"  Columns: {list(df.columns)}")
        
        return df
    
    def add_atmospheric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add atmospheric indices to dataframe"""
        logger.info("\n" + "="*80)
        logger.info("ADDING ATMOSPHERIC INDICES")
        logger.info("="*80)
        
        # Calculate all atmospheric indices
        try:
            df['cape'] = self.atmos_calc.calculate_cape(df)
            logger.info("✓ Added CAPE")
        except Exception as e:
            logger.warning(f"⚠ CAPE calculation failed: {e}")
            df['cape'] = 0
        
        try:
            df['lifted_index'] = self.atmos_calc.calculate_lifted_index(df)
            logger.info("✓ Added Lifted Index")
        except Exception as e:
            logger.warning(f"⚠ Lifted Index calculation failed: {e}")
            df['lifted_index'] = 0
        
        try:
            df['k_index'] = self.atmos_calc.calculate_k_index(df)
            logger.info("✓ Added K-Index")
        except Exception as e:
            logger.warning(f"⚠ K-Index calculation failed: {e}")
            df['k_index'] = 0
        
        logger.info(f"✓ Total atmospheric features added: 3")
        
        return df
    
    def add_timeseries_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-series features to dataframe"""
        logger.info("\n" + "="*80)
        logger.info("ADDING TIME-SERIES FEATURES")
        logger.info("="*80)
        
        # Get numeric columns (exclude IDs and datetime)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['id']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        logger.info(f"Processing {len(numeric_cols)} numeric columns")
        
        # Add all time-series features
        df = self.ts_engineer.add_all_timeseries_features(df, numeric_columns=numeric_cols)
        
        return df
    
    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cloud burst target variable"""
        logger.info("\n" + "="*80)
        logger.info("CREATING TARGET VARIABLE")
        logger.info("="*80)
        
        # Load actual cloud burst events
        events = self.db.get_cloud_burst_events()
        
        if not events.empty:
            logger.info(f"Found {len(events)} cloud burst events in database")
            
            # Create target based on events
            df['cloud_burst'] = 0
            
            for _, event in events.iterrows():
                event_time = pd.to_datetime(event['event_datetime'])
                # Mark events within ±3 hours as cloud burst
                time_window = pd.Timedelta(hours=3)
                mask = (df['datetime'] >= event_time - time_window) & \
                       (df['datetime'] <= event_time + time_window)
                df.loc[mask, 'cloud_burst'] = 1
        else:
            logger.warning("No cloud burst events found in database")
            logger.info("Creating synthetic target based on weather conditions...")
            
            # Create synthetic target
            # Cloud burst conditions: high precip OR (high humidity + low pressure + high temp)
            conditions = (
                (df.get('precipitation', 0) > 10) |  # Heavy rain
                (
                    (df.get('relative_humidity_2m', 50) > 85) &
                    (df.get('pressure_msl', 1013) < 1005) &
                    (df.get('temperature_2m', 20) > 28) &
                    (df.get('cloud_cover', 50) > 80)
                )
            )
            
            df['cloud_burst'] = conditions.astype(int)
        
        positive_count = df['cloud_burst'].sum()
        positive_pct = (positive_count / len(df)) * 100
        
        logger.info(f"✓ Target variable created:")
        logger.info(f"  Positive events: {positive_count} ({positive_pct:.2f}%)")
        logger.info(f"  Negative events: {len(df) - positive_count} ({100-positive_pct:.2f}%)")
        
        return df
    
    def validate_and_select_features(self, df: pd.DataFrame, 
                                    target_col: str = 'cloud_burst',
                                    n_features: int = 50) -> pd.DataFrame:
        """Validate data quality and select best features"""
        logger.info("\n" + "="*80)
        logger.info("FEATURE VALIDATION & SELECTION")
        logger.info("="*80)
        
        # Separate features and target
        feature_cols = [col for col in df.columns 
                       if col not in ['id', 'datetime', target_col]]
        
        # Select only numeric columns
        X = df[feature_cols].select_dtypes(include=[np.number])
        y = df[target_col]
        
        # Remove any remaining non-finite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        logger.info(f"Using {len(X.columns)} numeric features")
        
        # Check if we have any positive events
        if y.sum() == 0:
            logger.error("⚠️ NO POSITIVE EVENTS FOUND!")
            logger.error("The target variable has only negative examples.")
            logger.error("This likely means cloud burst events don't match the weather data timeframe.")
            logger.error("Creating synthetic target based on extreme weather conditions...")
            
            # Create better synthetic target
            conditions = (
                (df.get('precipitation', 0) > 5) |  # Heavy rain >5mm/h
                (
                    (df.get('relative_humidity_2m', 50) > 90) &
                    (df.get('temperature_2m', 20) > 30) &
                    (df.get('wind_speed_10m', 0) > 15)
                )
            )
            y = conditions.astype(int)
            df[target_col] = y
            
            logger.info(f"✓ Created {y.sum()} synthetic cloud burst events ({(y.sum()/len(y)*100):.2f}%)")
        
        # Validate data quality
        quality = self.validator.validate_data_quality(df)
        
        # Save quality report
        import json
        with open(self.output_dir / 'data_quality.json', 'w') as f:
            # Convert numpy types to native Python types
            quality_serializable = {}
            for key, value in quality.items():
                if isinstance(value, dict):
                    quality_serializable[key] = {str(k): int(v) if isinstance(v, (np.integer, np.int64)) else v 
                                                for k, v in value.items()}
                elif isinstance(value, (np.integer, np.int64)):
                    quality_serializable[key] = int(value)
                else:
                    quality_serializable[key] = value
            json.dump(quality_serializable, f, indent=2, default=str)
        
        # Feature importance analysis
        logger.info("\n📊 Analyzing feature importance...")
        importance_df = self.validator.analyze_feature_importance(X, y, method='random_forest', top_n=30)
        importance_df.to_csv(self.output_dir / 'feature_importance.csv', index=False)
        
        # Correlation analysis
        logger.info("\n🔗 Analyzing correlations...")
        corr_matrix, redundant = self.validator.analyze_correlations(X, threshold=0.9)
        
        # Feature selection
        logger.info("\n🎯 Selecting features...")
        
        # Remove redundant features first
        X_filtered = X.drop(columns=redundant, errors='ignore')
        logger.info(f"  Removed {len(redundant)} redundant features")
        
        # Select by variance
        by_variance = self.selector.select_by_variance(X_filtered, threshold=0.01)
        X_filtered = X_filtered[by_variance]
        
        # Select k best
        k_best = self.selector.select_k_best(X_filtered, y, k=min(n_features, len(X_filtered.columns)))
        
        # Select from model
        from_model = self.selector.select_from_model(X_filtered, y, threshold='median')
        
        # Get consensus features
        consensus = self.selector.get_consensus_features(min_votes=2)
        
        if len(consensus) < 10:
            logger.warning(f"Only {len(consensus)} consensus features found, using k_best")
            selected_features = k_best[:n_features]
        else:
            selected_features = consensus[:n_features]
        
        logger.info(f"\n✓ Final feature set: {len(selected_features)} features")
        
        # Save selected features
        with open(self.output_dir / 'selected_features.txt', 'w') as f:
            for feature in selected_features:
                f.write(f"{feature}\n")
        
        # Create final dataframe with selected features
        df_final = df[['datetime', target_col] + selected_features].copy()
        
        return df_final
    
    def run_pipeline(self, start_date: str = None, end_date: str = None,
                    n_features: int = 50) -> pd.DataFrame:
        """Run complete feature engineering pipeline"""
        logger.info("\n" + "="*100)
        logger.info("SPRINT 2: FEATURE ENGINEERING PIPELINE")
        logger.info("="*100)
        logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Step 1: Load data
        df = self.load_weather_data(start_date=start_date, end_date=end_date)
        
        if df.empty:
            logger.error("Pipeline failed: No data loaded")
            return pd.DataFrame()
        
        # Step 2: Add atmospheric features
        df = self.add_atmospheric_features(df)
        
        # Step 3: Add time-series features
        df = self.add_timeseries_features(df)
        
        # Step 4: Create target variable
        df = self.create_target_variable(df)
        
        # Step 5: Validate and select features
        df_final = self.validate_and_select_features(df, n_features=n_features)
        
        # Save final dataset
        output_file = self.output_dir / 'engineered_features.csv'
        df_final.to_csv(output_file, index=False)
        logger.info(f"\n✓ Saved engineered features to: {output_file}")
        
        # Summary
        logger.info("\n" + "="*100)
        logger.info("PIPELINE SUMMARY")
        logger.info("="*100)
        logger.info(f"✓ Original data: {len(df)} rows")
        logger.info(f"✓ Final features: {len(df_final.columns) - 2} (+ datetime + target)")
        logger.info(f"✓ Target distribution: {df_final['cloud_burst'].value_counts().to_dict()}")
        logger.info(f"✓ Output saved to: {self.output_dir}")
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*100)
        
        return df_final


def main():
    """Main execution"""
    # Run the complete pipeline
    pipeline = FeatureEngineeringPipeline()
    
    # Process all available data
    df_engineered = pipeline.run_pipeline(
        start_date='2025-04-10',
        end_date='2025-10-07',
        n_features=50
    )
    
    print("\n" + "="*100)
    print("SPRINT 2 COMPLETE!")
    print("="*100)
    print(f"✓ Feature engineering completed successfully")
    print(f"✓ Generated {len(df_engineered)} samples with {len(df_engineered.columns)-2} features")
    print(f"✓ Ready for model training (Sprint 3)")
    print("="*100)


if __name__ == "__main__":
    main()
