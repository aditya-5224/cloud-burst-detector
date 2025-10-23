"""
Rainfall Intensity Regression Model

Predicts rainfall intensity (mm/hour) instead of just binary cloudburst/no-cloudburst.
This addresses the critical gap: Perplexity analysis showed only 2.5% accuracy 
in intensity prediction.

Target: RMSE < 20mm/h for accurate intensity forecasts
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Tuple
import joblib

from src.data.events_database import CloudBurstEventsDB
from src.data.historical_weather import HistoricalWeatherCollector
from src.features.atmospheric_indices import add_atmospheric_indices_to_dataframe
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntensityRegressionTrainer:
    """Train regression model to predict rainfall intensity"""
    
    def __init__(self):
        self.events_db = CloudBurstEventsDB()
        self.weather_collector = HistoricalWeatherCollector()
        self.model = None
        
    def collect_training_data(self, hours_before: int = 24, hours_after: int = 6) -> pd.DataFrame:
        """
        Collect training data with intensity labels
        
        Args:
            hours_before: Hours of data before event
            hours_after: Hours after event
            
        Returns:
            DataFrame with weather features and intensity labels
        """
        logger.info("=" * 70)
        logger.info("COLLECTING DATA FOR INTENSITY REGRESSION")
        logger.info("=" * 70)
        
        all_data = []
        events = self.events_db.get_all_events()
        
        logger.info(f"\n📊 Found {len(events)} historical cloud burst events")
        
        for i, event in enumerate(events, 1):
            logger.info(f"\n[{i}/{len(events)}] Processing: {event['event_id']} - {event['location']}")
            logger.info(f"   Date: {event['date']}, Intensity: {event['intensity_mm_per_hour']:.1f} mm/h")
            
            try:
                # Fetch weather data
                event_date = datetime.strptime(event['date'], '%Y-%m-%d')
                start_date = (event_date - timedelta(hours=hours_before)).strftime('%Y-%m-%d')
                end_date = (event_date + timedelta(hours=hours_after)).strftime('%Y-%m-%d')
                
                weather_df = self.weather_collector.fetch_historical_open_meteo(
                    latitude=event['latitude'],
                    longitude=event['longitude'],
                    start_date=start_date,
                    end_date=end_date
                )
                
                if weather_df is None or len(weather_df) == 0:
                    logger.warning(f"   ⚠️  No weather data available")
                    continue
                
                logger.info(f"   ✅ Fetched {len(weather_df)} hourly records")
                
                # Label intensity: actual event intensity for event day, 0 for other days
                weather_df['intensity_mm_per_hour'] = 0.0
                event_day_mask = weather_df['time'].dt.date == event_date.date()
                weather_df.loc[event_day_mask, 'intensity_mm_per_hour'] = event['intensity_mm_per_hour']
                
                # Add event metadata
                weather_df['event_id'] = event['event_id']
                weather_df['is_cloudburst_day'] = event_day_mask.astype(int)
                
                all_data.append(weather_df)
                
            except Exception as e:
                logger.error(f"   ❌ Error processing event: {e}")
                continue
        
        if not all_data:
            logger.error("No training data collected!")
            return None
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        logger.info("\n" + "=" * 70)
        logger.info("DATA COLLECTION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total records: {len(combined_df)}")
        logger.info(f"Cloudburst hours (>50mm/h): {(combined_df['intensity_mm_per_hour'] > 50).sum()}")
        logger.info(f"Heavy rain hours (20-50mm/h): {((combined_df['intensity_mm_per_hour'] >= 20) & (combined_df['intensity_mm_per_hour'] <= 50)).sum()}")
        logger.info(f"Normal hours (<20mm/h): {(combined_df['intensity_mm_per_hour'] < 20).sum()}")
        logger.info(f"Max intensity: {combined_df['intensity_mm_per_hour'].max():.1f} mm/h")
        logger.info(f"Mean intensity: {combined_df['intensity_mm_per_hour'].mean():.2f} mm/h")
        
        # Calculate atmospheric stability indices
        logger.info("\n🔬 Calculating atmospheric stability indices...")
        try:
            combined_df = add_atmospheric_indices_to_dataframe(combined_df)
            logger.info("✅ Atmospheric indices calculated successfully")
        except Exception as e:
            logger.warning(f"⚠️  Could not calculate all atmospheric indices: {e}")
        
        return combined_df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for intensity regression
        
        Args:
            df: DataFrame with weather data
            
        Returns:
            X (features), y (intensity labels)
        """
        logger.info("\n📊 Preparing features for intensity regression...")
        
        # Feature columns
        feature_cols = [
            'temperature_2m',
            'relative_humidity_2m',
            'precipitation',
            'pressure_msl',
            'cloud_cover_total',
            'wind_speed_10m',
            'wind_direction_10m',
            'cape',
            'lifted_index',
            'k_index',
            'total_totals',
            'showalter_index'
        ]
        
        # Handle missing columns
        for col in feature_cols:
            if col not in df.columns:
                if col == 'cloud_cover_total' and 'cloud_cover' in df.columns:
                    df['cloud_cover_total'] = df['cloud_cover']
                else:
                    df[col] = 0
        
        X = df[feature_cols].copy()
        y = df['intensity_mm_per_hour'].copy()
        
        # Fill missing values
        X = X.fillna(0)
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Labels shape: {y.shape}")
        logger.info(f"Features: {list(X.columns)}")
        logger.info(f"Intensity range: {y.min():.1f} - {y.max():.1f} mm/h")
        
        return X, y
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, model_type: str = 'random_forest') -> Dict:
        """
        Train intensity regression model
        
        Args:
            X: Feature matrix
            y: Intensity labels
            model_type: 'random_forest' or 'gradient_boosting'
            
        Returns:
            Dictionary with training results
        """
        logger.info("\n" + "=" * 70)
        logger.info(f"TRAINING {model_type.upper().replace('_', ' ')} REGRESSION MODEL")
        logger.info("=" * 70)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        logger.info(f"\nData split:")
        logger.info(f"  Training set: {len(X_train)} samples")
        logger.info(f"  Test set: {len(X_test)} samples")
        logger.info(f"  Training intensity range: {y_train.min():.1f} - {y_train.max():.1f} mm/h")
        logger.info(f"  Test intensity range: {y_test.min():.1f} - {y_test.max():.1f} mm/h")
        
        # Train model
        logger.info(f"\n🚀 Training {model_type} model...")
        
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model.fit(X_train, y_train)
        logger.info("✅ Model training complete")
        
        # Evaluate
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        logger.info("\n" + "=" * 70)
        logger.info("MODEL PERFORMANCE")
        logger.info("=" * 70)
        logger.info(f"\n📊 Training Set:")
        logger.info(f"   RMSE: {train_rmse:.2f} mm/h")
        logger.info(f"   MAE:  {train_mae:.2f} mm/h")
        logger.info(f"   R²:   {train_r2:.4f}")
        
        logger.info(f"\n📊 Test Set:")
        logger.info(f"   RMSE: {test_rmse:.2f} mm/h")
        logger.info(f"   MAE:  {test_mae:.2f} mm/h")
        logger.info(f"   R²:   {test_r2:.4f}")
        
        # Target assessment
        logger.info(f"\n🎯 Target Assessment:")
        if test_rmse < 20:
            logger.info(f"   ✅ EXCELLENT: RMSE {test_rmse:.2f} < 20 mm/h (TARGET MET!)")
        elif test_rmse < 30:
            logger.info(f"   ✅ GOOD: RMSE {test_rmse:.2f} < 30 mm/h")
        elif test_rmse < 50:
            logger.info(f"   ⚠️  MODERATE: RMSE {test_rmse:.2f} < 50 mm/h (needs improvement)")
        else:
            logger.info(f"   ❌ POOR: RMSE {test_rmse:.2f} > 50 mm/h (significant improvement needed)")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("\n📈 Top 5 Most Important Features for Intensity:")
            for idx, row in feature_importance.head().iterrows():
                logger.info(f"   {row['feature']}: {row['importance']:.4f}")
        
        return {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'predictions_test': y_pred_test,
            'actual_test': y_test,
            'feature_importance': feature_importance if hasattr(self.model, 'feature_importances_') else None
        }
    
    def save_model(self, output_dir: str = "./models/trained"):
        """Save trained intensity regression model"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save model
        model_file = output_path / "intensity_regression_model.pkl"
        joblib.dump(self.model, model_file)
        logger.info(f"\n✅ Model saved to: {model_file}")
        
        # Also save with timestamp
        model_file_backup = output_path.parent / f"intensity_regression_model_{timestamp}.joblib"
        joblib.dump(self.model, model_file_backup)
        logger.info(f"✅ Backup saved to: {model_file_backup}")
        
        return model_file


def main():
    """Main training pipeline for intensity regression"""
    
    print("\n" + "="*70)
    print("RAINFALL INTENSITY REGRESSION MODEL TRAINING")
    print("="*70)
    
    trainer = IntensityRegressionTrainer()
    
    # Step 1: Collect training data
    print("\n📥 STEP 1: Collecting training data...")
    df = trainer.collect_training_data(hours_before=48, hours_after=12)
    
    if df is None or len(df) == 0:
        print("❌ No training data available. Exiting.")
        return
    
    # Step 2: Prepare features
    print("\n🔧 STEP 2: Preparing features...")
    X, y = trainer.prepare_features(df)
    
    # Step 3: Train Random Forest model
    print("\n🎓 STEP 3: Training Random Forest Regression Model...")
    rf_results = trainer.train_model(X, y, model_type='random_forest')
    
    # Step 4: Train Gradient Boosting model for comparison
    print("\n🎓 STEP 4: Training Gradient Boosting Model (for comparison)...")
    gb_results = trainer.train_model(X, y, model_type='gradient_boosting')
    
    # Step 5: Save best model
    print("\n💾 STEP 5: Saving model...")
    
    # Choose best model based on test RMSE
    if rf_results['test_rmse'] < gb_results['test_rmse']:
        print(f"\n✅ Random Forest performs better (RMSE: {rf_results['test_rmse']:.2f} vs {gb_results['test_rmse']:.2f})")
        trainer.model = RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        )
        trainer.model.fit(X, y)  # Retrain on full dataset
        best_rmse = rf_results['test_rmse']
    else:
        print(f"\n✅ Gradient Boosting performs better (RMSE: {gb_results['test_rmse']:.2f} vs {rf_results['test_rmse']:.2f})")
        trainer.model = GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            min_samples_split=5, min_samples_leaf=2, random_state=42
        )
        trainer.model.fit(X, y)
        best_rmse = gb_results['test_rmse']
    
    model_path = trainer.save_model()
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📊 Best Model Performance:")
    print(f"   Test RMSE: {best_rmse:.2f} mm/h")
    print(f"   Target: < 20 mm/h")
    
    if best_rmse < 20:
        print(f"\n🎉 SUCCESS: Target achieved! RMSE = {best_rmse:.2f} mm/h")
    else:
        print(f"\n⚠️  Target not met. RMSE = {best_rmse:.2f} mm/h (need < 20 mm/h)")
        print(f"   Consider: More training data, feature engineering, or ensemble methods")
    
    print(f"\n💾 Model saved to: {model_path}")
    print(f"\n✅ Intensity regression model ready for predictions!")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
