"""
Train Model with Historical Cloud Burst Events

This script trains your ML model using real historical cloud burst events
from the events database, improving prediction accuracy.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import List, Dict
import joblib

from src.data.events_database import CloudBurstEventsDB
from src.data.historical_weather import HistoricalWeatherCollector
from src.features.atmospheric_indices import add_atmospheric_indices_to_dataframe
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, make_scorer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HistoricalDataTrainer:
    """Train models using real historical cloud burst events"""
    
    def __init__(self):
        self.events_db = CloudBurstEventsDB()
        self.weather_collector = HistoricalWeatherCollector()
        self.model = None
        
    def collect_training_data(self, hours_before: int = 24, hours_after: int = 6) -> pd.DataFrame:
        """
        Collect training data from all historical events
        
        Args:
            hours_before: Hours of data to collect before event
            hours_after: Hours of data to collect after event
            
        Returns:
            DataFrame with weather features and labels
        """
        logger.info("=" * 70)
        logger.info("COLLECTING TRAINING DATA FROM HISTORICAL EVENTS")
        logger.info("=" * 70)
        
        all_data = []
        events = self.events_db.get_all_events()
        
        logger.info(f"\n📊 Found {len(events)} historical cloud burst events")
        
        for i, event in enumerate(events, 1):
            logger.info(f"\n[{i}/{len(events)}] Processing: {event['event_id']} - {event['location']}")
            logger.info(f"   Date: {event['date']}, Rainfall: {event['rainfall_mm']}mm")
            
            try:
                # Fetch weather data around event
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
                
                # Label data: 1 for cloud burst day, 0 for before/after
                weather_df['cloud_burst_event'] = 0
                event_day_mask = weather_df['time'].dt.date == event_date.date()
                weather_df.loc[event_day_mask, 'cloud_burst_event'] = 1
                
                # Add event metadata
                weather_df['event_id'] = event['event_id']
                weather_df['event_intensity'] = event['intensity_mm_per_hour']
                
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
        logger.info(f"Cloud burst records: {combined_df['cloud_burst_event'].sum()}")
        logger.info(f"Normal records: {(combined_df['cloud_burst_event'] == 0).sum()}")
        logger.info(f"Cloud burst percentage: {combined_df['cloud_burst_event'].mean()*100:.1f}%")
        
        # Calculate atmospheric stability indices
        logger.info("\n🔬 Calculating atmospheric stability indices...")
        try:
            combined_df = add_atmospheric_indices_to_dataframe(combined_df)
            logger.info("✅ Atmospheric indices calculated successfully")
        except Exception as e:
            logger.warning(f"⚠️  Could not calculate all atmospheric indices: {e}")
            logger.warning("   Continuing with available features...")
        
        return combined_df
    
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """
        Prepare features for training
        
        Args:
            df: DataFrame with weather data
            
        Returns:
            X (features), y (labels)
        """
        logger.info("\n📊 Preparing features for training...")
        
        # Select feature columns (including atmospheric indices)
        feature_cols = [
            'temperature_2m',
            'relative_humidity_2m',
            'precipitation',
            'pressure_msl',
            'cloud_cover_total',
            'wind_speed_10m',
            'wind_direction_10m',
            'cape',  # Convective Available Potential Energy from Open-Meteo
            'lifted_index',  # Atmospheric stability
            'k_index',  # Thunderstorm potential
            'total_totals',  # Severe weather index
            'showalter_index'  # Instability index
        ]
        
        # Handle missing columns and normalize cloud_cover naming
        for col in feature_cols:
            if col not in df.columns:
                # Handle cloud_cover vs cloud_cover_total naming
                if col == 'cloud_cover_total' and 'cloud_cover' in df.columns:
                    df['cloud_cover_total'] = df['cloud_cover']
                else:
                    df[col] = 0
        
        X = df[feature_cols].copy()
        y = df['cloud_burst_event'].copy()
        
        # Fill missing values
        X = X.fillna(0)
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Labels shape: {y.shape}")
        logger.info(f"Features: {list(X.columns)}")
        
        return X, y
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Train Random Forest model
        
        Args:
            X: Feature matrix
            y: Labels
            
        Returns:
            Dictionary with training results
        """
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING RANDOM FOREST MODEL")
        logger.info("=" * 70)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"\nData split:")
        logger.info(f"  Training set: {len(X_train)} samples")
        logger.info(f"  Test set: {len(X_test)} samples")
        logger.info(f"  Training positives: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
        logger.info(f"  Test positives: {y_test.sum()} ({y_test.mean()*100:.1f}%)")
        
        # Train model with better parameters for imbalanced data
        logger.info("\n🚀 Training model...")
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',  # Handle imbalanced data
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        logger.info("✅ Model training complete")
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        logger.info("\n" + "=" * 70)
        logger.info("MODEL PERFORMANCE")
        logger.info("=" * 70)
        logger.info(f"Accuracy:  {accuracy*100:.2f}%")
        logger.info(f"Precision: {precision*100:.2f}%")
        logger.info(f"Recall:    {recall*100:.2f}%")
        logger.info(f"F1-Score:  {f1*100:.2f}%")
        
        logger.info("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"  True Negatives:  {cm[0,0]}")
        logger.info(f"  False Positives: {cm[0,1]}")
        logger.info(f"  False Negatives: {cm[1,0]}")
        logger.info(f"  True Positives:  {cm[1,1]}")
        
        logger.info("\nClassification Report:")
        logger.info("\n" + classification_report(y_test, y_pred, 
                                                 target_names=['Normal', 'Cloud Burst']))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("\nTop 5 Most Important Features:")
        for idx, row in feature_importance.head().iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'feature_importance': feature_importance
        }
    
    def perform_cross_validation(self, X: pd.DataFrame, y: pd.Series, n_folds: int = 5) -> Dict:
        """
        Perform K-Fold Cross-Validation to assess model stability and generalization
        
        Args:
            X: Feature matrix
            y: Labels
            n_folds: Number of folds (default: 5)
            
        Returns:
            Dictionary with cross-validation results
        """
        logger.info("\n" + "=" * 70)
        logger.info("K-FOLD CROSS-VALIDATION")
        logger.info("=" * 70)
        logger.info(f"\n🔄 Performing {n_folds}-Fold Stratified Cross-Validation...")
        logger.info(f"   This tests the model on {n_folds} different train/test splits")
        logger.info(f"   to measure stability and prevent overfitting\n")
        
        # Create stratified k-fold splitter
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Create model with same parameters
        cv_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # Define scoring metrics
        scoring = {
            'accuracy': 'accuracy',
            'precision': make_scorer(precision_score, zero_division=0),
            'recall': make_scorer(recall_score, zero_division=0),
            'f1': make_scorer(f1_score, zero_division=0)
        }
        
        # Perform cross-validation
        logger.info("⏳ Running cross-validation (this may take a minute)...")
        cv_results = cross_validate(
            cv_model, X, y,
            cv=skf,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )
        
        # Calculate statistics
        results = {
            'accuracy': {
                'mean': cv_results['test_accuracy'].mean(),
                'std': cv_results['test_accuracy'].std(),
                'scores': cv_results['test_accuracy'].tolist()
            },
            'precision': {
                'mean': cv_results['test_precision'].mean(),
                'std': cv_results['test_precision'].std(),
                'scores': cv_results['test_precision'].tolist()
            },
            'recall': {
                'mean': cv_results['test_recall'].mean(),
                'std': cv_results['test_recall'].std(),
                'scores': cv_results['test_recall'].tolist()
            },
            'f1_score': {
                'mean': cv_results['test_f1'].mean(),
                'std': cv_results['test_f1'].std(),
                'scores': cv_results['test_f1'].tolist()
            }
        }
        
        # Display results
        logger.info("\n" + "=" * 70)
        logger.info("CROSS-VALIDATION RESULTS")
        logger.info("=" * 70)
        
        logger.info(f"\n📊 {n_folds}-Fold Cross-Validation Scores:")
        logger.info(f"   Accuracy:  {results['accuracy']['mean']*100:.2f}% ± {results['accuracy']['std']*100:.2f}%")
        logger.info(f"   Precision: {results['precision']['mean']*100:.2f}% ± {results['precision']['std']*100:.2f}%")
        logger.info(f"   Recall:    {results['recall']['mean']*100:.2f}% ± {results['recall']['std']*100:.2f}%")
        logger.info(f"   F1-Score:  {results['f1_score']['mean']*100:.2f}% ± {results['f1_score']['std']*100:.2f}%")
        
        logger.info(f"\n📈 Individual Fold Scores:")
        for i in range(n_folds):
            logger.info(f"   Fold {i+1}: Acc={results['accuracy']['scores'][i]*100:.1f}%, "
                       f"Prec={results['precision']['scores'][i]*100:.1f}%, "
                       f"Rec={results['recall']['scores'][i]*100:.1f}%, "
                       f"F1={results['f1_score']['scores'][i]*100:.1f}%")
        
        # Interpretation
        logger.info(f"\n🎯 Model Stability Assessment:")
        acc_std = results['accuracy']['std'] * 100
        if acc_std < 5:
            logger.info(f"   ✅ EXCELLENT: Low variance (±{acc_std:.2f}%) - Model is stable across folds")
        elif acc_std < 10:
            logger.info(f"   ✅ GOOD: Moderate variance (±{acc_std:.2f}%) - Model generalizes well")
        else:
            logger.info(f"   ⚠️  HIGH variance (±{acc_std:.2f}%) - Model may be overfitting")
        
        return results
    
    def save_model(self, output_dir: str = "./models/trained"):
        """Save trained model"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save model
        model_file = output_path / "random_forest_model.pkl"
        joblib.dump(self.model, model_file)
        logger.info(f"\n✅ Model saved to: {model_file}")
        
        # Also save with timestamp
        model_file_backup = output_path.parent / f"random_forest_model_{timestamp}.joblib"
        joblib.dump(self.model, model_file_backup)
        logger.info(f"✅ Backup saved to: {model_file_backup}")
        
        return model_file


def main():
    """Main training pipeline"""
    
    print("\n" + "="*70)
    print("CLOUD BURST PREDICTION - HISTORICAL DATA TRAINING")
    print("="*70)
    
    trainer = HistoricalDataTrainer()
    
    # Step 1: Collect training data from historical events
    print("\n📥 STEP 1: Collecting training data from historical events...")
    df = trainer.collect_training_data(hours_before=48, hours_after=12)
    
    if df is None or len(df) == 0:
        print("❌ No training data available. Exiting.")
        return
    
    # Step 2: Prepare features
    print("\n🔧 STEP 2: Preparing features...")
    X, y = trainer.prepare_features(df)
    
    # Step 3: Train model with train/test split
    print("\n🎓 STEP 3: Training model...")
    train_results = trainer.train_model(X, y)
    
    # Step 4: Perform K-Fold Cross-Validation
    print("\n🔄 STEP 4: Performing K-Fold Cross-Validation...")
    cv_results = trainer.perform_cross_validation(X, y, n_folds=5)
    
    # Step 5: Save model
    print("\n💾 STEP 5: Saving model...")
    model_path = trainer.save_model()
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📊 Final Metrics (Single Train/Test Split):")
    print(f"   Accuracy:  {train_results['accuracy']*100:.2f}%")
    print(f"   Precision: {train_results['precision']*100:.2f}%")
    print(f"   Recall:    {train_results['recall']*100:.2f}%")
    print(f"   F1-Score:  {train_results['f1_score']*100:.2f}%")
    
    print(f"\n📊 Cross-Validation Metrics ({5}-Fold):")
    print(f"   Accuracy:  {cv_results['accuracy']['mean']*100:.2f}% ± {cv_results['accuracy']['std']*100:.2f}%")
    print(f"   Precision: {cv_results['precision']['mean']*100:.2f}% ± {cv_results['precision']['std']*100:.2f}%")
    print(f"   Recall:    {cv_results['recall']['mean']*100:.2f}% ± {cv_results['recall']['std']*100:.2f}%")
    print(f"   F1-Score:  {cv_results['f1_score']['mean']*100:.2f}% ± {cv_results['f1_score']['std']*100:.2f}%")
    
    print(f"\n💾 Model saved to: {model_path}")
    print(f"\n✅ Your model is now trained with real historical cloud burst events!")
    print(f"   You can now use it in the dashboard for more accurate predictions.")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
