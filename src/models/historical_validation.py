"""
Historical Validation Module

This module validates model predictions against known historical cloud burst events.
It helps assess model accuracy and identify areas for improvement.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import joblib
from datetime import datetime, timedelta
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.historical_weather import HistoricalWeatherCollector
from src.features.engineer import FeatureEngineer

logger = logging.getLogger(__name__)


class HistoricalValidator:
    """Validate model predictions against historical cloud burst events"""
    
    def __init__(self, model_path: str = "./models/trained/random_forest_model.pkl"):
        """Initialize the validator with a trained model"""
        self.model_path = Path(model_path)
        self.model = None
        self.feature_engineer = FeatureEngineer()
        self.collector = HistoricalWeatherCollector()
        
        # Load model if it exists
        if self.model_path.exists():
            self.load_model()
        else:
            logger.warning(f"Model not found at {model_path}")
    
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"✅ Model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            self.model = None
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features from historical weather data
        
        Args:
            df: Raw weather DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        try:
            # Create features for each hour
            features_list = []
            
            for idx, row in df.iterrows():
                weather_data = {
                    'temperature': row['temperature_2m'],
                    'humidity': row['relative_humidity_2m'],
                    'precipitation': row['precipitation'],
                    'pressure': row['pressure_msl'],
                    'cloud_cover': row.get('cloud_cover', 0),
                    'wind_speed': row.get('wind_speed_10m', 0),
                    'wind_direction': row.get('wind_direction_10m', 0)
                }
                
                # Engineer features
                features = self.feature_engineer.engineer_features(weather_data)
                features['time'] = row['time']
                
                if 'during_cloudburst' in row:
                    features['actual_cloudburst'] = row['during_cloudburst']
                
                features_list.append(features)
            
            features_df = pd.DataFrame(features_list)
            logger.info(f"✅ Prepared features for {len(features_df)} records")
            
            return features_df
            
        except Exception as e:
            logger.error(f"❌ Error preparing features: {e}")
            return pd.DataFrame()
    
    def predict_on_historical(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on historical data
        
        Args:
            features_df: DataFrame with engineered features
            
        Returns:
            DataFrame with predictions added
        """
        if self.model is None:
            logger.error("No model loaded")
            return features_df
        
        try:
            # Get feature columns (exclude time and actual labels)
            exclude_cols = ['time', 'actual_cloudburst']
            feature_cols = [col for col in features_df.columns if col not in exclude_cols]
            
            X = features_df[feature_cols]
            
            # Make predictions
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)[:, 1]  # Probability of cloud burst
            
            features_df['predicted_cloudburst'] = predictions
            features_df['cloudburst_probability'] = probabilities
            
            logger.info(f"✅ Made predictions for {len(features_df)} records")
            
            return features_df
            
        except Exception as e:
            logger.error(f"❌ Error making predictions: {e}")
            return features_df
    
    def calculate_metrics(self, results_df: pd.DataFrame) -> Dict:
        """
        Calculate validation metrics
        
        Args:
            results_df: DataFrame with actual and predicted values
            
        Returns:
            Dictionary with metrics
        """
        if 'actual_cloudburst' not in results_df.columns:
            logger.warning("No actual cloud burst labels found")
            return {}
        
        try:
            actual = results_df['actual_cloudburst']
            predicted = results_df['predicted_cloudburst']
            
            # Calculate metrics
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, 
                f1_score, confusion_matrix, roc_auc_score
            )
            
            metrics = {
                'accuracy': accuracy_score(actual, predicted),
                'precision': precision_score(actual, predicted, zero_division=0),
                'recall': recall_score(actual, predicted, zero_division=0),
                'f1_score': f1_score(actual, predicted, zero_division=0)
            }
            
            # Add ROC AUC if probabilities available
            if 'cloudburst_probability' in results_df.columns:
                metrics['roc_auc'] = roc_auc_score(actual, results_df['cloudburst_probability'])
            
            # Confusion matrix
            cm = confusion_matrix(actual, predicted)
            metrics['confusion_matrix'] = {
                'true_negatives': int(cm[0, 0]),
                'false_positives': int(cm[0, 1]),
                'false_negatives': int(cm[1, 0]),
                'true_positives': int(cm[1, 1])
            }
            
            # Calculate warning time (how early did we predict?)
            if actual.sum() > 0:  # If there were actual cloud bursts
                cloudburst_times = results_df[actual == 1]['time']
                first_cloudburst = cloudburst_times.min()
                
                # Find first high-probability prediction before cloud burst
                high_prob = results_df[
                    (results_df['cloudburst_probability'] > 0.7) &
                    (results_df['time'] < first_cloudburst)
                ]
                
                if len(high_prob) > 0:
                    first_warning = high_prob['time'].min()
                    warning_time = (first_cloudburst - first_warning).total_seconds() / 3600
                    metrics['warning_time_hours'] = warning_time
                else:
                    metrics['warning_time_hours'] = 0
            
            logger.info("✅ Calculated validation metrics")
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error calculating metrics: {e}")
            return {}
    
    def validate_event(
        self, 
        event: Dict,
        hours_before: int = 24
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Validate model predictions for a single historical event
        
        Args:
            event: Cloud burst event dictionary
            hours_before: Hours of data before event to analyze
            
        Returns:
            Tuple of (results DataFrame, metrics dictionary)
        """
        logger.info(f"Validating event: {event['location']} on {event['date']}")
        
        # Collect historical data
        df = self.collector.collect_event_historical_data(event, hours_before=hours_before)
        
        if df is None or df.empty:
            logger.error("No data collected for event")
            return pd.DataFrame(), {}
        
        # Prepare features
        features_df = self.prepare_features(df)
        
        if features_df.empty:
            logger.error("Failed to prepare features")
            return pd.DataFrame(), {}
        
        # Make predictions
        results_df = self.predict_on_historical(features_df)
        
        # Calculate metrics
        metrics = self.calculate_metrics(results_df)
        
        # Add event info to metrics
        metrics['event'] = {
            'date': event['date'],
            'location': event['location'],
            'rainfall_mm': event.get('rainfall_mm', 'Unknown')
        }
        
        return results_df, metrics
    
    def validate_all_events(self) -> Dict:
        """
        Validate model on all known historical events
        
        Returns:
            Dictionary with aggregated validation results
        """
        events = self.collector.fetch_known_cloudburst_events()
        
        all_results = []
        all_metrics = []
        
        logger.info(f"Validating {len(events)} historical events...")
        
        for i, event in enumerate(events, 1):
            print(f"\n{'='*70}")
            print(f"Validating Event {i}/{len(events)}: {event['location']}")
            print(f"Date: {event['date']}")
            print(f"{'='*70}")
            
            results_df, metrics = self.validate_event(event)
            
            if not results_df.empty and metrics:
                all_results.append(results_df)
                all_metrics.append(metrics)
                
                # Print individual event metrics
                print(f"\n📊 Event Results:")
                print(f"   Accuracy: {metrics.get('accuracy', 0):.2%}")
                print(f"   Precision: {metrics.get('precision', 0):.2%}")
                print(f"   Recall: {metrics.get('recall', 0):.2%}")
                print(f"   F1 Score: {metrics.get('f1_score', 0):.2%}")
                
                if 'warning_time_hours' in metrics:
                    print(f"   Warning Time: {metrics['warning_time_hours']:.1f} hours before event")
                
                cm = metrics.get('confusion_matrix', {})
                print(f"\n   Confusion Matrix:")
                print(f"     True Positives: {cm.get('true_positives', 0)}")
                print(f"     False Positives: {cm.get('false_positives', 0)}")
                print(f"     True Negatives: {cm.get('true_negatives', 0)}")
                print(f"     False Negatives: {cm.get('false_negatives', 0)}")
            else:
                print(f"❌ Could not validate event")
        
        # Aggregate metrics
        if all_metrics:
            aggregated = {
                'total_events_validated': len(all_metrics),
                'average_accuracy': np.mean([m['accuracy'] for m in all_metrics]),
                'average_precision': np.mean([m['precision'] for m in all_metrics]),
                'average_recall': np.mean([m['recall'] for m in all_metrics]),
                'average_f1_score': np.mean([m['f1_score'] for m in all_metrics]),
                'individual_events': all_metrics
            }
            
            warning_times = [m['warning_time_hours'] for m in all_metrics if 'warning_time_hours' in m]
            if warning_times:
                aggregated['average_warning_time_hours'] = np.mean(warning_times)
            
            # Save results
            results_path = Path("./data/historical/validation_results.json")
            results_path.parent.mkdir(parents=True, exist_ok=True)
            
            import json
            with open(results_path, 'w') as f:
                json.dump(aggregated, f, indent=2, default=str)
            
            logger.info(f"✅ Validation results saved to {results_path}")
            
            return aggregated
        else:
            logger.error("No events were successfully validated")
            return {}
    
    def generate_validation_report(self, results: Dict) -> str:
        """
        Generate a human-readable validation report
        
        Args:
            results: Validation results dictionary
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("="*70)
        report.append("CLOUD BURST MODEL VALIDATION REPORT")
        report.append("="*70)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Model: {self.model_path}")
        report.append(f"\n{'='*70}")
        report.append("OVERALL PERFORMANCE")
        report.append("="*70)
        
        report.append(f"\nEvents Validated: {results.get('total_events_validated', 0)}")
        report.append(f"\nMetrics:")
        report.append(f"  • Accuracy:  {results.get('average_accuracy', 0):.2%}")
        report.append(f"  • Precision: {results.get('average_precision', 0):.2%}")
        report.append(f"  • Recall:    {results.get('average_recall', 0):.2%}")
        report.append(f"  • F1 Score:  {results.get('average_f1_score', 0):.2%}")
        
        if 'average_warning_time_hours' in results:
            report.append(f"\nAverage Warning Time: {results['average_warning_time_hours']:.1f} hours")
        
        report.append(f"\n{'='*70}")
        report.append("INDIVIDUAL EVENT RESULTS")
        report.append("="*70)
        
        for i, event_result in enumerate(results.get('individual_events', []), 1):
            event_info = event_result.get('event', {})
            report.append(f"\n{i}. {event_info.get('location', 'Unknown')}")
            report.append(f"   Date: {event_info.get('date', 'Unknown')}")
            report.append(f"   Accuracy: {event_result.get('accuracy', 0):.2%}")
            report.append(f"   F1 Score: {event_result.get('f1_score', 0):.2%}")
            
            if 'warning_time_hours' in event_result:
                report.append(f"   Warning: {event_result['warning_time_hours']:.1f}h before")
        
        report.append(f"\n{'='*70}")
        
        return "\n".join(report)


def main():
    """Main function to run historical validation"""
    print("\n🔍 Cloud Burst Model Historical Validation\n")
    
    # Initialize validator
    validator = HistoricalValidator()
    
    if validator.model is None:
        print("❌ No model found. Please train a model first.")
        print("   Run: python src/models/train.py")
        return
    
    # Validate all events
    results = validator.validate_all_events()
    
    if results:
        # Generate and display report
        print("\n\n")
        report = validator.generate_validation_report(results)
        print(report)
        
        # Save report
        report_path = Path("./data/historical/validation_report.txt")
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\n📄 Full report saved to: {report_path}")
    else:
        print("\n❌ Validation failed. No results generated.")


if __name__ == "__main__":
    main()
