"""
Query-Based Historical Validation System

This module allows you to check your model's performance by querying with:
- Coordinates and Date
- The system fetches historical weather data
- Runs your model's prediction
- Compares with actual cloud burst event
- Shows detailed performance metrics
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.events_database import CloudBurstEventsDB
from src.data.historical_weather import HistoricalWeatherCollector
from src.features.feature_engineering import WeatherFeatureEngineer as FeatureEngineer
from src.models.alert_system import TieredAlertSystem, determine_region
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, Optional, Tuple
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryBasedValidator:
    """
    Validate model predictions by querying specific dates and locations.
    
    Features:
    - Works for ANY event (past, present, future)
    - 5-tier alert system (Normal/Low/Medium/High/Extreme)
    - Regional threshold adjustments
    - Comprehensive validation metrics
    """
    
    def __init__(self, model_path: str = "./models/trained/random_forest_model.pkl"):
        """Initialize the validator"""
        self.events_db = CloudBurstEventsDB()
        self.weather_collector = HistoricalWeatherCollector()
        self.feature_engineer = FeatureEngineer()
        self.alert_system = TieredAlertSystem(enable_regional_adjustment=True)
        
        # Load model
        self.model = None
        self.model_path = Path(model_path)
        if self.model_path.exists():
            self.model = joblib.load(self.model_path)
            logger.info(f"✅ Model loaded from {model_path}")
        else:
            logger.warning(f"⚠️  Model not found at {model_path}")
    
    def query_and_validate(
        self,
        latitude: float,
        longitude: float,
        date: str,  # YYYY-MM-DD
        hours_before: int = 6,
        hours_after: int = 3
    ) -> Dict:
        """
        Query a specific date/location and validate prediction
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            date: Date to check (YYYY-MM-DD)
            hours_before: Hours of data before event to analyze
            hours_after: Hours after event
            
        Returns:
            Dictionary with event info, weather data, prediction, and validation
        """
        print("\n" + "="*70)
        print("QUERY-BASED VALIDATION")
        print("="*70)
        print(f"\n📍 Query Parameters:")
        print(f"   Coordinates: ({latitude}, {longitude})")
        print(f"   Date: {date}")
        print(f"   Analysis Window: {hours_before}h before to {hours_after}h after")
        
        # Step 1: Check if cloud burst event exists
        print(f"\n🔍 Step 1: Searching for cloud burst event...")
        event = self.events_db.get_event_by_date_location(
            date=date,
            latitude=latitude,
            longitude=longitude,
            radius_km=50.0  # 50km radius
        )
        
        result = {
            'query': {
                'latitude': latitude,
                'longitude': longitude,
                'date': date,
                'timestamp': datetime.now().isoformat()
            },
            'event_found': event is not None
        }
        
        if event:
            print(f"✅ Found event: {event['event_id']}")
            print(f"   Location: {event['location']}")
            print(f"   Date: {event['date']} {event.get('time', '')}")
            print(f"   Rainfall: {event['rainfall_mm']}mm in {event['duration_hours']}h")
            print(f"   Intensity: {event['intensity_mm_per_hour']:.0f}mm/hour")
            if event.get('deaths'):
                print(f"   Deaths: {event['deaths']}")
            
            result['actual_event'] = event
        else:
            print(f"ℹ️  No cloud burst event found in database")
            print(f"   (Searching within 50km radius and ±1 day)")
            result['actual_event'] = None
        
        # Step 2: Fetch historical weather data
        print(f"\n🌤️  Step 2: Fetching historical weather data...")
        
        target_date = datetime.strptime(date, '%Y-%m-%d')
        start_date = (target_date - timedelta(hours=hours_before)).strftime('%Y-%m-%d')
        end_date = (target_date + timedelta(hours=hours_after)).strftime('%Y-%m-%d')
        
        weather_df = self.weather_collector.fetch_historical_open_meteo(
            latitude=latitude,
            longitude=longitude,
            start_date=start_date,
            end_date=end_date
        )
        
        if weather_df is None or weather_df.empty:
            print(f"❌ Failed to fetch weather data")
            result['weather_data'] = None
            result['prediction'] = None
            return result
        
        print(f"✅ Fetched {len(weather_df)} hourly weather records")
        print(f"   Time range: {weather_df['time'].min()} to {weather_df['time'].max()}")
        
        # Show weather conditions
        print(f"\n📊 Weather Conditions on {date}:")
        target_weather = weather_df[weather_df['time'].dt.date == target_date.date()]
        if len(target_weather) > 0:
            print(f"   Temperature: {target_weather['temperature_2m'].mean():.1f}°C")
            print(f"   Humidity: {target_weather['relative_humidity_2m'].mean():.1f}%")
            print(f"   Total Precipitation: {target_weather['precipitation'].sum():.1f}mm")
            print(f"   Max Hourly Rain: {target_weather['precipitation'].max():.1f}mm/h")
            print(f"   Pressure: {target_weather['pressure_msl'].mean():.0f}hPa")
        
        result['weather_data'] = {
            'records_count': len(weather_df),
            'date_range': {
                'start': str(weather_df['time'].min()),
                'end': str(weather_df['time'].max())
            },
            'target_date_summary': {
                'avg_temperature': target_weather['temperature_2m'].mean() if len(target_weather) > 0 else None,
                'avg_humidity': target_weather['relative_humidity_2m'].mean() if len(target_weather) > 0 else None,
                'total_precipitation': target_weather['precipitation'].sum() if len(target_weather) > 0 else None,
                'max_precipitation': target_weather['precipitation'].max() if len(target_weather) > 0 else None
            }
        }
        
        # Step 3: Run model prediction
        print(f"\n🤖 Step 3: Running model prediction...")
        
        if self.model is None:
            print(f"⚠️  No model available for prediction")
            result['prediction'] = None
            result['validation'] = None
            return result
        
        # Prepare features and predict
        predictions_df = self._predict_on_weather_data(weather_df)
        
        if predictions_df is None:
            print(f"❌ Prediction failed")
            result['prediction'] = None
            return result
        
        # Apply tiered alert system (works for ALL events - universal)
        region = determine_region(latitude, longitude)
        predictions_df = self.alert_system.classify_hourly_predictions(
            predictions_df,
            intensity_column=None,  # Will add when intensity model is implemented
            region=region
        )
        
        # Find predictions around target time
        target_predictions = predictions_df[predictions_df['time'].dt.date == target_date.date()]
        
        if len(target_predictions) > 0:
            # Count high-risk hours (threshold adjusted to 0.8 to reduce alert fatigue)
            # Analysis shows 0.8 threshold flags 25% of hours vs 66.7% at 0.5
            HIGH_RISK_THRESHOLD = 0.8
            high_risk_count = (target_predictions['cloudburst_probability'] > HIGH_RISK_THRESHOLD).sum()
            predicted_cloudburst = (target_predictions['predicted_cloudburst'] == 1).any()
            max_probability = target_predictions['cloudburst_probability'].max()
            
            # Get alert statistics
            alert_stats = self.alert_system.get_summary_statistics(target_predictions)
            
            print(f"\n📈 Model Prediction Results:")
            print(f"   Predicted Cloud Burst: {'YES' if predicted_cloudburst else 'NO'}")
            print(f"   Max Probability: {max_probability:.1%}")
            print(f"   High-Risk Hours (≥80%): {high_risk_count} out of {len(target_predictions)}")
            print(f"   Actionable Alerts (≥MEDIUM): {alert_stats['actionable_alerts']} ({alert_stats['actionable_percentage']:.1f}%)")
            print(f"   Peak Alert Level: {alert_stats['max_severity_level']}")
            
            # Show hourly predictions with alert levels
            print(f"\n   Hourly Predictions for {date}:")
            for _, row in target_predictions.iterrows():
                hour = row['time'].strftime('%H:%M')
                prob = row['cloudburst_probability']
                alert_emoji = row['alert_emoji']
                alert_level = row['alert_level']
                print(f"      {hour}: {prob:.1%} - {alert_emoji} {alert_level}")
            
            # Show alert distribution
            print(f"\n   Alert Level Distribution:")
            level_counts = alert_stats['level_counts']
            for level in ['NORMAL', 'LOW', 'MEDIUM', 'HIGH', 'EXTREME']:
                count = level_counts.get(level, 0)
                pct = alert_stats['level_percentages'].get(level, 0)
                emoji = self.alert_system.ALERT_LEVELS[level].emoji
                if count > 0:
                    print(f"      {emoji} {level}: {count} hours ({pct:.1f}%)")
            
            result['prediction'] = {
                'predicted_cloudburst': bool(predicted_cloudburst),
                'max_probability': float(max_probability),
                'high_risk_hours': int(high_risk_count),
                'total_hours': int(len(target_predictions)),
                'region': region,
                'alert_statistics': alert_stats,
                'hourly_predictions': [
                    {
                        'time': str(row['time']),
                        'probability': float(row['cloudburst_probability']),
                        'predicted': int(row['predicted_cloudburst']),
                        'alert_level': row['alert_level'],
                        'alert_emoji': row['alert_emoji'],
                        'alert_color': row['alert_color'],
                        'alert_severity': int(row['alert_severity']),
                        'alert_action': row['alert_action']
                    }
                    for _, row in target_predictions.iterrows()
                ]
            }
        else:
            print(f"⚠️  No predictions available for target date")
            result['prediction'] = None
        
        # Step 4: Validation (if event exists)
        if event and result.get('prediction'):
            print(f"\n✅ Step 4: Validation Against Actual Event")
            
            actual_cloudburst = True  # Event exists in database
            predicted_cloudburst = result['prediction']['predicted_cloudburst']
            
            # Determine accuracy
            correct_prediction = actual_cloudburst == predicted_cloudburst
            
            print(f"\n🎯 Validation Results:")
            print(f"   Actual Event: {'YES ✓' if actual_cloudburst else 'NO'}")
            print(f"   Model Predicted: {'YES ✓' if predicted_cloudburst else 'NO'}")
            print(f"   Prediction: {'✅ CORRECT' if correct_prediction else '❌ INCORRECT'}")
            
            if correct_prediction and predicted_cloudburst:
                print(f"   Result: TRUE POSITIVE ✅")
                print(f"   → Model successfully predicted the cloud burst!")
                validation_type = 'TRUE_POSITIVE'
            elif correct_prediction and not predicted_cloudburst:
                print(f"   Result: TRUE NEGATIVE ✅")
                validation_type = 'TRUE_NEGATIVE'
            elif not correct_prediction and predicted_cloudburst:
                print(f"   Result: FALSE POSITIVE ⚠️")
                print(f"   → Model predicted cloud burst but none occurred")
                validation_type = 'FALSE_POSITIVE'
            else:
                print(f"   Result: FALSE NEGATIVE ❌")
                print(f"   → Model FAILED to predict the cloud burst!")
                validation_type = 'FALSE_NEGATIVE'
            
            # Calculate warning time if correct prediction
            warning_time_hours = None
            if predicted_cloudburst and actual_cloudburst:
                # Find first high-probability prediction (using same threshold as above)
                high_prob_times = predictions_df[predictions_df['cloudburst_probability'] > HIGH_RISK_THRESHOLD]['time']
                if len(high_prob_times) > 0:
                    first_warning = high_prob_times.min()
                    # Handle time field that may be a time string or description like "afternoon"
                    event_time_str = event.get('time', '12:00')
                    # If time is descriptive (like "afternoon"), use default time
                    if ':' not in event_time_str:
                        event_time_str = '14:00'  # Default to 2pm for descriptive times
                    event_time = datetime.strptime(f"{event['date']} {event_time_str}", '%Y-%m-%d %H:%M')
                    warning_time_hours = (event_time - first_warning).total_seconds() / 3600
                    
                    if warning_time_hours > 0:
                        print(f"   Warning Time: {warning_time_hours:.1f} hours before event")
                    else:
                        print(f"   Warning Time: Detected during event")
            
            result['validation'] = {
                'actual_cloudburst': actual_cloudburst,
                'predicted_cloudburst': predicted_cloudburst,
                'correct_prediction': correct_prediction,
                'validation_type': validation_type,
                'warning_time_hours': warning_time_hours,
                'max_probability': result['prediction']['max_probability']
            }
        elif not event:
            print(f"\n📊 Step 4: No event validation (no actual event in database)")
            result['validation'] = {
                'actual_cloudburst': False,
                'predicted_cloudburst': result['prediction']['predicted_cloudburst'] if result.get('prediction') else False,
                'note': 'No cloud burst event in database for this date/location'
            }
        
        # Save detailed results
        self._save_query_result(result)
        
        print("\n" + "="*70)
        return result
    
    def _predict_on_weather_data(self, weather_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Run model predictions on weather data"""
        try:
            # The model expects 13 features in this exact order:
            # 1-8: Basic weather features
            # 9-13: Atmospheric stability indices (including duplicate cape)
            
            time_col = weather_df['time'].copy()
            
            # Calculate atmospheric indices for each row
            from src.features.atmospheric_indices import AtmosphericIndices
            indices_calc = AtmosphericIndices()
            
            # Calculate indices for all rows
            lifted_indices = []
            k_indices = []
            total_totals = []
            showalter_indices = []
            
            for _, row in weather_df.iterrows():
                # Get basic parameters
                temp = row['temperature_2m']
                pressure = row['pressure_msl']
                humidity = row['relative_humidity_2m']
                
                # Calculate all atmospheric indices (automatically estimates upper levels)
                indices = indices_calc.calculate_all_indices(
                    surface_temp_c=temp,
                    surface_pressure_hpa=pressure,
                    surface_rh=humidity
                )
                
                lifted_indices.append(indices['lifted_index'])
                k_indices.append(indices['k_index'])
                total_totals.append(indices['total_totals'])
                showalter_indices.append(indices['showalter_index'])
            
            # Get CAPE value (will be duplicated)
            cape_values = weather_df.get('cape', pd.Series([0] * len(weather_df)))
            
            # Create feature dataframe with ALL 13 features in EXACT order
            # CRITICAL: Must match model.feature_names_in_ exactly including duplicate cape
            # We need to create DataFrame with duplicate column names using numpy array
            import numpy as np
            
            # Build the feature array in exact order with all 13 columns
            feature_array = np.column_stack([
                weather_df['temperature_2m'].values,
                weather_df['relative_humidity_2m'].values,
                weather_df['precipitation'].values,
                weather_df['pressure_msl'].values,
                weather_df.get('cloud_cover_total', weather_df.get('cloud_cover', pd.Series([0] * len(weather_df)))).values,
                weather_df.get('wind_speed_10m', pd.Series([0] * len(weather_df))).values,
                weather_df.get('wind_direction_10m', pd.Series([0] * len(weather_df))).values,
                cape_values.values,  # CAPE #1
                cape_values.values,  # CAPE #2 (duplicate)
                lifted_indices,
                k_indices,
                total_totals,
                showalter_indices
            ])
            
            # Create DataFrame with exact column names (including duplicate 'cape')
            column_names = [
                'temperature_2m', 'relative_humidity_2m', 'precipitation', 'pressure_msl',
                'cloud_cover_total', 'wind_speed_10m', 'wind_direction_10m',
                'cape', 'cape',  # Duplicate cape as model expects
                'lifted_index', 'k_index', 'total_totals', 'showalter_index'
            ]
            
            X = pd.DataFrame(feature_array, columns=column_names)
            
            # Handle any missing values
            X = X.fillna(0)
            
            print(f"   Using {len(X.columns)} features for prediction")
            print(f"   Features: {list(X.columns)}")
            
            # Make predictions
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)[:, 1]
            
            # Create result dataframe
            result_df = pd.DataFrame({
                'time': time_col,
                'predicted_cloudburst': predictions,
                'cloudburst_probability': probabilities
            })
            
            print(f"✅ Generated predictions for {len(result_df)} time points")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _save_query_result(self, result: Dict):
        """Save query result to file"""
        output_dir = Path("./data/historical/query_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"query_{timestamp}.json"
        output_path = output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        logger.info(f"Query result saved to {output_path}")
    
    def batch_validate_database(self) -> Dict:
        """
        Validate model against all events in database
        
        Returns:
            Summary of validation results
        """
        print("\n" + "="*70)
        print("BATCH VALIDATION - ALL DATABASE EVENTS")
        print("="*70)
        
        all_events = self.events_db.get_all_events()
        print(f"\nValidating against {len(all_events)} events...\n")
        
        results = {
            'total_events': len(all_events),
            'events': [],
            'summary': {
                'true_positives': 0,
                'false_negatives': 0,
                'correct_predictions': 0,
                'total_warning_time': 0,
                'warnings_count': 0
            }
        }
        
        for i, event in enumerate(all_events, 1):
            print(f"\n{'='*70}")
            print(f"Event {i}/{len(all_events)}: {event['event_id']}")
            print(f"{'='*70}")
            
            result = self.query_and_validate(
                latitude=event['latitude'],
                longitude=event['longitude'],
                date=event['date'],
                hours_before=12,
                hours_after=2
            )
            
            if result.get('validation'):
                val = result['validation']
                results['events'].append({
                    'event_id': event['event_id'],
                    'location': event['location'],
                    'date': event['date'],
                    'validation': val
                })
                
                if val['correct_prediction']:
                    results['summary']['correct_predictions'] += 1
                
                if val['validation_type'] == 'TRUE_POSITIVE':
                    results['summary']['true_positives'] += 1
                    if val.get('warning_time_hours') and val['warning_time_hours'] > 0:
                        results['summary']['total_warning_time'] += val['warning_time_hours']
                        results['summary']['warnings_count'] += 1
                elif val['validation_type'] == 'FALSE_NEGATIVE':
                    results['summary']['false_negatives'] += 1
        
        # Calculate summary statistics
        total = results['total_events']
        correct = results['summary']['correct_predictions']
        tp = results['summary']['true_positives']
        fn = results['summary']['false_negatives']
        
        results['summary']['accuracy'] = (correct / total * 100) if total > 0 else 0
        results['summary']['recall'] = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
        
        if results['summary']['warnings_count'] > 0:
            results['summary']['avg_warning_time'] = (
                results['summary']['total_warning_time'] / results['summary']['warnings_count']
            )
        
        # Print summary
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        print(f"\nTotal Events Validated: {total}")
        print(f"Correct Predictions: {correct} ({results['summary']['accuracy']:.1f}%)")
        print(f"True Positives: {tp}")
        print(f"False Negatives: {fn}")
        print(f"Recall: {results['summary']['recall']:.1f}%")
        
        if results['summary'].get('avg_warning_time'):
            print(f"Average Warning Time: {results['summary']['avg_warning_time']:.1f} hours")
        
        # Save summary
        output_path = Path("./data/historical/batch_validation_summary.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n✅ Full results saved to: {output_path}")
        
        return results


def main():
    """Main function for interactive query"""
    validator = QueryBasedValidator()
    
    print("\n" + "="*70)
    print("QUERY-BASED VALIDATION SYSTEM")
    print("="*70)
    print("\nTest your model's performance on specific dates and locations!")
    print("\nOptions:")
    print("1. Query specific date/location")
    print("2. Validate against all database events")
    print("3. Show database statistics")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        # Manual query
        print("\n" + "-"*70)
        lat = float(input("Enter latitude: "))
        lon = float(input("Enter longitude: "))
        date = input("Enter date (YYYY-MM-DD): ").strip()
        
        result = validator.query_and_validate(lat, lon, date)
        
    elif choice == '2':
        # Batch validation
        results = validator.batch_validate_database()
        
    elif choice == '3':
        # Show statistics
        stats = validator.events_db.get_statistics()
        print("\n📊 Database Statistics:")
        print(json.dumps(stats, indent=2, default=str))
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()
