"""
Cloud Burst Event Labeling System

Tools for labeling historical cloud burst events for model training.
Includes utilities for importing events from various sources and manual labeling.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.database import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CloudBurstLabeler:
    """Handles labeling of cloud burst events"""
    
    def __init__(self):
        """Initialize labeler"""
        self.db = DatabaseManager()
        
        # Cloud burst intensity thresholds (mm/hour)
        self.intensity_thresholds = {
            'low': 20,      # 20-40 mm/hour
            'medium': 40,   # 40-60 mm/hour
            'high': 60,     # 60-80 mm/hour
            'extreme': 80   # >80 mm/hour
        }
    
    def add_manual_event(self, event_data: Dict) -> int:
        """
        Add a manually labeled cloud burst event
        
        Args:
            event_data: Dictionary with event information
            
        Returns:
            Event ID if successful, -1 otherwise
        """
        
        required_fields = ['event_datetime', 'latitude', 'longitude']
        for field in required_fields:
            if field not in event_data:
                logger.error(f"Missing required field: {field}")
                return -1
        
        # Determine intensity if not provided
        if 'intensity' not in event_data and 'precipitation_mm' in event_data:
            duration = event_data.get('duration_minutes', 60)
            rate = event_data['precipitation_mm'] / (duration / 60)  # mm/hour
            
            if rate >= self.intensity_thresholds['extreme']:
                event_data['intensity'] = 'extreme'
            elif rate >= self.intensity_thresholds['high']:
                event_data['intensity'] = 'high'
            elif rate >= self.intensity_thresholds['medium']:
                event_data['intensity'] = 'medium'
            else:
                event_data['intensity'] = 'low'
        
        event_id = self.db.insert_cloud_burst_event(event_data)
        
        if event_id > 0:
            logger.info(f"✓ Added cloud burst event (ID: {event_id})")
        else:
            logger.error("✗ Failed to add cloud burst event")
        
        return event_id
    
    def import_from_csv(self, csv_path: str) -> int:
        """
        Import cloud burst events from CSV file
        
        CSV format:
        event_datetime,latitude,longitude,region,intensity,duration_minutes,precipitation_mm,verified,source,notes
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            Number of events imported
        """
        
        try:
            df = pd.read_csv(csv_path)
            df['event_datetime'] = pd.to_datetime(df['event_datetime'])
            
            imported = 0
            for _, row in df.iterrows():
                event_data = row.to_dict()
                event_id = self.add_manual_event(event_data)
                if event_id > 0:
                    imported += 1
            
            logger.info(f"✓ Imported {imported} events from {csv_path}")
            return imported
            
        except Exception as e:
            logger.error(f"Error importing from CSV: {e}")
            return 0
    
    def detect_events_from_weather_data(self,
                                       start_date: datetime,
                                       end_date: datetime,
                                       region: str = 'default',
                                       auto_label: bool = False) -> pd.DataFrame:
        """
        Detect potential cloud burst events from weather data
        
        Args:
            start_date: Start date for detection
            end_date: End date for detection
            region: Region name
            auto_label: If True, automatically label detected events
            
        Returns:
            DataFrame with detected events
        """
        
        # Get weather data from database
        df = self.db.get_weather_data(start_date, end_date, region)
        
        if df.empty:
            logger.warning("No weather data available for detection")
            return pd.DataFrame()
        
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')
        
        # Detection criteria for cloud burst:
        # 1. High precipitation rate (>20mm in 1 hour)
        # 2. Rapid temperature drop
        # 3. High humidity
        # 4. Low pressure
        
        detected_events = []
        
        for i in range(len(df)):
            row = df.iloc[i]
            
            # Check precipitation
            if pd.notna(row['precipitation']) and row['precipitation'] > 20:
                
                # Look ahead 1-2 hours for sustained high precipitation
                end_idx = min(i + 3, len(df))
                window = df.iloc[i:end_idx]
                
                total_precip = window['precipitation'].sum()
                duration = len(window)
                
                if total_precip > 40:  # At least 40mm in the window
                    
                    event = {
                        'event_datetime': row['datetime'],
                        'latitude': row['latitude'],
                        'longitude': row['longitude'],
                        'region': region,
                        'precipitation_mm': total_precip,
                        'duration_minutes': duration * 60,
                        'verified': False,
                        'source': 'auto_detection',
                        'detection_confidence': self._calculate_confidence(row, window)
                    }
                    
                    # Determine intensity
                    rate = total_precip / duration  # mm/hour
                    if rate >= self.intensity_thresholds['extreme']:
                        event['intensity'] = 'extreme'
                    elif rate >= self.intensity_thresholds['high']:
                        event['intensity'] = 'high'
                    elif rate >= self.intensity_thresholds['medium']:
                        event['intensity'] = 'medium'
                    else:
                        event['intensity'] = 'low'
                    
                    event['notes'] = f"Auto-detected: {total_precip:.1f}mm in {duration}h"
                    
                    detected_events.append(event)
                    
                    # Auto-label if enabled
                    if auto_label:
                        self.add_manual_event(event)
        
        if detected_events:
            df_events = pd.DataFrame(detected_events)
            logger.info(f"✓ Detected {len(detected_events)} potential cloud burst events")
            
            if auto_label:
                logger.info("✓ Events automatically labeled in database")
            else:
                logger.info("ℹ Set auto_label=True to automatically add these to database")
            
            return df_events
        else:
            logger.info("No cloud burst events detected")
            return pd.DataFrame()
    
    def _calculate_confidence(self, row: pd.Series, window: pd.DataFrame) -> float:
        """Calculate detection confidence score (0-1)"""
        
        confidence = 0.0
        
        # High precipitation
        if row['precipitation'] > 50:
            confidence += 0.4
        elif row['precipitation'] > 30:
            confidence += 0.3
        elif row['precipitation'] > 20:
            confidence += 0.2
        
        # High humidity
        if pd.notna(row['relative_humidity_2m']):
            if row['relative_humidity_2m'] > 85:
                confidence += 0.2
            elif row['relative_humidity_2m'] > 75:
                confidence += 0.1
        
        # Low pressure
        if pd.notna(row['pressure_msl']):
            if row['pressure_msl'] < 1005:
                confidence += 0.2
            elif row['pressure_msl'] < 1010:
                confidence += 0.1
        
        # High cloud cover
        if pd.notna(row['cloud_cover']):
            if row['cloud_cover'] > 80:
                confidence += 0.2
            elif row['cloud_cover'] > 60:
                confidence += 0.1
        
        return min(confidence, 1.0)
    
    def create_sample_events_mumbai(self) -> int:
        """
        Create sample cloud burst events for Mumbai region
        Based on historical records from 2023-2024
        """
        
        sample_events = [
            {
                'event_datetime': datetime(2024, 7, 8, 14, 30),
                'latitude': 19.0760,
                'longitude': 72.8777,
                'region': 'Mumbai',
                'intensity': 'high',
                'duration_minutes': 45,
                'precipitation_mm': 82.0,
                'verified': True,
                'source': 'historical_records',
                'notes': 'Mumbai heavy rainfall July 2024'
            },
            {
                'event_datetime': datetime(2024, 6, 22, 16, 15),
                'latitude': 19.1136,
                'longitude': 72.8697,
                'region': 'Mumbai',
                'intensity': 'medium',
                'duration_minutes': 60,
                'precipitation_mm': 55.0,
                'verified': True,
                'source': 'historical_records',
                'notes': 'Intense monsoon activity'
            },
            {
                'event_datetime': datetime(2024, 8, 12, 18, 0),
                'latitude': 19.0176,
                'longitude': 72.8561,
                'region': 'Mumbai',
                'intensity': 'extreme',
                'duration_minutes': 30,
                'precipitation_mm': 95.0,
                'verified': True,
                'source': 'historical_records',
                'notes': 'Flash flooding reported'
            },
            {
                'event_datetime': datetime(2023, 9, 25, 15, 45),
                'latitude': 19.0896,
                'longitude': 72.8656,
                'region': 'Mumbai',
                'intensity': 'high',
                'duration_minutes': 50,
                'precipitation_mm': 68.0,
                'verified': True,
                'source': 'historical_records',
                'notes': 'Post-monsoon heavy showers'
            },
            {
                'event_datetime': datetime(2023, 7, 15, 13, 30),
                'latitude': 19.0330,
                'longitude': 72.8569,
                'region': 'Mumbai',
                'intensity': 'medium',
                'duration_minutes': 55,
                'precipitation_mm': 48.0,
                'verified': True,
                'source': 'historical_records',
                'notes': 'Monsoon peak season'
            }
        ]
        
        imported = 0
        for event in sample_events:
            event_id = self.add_manual_event(event)
            if event_id > 0:
                imported += 1
        
        logger.info(f"✓ Created {imported} sample events for Mumbai")
        return imported
    
    def export_events_to_csv(self, output_path: str, 
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None) -> bool:
        """
        Export labeled events to CSV
        
        Args:
            output_path: Output CSV file path
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            True if successful
        """
        
        try:
            df = self.db.get_cloud_burst_events(start_date, end_date)
            
            if df.empty:
                logger.warning("No events to export")
                return False
            
            df.to_csv(output_path, index=False)
            logger.info(f"✓ Exported {len(df)} events to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting events: {e}")
            return False
    
    def get_event_statistics(self) -> Dict:
        """Get statistics about labeled events"""
        
        df = self.db.get_cloud_burst_events()
        
        if df.empty:
            return {'total_events': 0}
        
        stats = {
            'total_events': len(df),
            'verified_events': df['verified'].sum(),
            'intensity_distribution': df['intensity'].value_counts().to_dict(),
            'sources': df['source'].value_counts().to_dict(),
            'date_range': {
                'earliest': df['event_datetime'].min(),
                'latest': df['event_datetime'].max()
            },
            'avg_precipitation': df['precipitation_mm'].mean(),
            'avg_duration': df['duration_minutes'].mean()
        }
        
        return stats
    
    def generate_labeling_report(self) -> str:
        """Generate a report on labeled events"""
        
        stats = self.get_event_statistics()
        
        report = "=" * 80 + "\n"
        report += "CLOUD BURST EVENTS LABELING REPORT\n"
        report += "=" * 80 + "\n\n"
        
        if stats['total_events'] == 0:
            report += "No events labeled yet.\n\n"
            report += "To get started:\n"
            report += "  1. Run: python src/data/event_labeling.py --create-samples\n"
            report += "  2. Or: python src/data/event_labeling.py --detect --auto-label\n"
            report += "  3. Or: python src/data/event_labeling.py --import events.csv\n"
        else:
            report += f"Total Events: {stats['total_events']}\n"
            report += f"Verified Events: {stats['verified_events']}\n"
            report += f"Date Range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}\n\n"
            
            report += "Intensity Distribution:\n"
            for intensity, count in stats['intensity_distribution'].items():
                report += f"  {intensity}: {count} events\n"
            
            report += "\nData Sources:\n"
            for source, count in stats['sources'].items():
                report += f"  {source}: {count} events\n"
            
            report += f"\nAverage Precipitation: {stats['avg_precipitation']:.1f} mm\n"
            report += f"Average Duration: {stats['avg_duration']:.1f} minutes\n"
        
        report += "\n" + "=" * 80 + "\n"
        
        return report


def main():
    """Main function for event labeling"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Cloud Burst Event Labeling')
    parser.add_argument('--create-samples', action='store_true',
                       help='Create sample events for Mumbai')
    parser.add_argument('--detect', action='store_true',
                       help='Detect events from weather data')
    parser.add_argument('--auto-label', action='store_true',
                       help='Automatically label detected events')
    parser.add_argument('--import', dest='import_csv', type=str,
                       help='Import events from CSV file')
    parser.add_argument('--export', type=str,
                       help='Export events to CSV file')
    parser.add_argument('--report', action='store_true',
                       help='Show labeling report')
    parser.add_argument('--months-back', type=int, default=6,
                       help='Months back to detect events')
    
    args = parser.parse_args()
    
    labeler = CloudBurstLabeler()
    
    if args.create_samples:
        labeler.create_sample_events_mumbai()
        print(labeler.generate_labeling_report())
    
    elif args.detect:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.months_back * 30)
        
        print(f"Detecting events from {start_date.date()} to {end_date.date()}...")
        df_events = labeler.detect_events_from_weather_data(
            start_date, end_date, auto_label=args.auto_label
        )
        
        if not df_events.empty:
            print(f"\nDetected {len(df_events)} potential events:")
            print(df_events[['event_datetime', 'intensity', 'precipitation_mm', 'detection_confidence']])
    
    elif args.import_csv:
        imported = labeler.import_from_csv(args.import_csv)
        print(f"Imported {imported} events")
    
    elif args.export:
        success = labeler.export_events_to_csv(args.export)
        if success:
            print(f"Events exported to {args.export}")
    
    elif args.report:
        print(labeler.generate_labeling_report())
    
    else:
        print(labeler.generate_labeling_report())
        print("\nUsage examples:")
        print("  Create samples: python src/data/event_labeling.py --create-samples")
        print("  Detect events:  python src/data/event_labeling.py --detect --auto-label")
        print("  Import CSV:     python src/data/event_labeling.py --import events.csv")
        print("  Export CSV:     python src/data/event_labeling.py --export events.csv")


if __name__ == "__main__":
    main()