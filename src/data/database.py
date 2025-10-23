"""
Database Models and Connection for Cloud Burst Prediction System

Handles data persistence for weather data, satellite imagery metadata,
predictions, and cloud burst events.
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self, db_path: str = "./data/cloudburst.db"):
        """Initialize database manager"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self.init_database()
    
    @staticmethod
    def _convert_datetime(dt) -> Optional[str]:
        """Convert datetime/Timestamp to ISO format string for SQLite"""
        if dt is None:
            return None
        if isinstance(dt, pd.Timestamp):
            return dt.isoformat()
        if isinstance(dt, datetime):
            return dt.isoformat()
        if isinstance(dt, str):
            return dt
        return str(dt)
    
    def get_connection(self):
        """Get database connection"""
        if self.conn is None:
            self.conn = sqlite3.connect(str(self.db_path))
            self.conn.row_factory = sqlite3.Row
        return self.conn
    
    def init_database(self):
        """Initialize database schema"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Weather data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS weather_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                datetime TIMESTAMP NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                region VARCHAR(100),
                temperature_2m REAL,
                relative_humidity_2m REAL,
                pressure_msl REAL,
                wind_speed_10m REAL,
                wind_direction_10m REAL,
                cloud_cover REAL,
                precipitation REAL,
                source VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(datetime, latitude, longitude, source)
            )
        """)
        
        # Satellite imagery metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS satellite_imagery (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                datetime TIMESTAMP NOT NULL,
                region VARCHAR(100) NOT NULL,
                image_path TEXT,
                cloud_coverage_percentage REAL,
                quality_score REAL,
                source VARCHAR(50),
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(datetime, region)
            )
        """)
        
        # Cloud burst events table (labeled data)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cloud_burst_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_datetime TIMESTAMP NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                region VARCHAR(100),
                intensity VARCHAR(50),
                duration_minutes INTEGER,
                precipitation_mm REAL,
                verified BOOLEAN DEFAULT 0,
                source VARCHAR(100),
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(event_datetime, latitude, longitude)
            )
        """)
        
        # Predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_datetime TIMESTAMP NOT NULL,
                target_datetime TIMESTAMP NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                region VARCHAR(100),
                model_name VARCHAR(100),
                model_version VARCHAR(50),
                prediction INTEGER NOT NULL,
                probability REAL NOT NULL,
                confidence VARCHAR(20),
                features TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Model performance metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name VARCHAR(100) NOT NULL,
                model_version VARCHAR(50) NOT NULL,
                training_date TIMESTAMP NOT NULL,
                accuracy REAL,
                precision_score REAL,
                recall REAL,
                f1_score REAL,
                roc_auc REAL,
                training_samples INTEGER,
                test_samples INTEGER,
                hyperparameters TEXT,
                feature_importance TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Data collection logs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_collection_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                collection_datetime TIMESTAMP NOT NULL,
                data_type VARCHAR(50) NOT NULL,
                source VARCHAR(100),
                records_collected INTEGER,
                success BOOLEAN,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for better query performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_weather_datetime 
            ON weather_data(datetime)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_datetime 
            ON cloud_burst_events(event_datetime)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_datetime 
            ON predictions(prediction_datetime)
        """)
        
        conn.commit()
        logger.info("Database initialized successfully")
    
    def insert_weather_data(self, data: pd.DataFrame) -> int:
        """Insert weather data into database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        inserted = 0
        for _, row in data.iterrows():
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO weather_data 
                    (datetime, latitude, longitude, region, temperature_2m, 
                     relative_humidity_2m, pressure_msl, wind_speed_10m, 
                     wind_direction_10m, cloud_cover, precipitation, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    self._convert_datetime(row.get('datetime')),
                    row.get('latitude'),
                    row.get('longitude'),
                    row.get('region', 'default'),
                    row.get('temperature_2m'),
                    row.get('relative_humidity_2m'),
                    row.get('pressure_msl'),
                    row.get('wind_speed_10m'),
                    row.get('wind_direction_10m'),
                    row.get('cloud_cover'),
                    row.get('precipitation'),
                    row.get('source', 'unknown')
                ))
                if cursor.rowcount > 0:
                    inserted += 1
            except Exception as e:
                logger.error(f"Error inserting weather data row: {e}")
                continue
        
        conn.commit()
        logger.info(f"Inserted {inserted} weather records")
        return inserted
    
    def insert_cloud_burst_event(self, event: Dict) -> int:
        """Insert a cloud burst event (labeled data)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO cloud_burst_events 
                (event_datetime, latitude, longitude, region, intensity, 
                 duration_minutes, precipitation_mm, verified, source, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self._convert_datetime(event['event_datetime']),
                event['latitude'],
                event['longitude'],
                event.get('region', 'default'),
                event.get('intensity', 'unknown'),
                event.get('duration_minutes'),
                event.get('precipitation_mm'),
                event.get('verified', False),
                event.get('source', 'manual'),
                event.get('notes', '')
            ))
            conn.commit()
            return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error inserting cloud burst event: {e}")
            return -1
    
    def insert_prediction(self, prediction: Dict) -> int:
        """Insert a prediction"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO predictions 
                (prediction_datetime, target_datetime, latitude, longitude, 
                 region, model_name, model_version, prediction, probability, 
                 confidence, features)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self._convert_datetime(prediction.get('prediction_datetime', datetime.now())),
                self._convert_datetime(prediction['target_datetime']),
                prediction['latitude'],
                prediction['longitude'],
                prediction.get('region', 'default'),
                prediction['model_name'],
                prediction.get('model_version', '1.0'),
                prediction['prediction'],
                prediction['probability'],
                prediction.get('confidence', 'medium'),
                json.dumps(prediction.get('features', {}))
            ))
            conn.commit()
            return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error inserting prediction: {e}")
            return -1
    
    def insert_model_metrics(self, metrics: Dict) -> int:
        """Insert model performance metrics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO model_metrics 
                (model_name, model_version, training_date, accuracy, 
                 precision_score, recall, f1_score, roc_auc, 
                 training_samples, test_samples, hyperparameters, feature_importance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics['model_name'],
                metrics.get('model_version', '1.0'),
                self._convert_datetime(metrics.get('training_date', datetime.now())),
                metrics.get('accuracy'),
                metrics.get('precision'),
                metrics.get('recall'),
                metrics.get('f1_score'),
                metrics.get('roc_auc'),
                metrics.get('training_samples'),
                metrics.get('test_samples'),
                json.dumps(metrics.get('hyperparameters', {})),
                json.dumps(metrics.get('feature_importance', {}))
            ))
            conn.commit()
            return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error inserting model metrics: {e}")
            return -1
    
    def log_data_collection(self, log_entry: Dict) -> int:
        """Log data collection activity"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO data_collection_logs 
                (collection_datetime, data_type, source, records_collected, 
                 success, error_message)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                log_entry.get('collection_datetime', datetime.now()),
                log_entry['data_type'],
                log_entry.get('source', 'unknown'),
                log_entry.get('records_collected', 0),
                log_entry.get('success', True),
                log_entry.get('error_message', '')
            ))
            conn.commit()
            return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error logging data collection: {e}")
            return -1
    
    def get_weather_data(self, start_date: datetime, end_date: datetime, 
                        region: str = None) -> pd.DataFrame:
        """Retrieve weather data for a date range"""
        conn = self.get_connection()
        
        query = """
            SELECT * FROM weather_data 
            WHERE datetime BETWEEN ? AND ?
        """
        params = [start_date, end_date]
        
        if region:
            query += " AND region = ?"
            params.append(region)
        
        query += " ORDER BY datetime"
        
        df = pd.read_sql_query(query, conn, params=params)
        return df
    
    def get_cloud_burst_events(self, start_date: datetime = None, 
                               end_date: datetime = None) -> pd.DataFrame:
        """Retrieve cloud burst events"""
        conn = self.get_connection()
        
        if start_date and end_date:
            query = """
                SELECT * FROM cloud_burst_events 
                WHERE event_datetime BETWEEN ? AND ?
                ORDER BY event_datetime
            """
            params = [start_date, end_date]
        else:
            query = "SELECT * FROM cloud_burst_events ORDER BY event_datetime"
            params = []
        
        df = pd.read_sql_query(query, conn, params=params)
        return df
    
    def get_latest_model_metrics(self, model_name: str) -> Dict:
        """Get latest metrics for a model"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM model_metrics 
            WHERE model_name = ? 
            ORDER BY training_date DESC 
            LIMIT 1
        """, (model_name,))
        
        row = cursor.fetchone()
        if row:
            return dict(row)
        return {}
    
    def get_prediction_accuracy(self, start_date: datetime, end_date: datetime) -> Dict:
        """Calculate prediction accuracy by comparing with actual events"""
        conn = self.get_connection()
        
        # Get predictions in date range
        predictions = pd.read_sql_query("""
            SELECT * FROM predictions 
            WHERE target_datetime BETWEEN ? AND ?
        """, conn, params=[start_date, end_date])
        
        # Get actual events in date range
        events = pd.read_sql_query("""
            SELECT * FROM cloud_burst_events 
            WHERE event_datetime BETWEEN ? AND ?
        """, conn, params=[start_date, end_date])
        
        # Calculate metrics (simplified - in production, use more sophisticated matching)
        total_predictions = len(predictions)
        total_events = len(events)
        
        return {
            'total_predictions': total_predictions,
            'total_actual_events': total_events,
            'prediction_rate': total_predictions / max(1, total_events)
        }
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("Database connection closed")


def main():
    """Test database operations"""
    db = DatabaseManager()
    
    # Test inserting weather data
    sample_weather = pd.DataFrame({
        'datetime': [datetime.now()],
        'latitude': [19.0760],
        'longitude': [72.8777],
        'region': ['Mumbai'],
        'temperature_2m': [28.5],
        'relative_humidity_2m': [75.0],
        'pressure_msl': [1010.0],
        'wind_speed_10m': [5.2],
        'wind_direction_10m': [180.0],
        'cloud_cover': [65.0],
        'precipitation': [0.0],
        'source': ['test']
    })
    
    inserted = db.insert_weather_data(sample_weather)
    print(f"Inserted {inserted} weather records")
    
    # Test inserting cloud burst event
    sample_event = {
        'event_datetime': datetime(2024, 7, 15, 14, 30),
        'latitude': 19.0760,
        'longitude': 72.8777,
        'region': 'Mumbai',
        'intensity': 'high',
        'duration_minutes': 45,
        'precipitation_mm': 85.0,
        'verified': True,
        'source': 'historical_records',
        'notes': 'Test cloud burst event'
    }
    
    event_id = db.insert_cloud_burst_event(sample_event)
    print(f"Inserted cloud burst event with ID: {event_id}")
    
    db.close()


if __name__ == "__main__":
    main()