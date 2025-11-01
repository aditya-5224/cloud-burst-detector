"""
Automated Model Retraining Pipeline

Handles periodic model retraining, evaluation, and deployment with version control.
"""
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelRetrainingPipeline:
    """Automated model retraining with versioning and validation"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.models_dir = Path("models/trained")
        self.versions_dir = Path("models/versions")
        self.metrics_dir = Path("models/metrics")
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        self.min_accuracy_threshold = 0.75
        self.min_samples_for_retraining = 100
        
    def collect_training_data(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
        """
        Collect new training data from various sources
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            Tuple of (features, labels) or None if insufficient data
        """
        logger.info(f"Collecting training data from {start_date} to {end_date}")
        
        # Collect from processed data directory
        data_dir = Path("data/processed")
        all_features = []
        all_labels = []
        
        if data_dir.exists():
            for csv_file in data_dir.glob("engineered_features_*.csv"):
                try:
                    df = pd.read_csv(csv_file)
                    
                    # Filter by date if timestamp column exists
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
                    
                    # Check if label column exists
                    if 'cloud_burst' in df.columns or 'label' in df.columns:
                        label_col = 'cloud_burst' if 'cloud_burst' in df.columns else 'label'
                        
                        # Separate features and labels
                        features = df.drop(columns=[label_col, 'timestamp'], errors='ignore')
                        labels = df[label_col]
                        
                        all_features.append(features)
                        all_labels.append(labels)
                        
                        logger.info(f"Loaded {len(features)} samples from {csv_file.name}")
                        
                except Exception as e:
                    logger.error(f"Error loading {csv_file}: {e}")
                    continue
        
        # Combine all data
        if all_features:
            X = pd.concat(all_features, ignore_index=True)
            y = pd.concat(all_labels, ignore_index=True)
            
            logger.info(f"Total collected samples: {len(X)}")
            
            # Check if we have enough samples
            if len(X) >= self.min_samples_for_retraining:
                return X, y
            else:
                logger.warning(f"Insufficient samples for retraining: {len(X)} < {self.min_samples_for_retraining}")
                return None
        else:
            logger.warning("No training data found")
            return None
    
    def train_model(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        model_type: str = 'random_forest'
    ) -> Tuple[object, Dict]:
        """
        Train a new model and evaluate performance
        
        Args:
            X: Feature matrix
            y: Target labels
            model_type: Type of model to train
            
        Returns:
            Tuple of (trained_model, metrics_dict)
        """
        logger.info(f"Training {model_type} model with {len(X)} samples")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize model based on type
        if model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'svm':
            model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Train model
        logger.info("Training model...")
        model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'positive_samples': int(y.sum()),
            'negative_samples': int(len(y) - y.sum())
        }
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        logger.info(f"Model trained - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
        
        return model, metrics
    
    def version_model(
        self, 
        model: object, 
        metrics: Dict, 
        model_type: str
    ) -> str:
        """
        Save model with version information
        
        Args:
            model: Trained model
            metrics: Performance metrics
            model_type: Type of model
            
        Returns:
            Version identifier
        """
        # Create version identifier
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"{model_type}_v{timestamp}"
        
        # Save model
        model_path = self.versions_dir / f"{version_id}.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Model saved: {model_path}")
        
        # Save metrics
        metrics_path = self.metrics_dir / f"{version_id}_metrics.json"
        metrics_data = {
            'version_id': version_id,
            'model_type': model_type,
            'timestamp': timestamp,
            'metrics': metrics,
            'model_path': str(model_path)
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        logger.info(f"Metrics saved: {metrics_path}")
        
        return version_id
    
    def compare_models(self, new_version: str, current_version: str = None) -> bool:
        """
        Compare new model with current production model
        
        Args:
            new_version: Version ID of new model
            current_version: Version ID of current model (or None to use latest)
            
        Returns:
            True if new model should be deployed
        """
        # Load new model metrics
        new_metrics_path = self.metrics_dir / f"{new_version}_metrics.json"
        if not new_metrics_path.exists():
            logger.error(f"New model metrics not found: {new_metrics_path}")
            return False
        
        with open(new_metrics_path, 'r') as f:
            new_metrics = json.load(f)['metrics']
        
        # Check if meets minimum threshold
        if new_metrics['accuracy'] < self.min_accuracy_threshold:
            logger.warning(f"New model accuracy {new_metrics['accuracy']:.4f} below threshold {self.min_accuracy_threshold}")
            return False
        
        # If no current version, deploy new model if it meets threshold
        if current_version is None:
            logger.info("No current model - deploying new model")
            return True
        
        # Load current model metrics
        current_metrics_path = self.metrics_dir / f"{current_version}_metrics.json"
        if not current_metrics_path.exists():
            logger.warning(f"Current model metrics not found: {current_metrics_path}")
            return True  # Deploy new model if current metrics missing
        
        with open(current_metrics_path, 'r') as f:
            current_metrics = json.load(f)['metrics']
        
        # Compare performance
        new_score = (new_metrics['accuracy'] + new_metrics['f1_score']) / 2
        current_score = (current_metrics['accuracy'] + current_metrics['f1_score']) / 2
        
        improvement = new_score - current_score
        
        logger.info(f"Model comparison:")
        logger.info(f"  Current: Acc={current_metrics['accuracy']:.4f}, F1={current_metrics['f1_score']:.4f}")
        logger.info(f"  New: Acc={new_metrics['accuracy']:.4f}, F1={new_metrics['f1_score']:.4f}")
        logger.info(f"  Improvement: {improvement:.4f}")
        
        # Deploy if improvement > 1%
        if improvement > 0.01:
            logger.info("✅ New model shows improvement - deploying")
            return True
        else:
            logger.info("❌ New model does not show significant improvement")
            return False
    
    def deploy_model(self, version_id: str, model_type: str):
        """
        Deploy model to production
        
        Args:
            version_id: Version ID to deploy
            model_type: Type of model
        """
        source_path = self.versions_dir / f"{version_id}.pkl"
        target_path = self.models_dir / f"{model_type}_model.pkl"
        
        if not source_path.exists():
            raise FileNotFoundError(f"Model version not found: {source_path}")
        
        # Copy model to production
        import shutil
        shutil.copy2(source_path, target_path)
        
        # Update deployment metadata
        metadata = {
            'deployed_version': version_id,
            'deployed_at': datetime.now().isoformat(),
            'model_type': model_type,
            'model_path': str(target_path)
        }
        
        metadata_path = self.models_dir / f"{model_type}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Model deployed: {version_id} -> {target_path}")
    
    def run_retraining_pipeline(
        self, 
        model_type: str = 'random_forest',
        days_back: int = 30
    ) -> Dict:
        """
        Execute complete retraining pipeline
        
        Args:
            model_type: Type of model to retrain
            days_back: Number of days of data to collect
            
        Returns:
            Dictionary with pipeline results
        """
        logger.info(f"Starting retraining pipeline for {model_type}")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Step 1: Collect data
        data = self.collect_training_data(start_date, end_date)
        if data is None:
            return {
                'success': False,
                'message': 'Insufficient training data',
                'timestamp': datetime.now().isoformat()
            }
        
        X, y = data
        
        # Step 2: Train model
        try:
            model, metrics = self.train_model(X, y, model_type)
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                'success': False,
                'message': f'Training failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
        
        # Step 3: Version model
        version_id = self.version_model(model, metrics, model_type)
        
        # Step 4: Compare with current model
        current_metadata_path = self.models_dir / f"{model_type}_metadata.json"
        current_version = None
        
        if current_metadata_path.exists():
            with open(current_metadata_path, 'r') as f:
                current_version = json.load(f).get('deployed_version')
        
        should_deploy = self.compare_models(version_id, current_version)
        
        # Step 5: Deploy if better
        if should_deploy:
            self.deploy_model(version_id, model_type)
            deployment_status = 'deployed'
        else:
            deployment_status = 'not_deployed'
        
        return {
            'success': True,
            'version_id': version_id,
            'metrics': metrics,
            'deployment_status': deployment_status,
            'current_version': current_version,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_model_history(self, model_type: str = None) -> List[Dict]:
        """
        Get history of all model versions
        
        Args:
            model_type: Filter by model type (optional)
            
        Returns:
            List of model version metadata
        """
        history = []
        
        for metrics_file in sorted(self.metrics_dir.glob("*_metrics.json"), reverse=True):
            with open(metrics_file, 'r') as f:
                data = json.load(f)
            
            if model_type is None or data['model_type'] == model_type:
                history.append(data)
        
        return history


def schedule_retraining(interval_days: int = 7):
    """
    Schedule periodic model retraining
    
    Args:
        interval_days: Number of days between retraining
    """
    import schedule
    import time
    
    pipeline = ModelRetrainingPipeline()
    
    def job():
        logger.info("🔄 Scheduled retraining job started")
        for model_type in ['random_forest', 'svm']:
            result = pipeline.run_retraining_pipeline(model_type=model_type)
            logger.info(f"Retraining result for {model_type}: {result}")
    
    # Schedule job
    schedule.every(interval_days).days.do(job)
    
    logger.info(f"Retraining scheduled every {interval_days} days")
    
    # Run scheduler
    while True:
        schedule.run_pending()
        time.sleep(3600)  # Check every hour


if __name__ == "__main__":
    # Example usage
    pipeline = ModelRetrainingPipeline()
    result = pipeline.run_retraining_pipeline(model_type='random_forest', days_back=30)
    print(json.dumps(result, indent=2))
