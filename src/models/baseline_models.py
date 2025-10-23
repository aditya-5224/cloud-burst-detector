"""
Baseline Machine Learning Models

Implementation of baseline models for cloud burst prediction including
Random Forest, SVM, and LSTM models with training, evaluation, and validation.
"""

import pandas as pd
import numpy as np
import yaml
import logging
import joblib
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)

# TensorFlow/Keras imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. LSTM models will be disabled.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineModels:
    """Class for training and evaluating baseline ML models"""
    
    def __init__(self, config_path: str = "./config/config.yaml"):
        """Initialize the baseline models trainer"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['models']
        self.training_config = self.config['training']
        self.models_path = Path(self.config['data']['models_path'])
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        
    def prepare_data(self, df: pd.DataFrame, 
                    target_column: str = 'cloud_burst_event',
                    test_size: float = None,
                    validation_size: float = None) -> Tuple[np.ndarray, ...]:
        """
        Prepare data for model training
        
        Args:
            df: DataFrame with features and target
            target_column: Name of target column
            test_size: Fraction of data for testing
            validation_size: Fraction of data for validation
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        
        # Use config values if not provided
        test_size = test_size or self.training_config['test_size']
        validation_size = validation_size or self.training_config['validation_size']
        
        # Create target column if it doesn't exist (for demo purposes)
        if target_column not in df.columns:
            logger.warning(f"Target column '{target_column}' not found. Creating synthetic target.")
            df = self._create_synthetic_target(df, target_column)
        
        # Prepare features
        feature_columns = [col for col in df.columns 
                          if col not in [target_column, 'datetime', 'latitude', 'longitude', 'source']]
        
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Convert to numpy arrays
        X = X.values
        y = y.values
        
        # Split data chronologically for time series
        train_end = int(len(X) * (1 - test_size - validation_size))
        val_end = int(len(X) * (1 - test_size))
        
        X_train = X[:train_end]
        X_val = X[train_end:val_end] if validation_size > 0 else None
        X_test = X[val_end:]
        
        y_train = y[:train_end]
        y_val = y[train_end:val_end] if validation_size > 0 else None
        y_test = y[val_end:]
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val) if X_val is not None else 0}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _create_synthetic_target(self, df: pd.DataFrame, 
                                target_column: str) -> pd.DataFrame:
        """Create synthetic cloud burst events for demonstration"""
        df = df.copy()
        
        # Create synthetic target based on weather conditions
        # High probability conditions: high humidity, high temperature, low pressure
        conditions = (
            (df.get('relative_humidity_2m', 50) > 80) &
            (df.get('temperature_2m', 20) > 28) &
            (df.get('pressure_msl', 1013) < 1005) &
            (df.get('cloud_cover', 50) > 70)
        )
        
        # Add some randomness
        np.random.seed(42)
        random_events = np.random.random(len(df)) < 0.05  # 5% random events
        
        df[target_column] = (conditions | random_events).astype(int)
        
        logger.info(f"Created synthetic target with {df[target_column].sum()} positive events "
                   f"({df[target_column].mean():.1%} of data)")
        
        return df
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray = None, y_val: np.ndarray = None) -> RandomForestClassifier:
        """
        Train Random Forest model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            
        Returns:
            Trained Random Forest model
        """
        
        logger.info("Training Random Forest model...")
        
        rf_config = self.model_config['random_forest']
        
        model = RandomForestClassifier(
            n_estimators=rf_config['n_estimators'],
            max_depth=rf_config['max_depth'],
            random_state=rf_config['random_state'],
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate on validation set if available
        if X_val is not None and y_val is not None:
            val_score = model.score(X_val, y_val)
            logger.info(f"Random Forest validation accuracy: {val_score:.4f}")
        
        self.models['random_forest'] = model
        logger.info("Random Forest training completed")
        
        return model
    
    def train_svm(self, X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray = None, y_val: np.ndarray = None) -> SVC:
        """
        Train SVM model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            
        Returns:
            Trained SVM model
        """
        
        logger.info("Training SVM model...")
        
        svm_config = self.model_config['svm']
        
        # Scale features for SVM
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        self.scalers['svm'] = scaler
        
        model = SVC(
            kernel=svm_config['kernel'],
            C=svm_config['C'],
            gamma=svm_config['gamma'],
            probability=True,  # Enable probability predictions
            random_state=42
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate on validation set if available
        if X_val is not None and y_val is not None:
            X_val_scaled = scaler.transform(X_val)
            val_score = model.score(X_val_scaled, y_val)
            logger.info(f"SVM validation accuracy: {val_score:.4f}")
        
        self.models['svm'] = model
        logger.info("SVM training completed")
        
        return model
    
    def train_lstm(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray = None, y_val: np.ndarray = None,
                   sequence_length: int = 24) -> Optional[Any]:
        """
        Train LSTM model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            sequence_length: Length of input sequences
            
        Returns:
            Trained LSTM model or None if TensorFlow not available
        """
        
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. Skipping LSTM training.")
            return None
        
        logger.info("Training LSTM model...")
        
        lstm_config = self.model_config['lstm']
        
        # Prepare sequences for LSTM
        X_train_seq, y_train_seq = self._create_sequences(X_train, y_train, sequence_length)
        
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self._create_sequences(X_val, y_val, sequence_length)
        else:
            X_val_seq, y_val_seq = None, None
        
        # Scale features for LSTM
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_seq.reshape(-1, X_train_seq.shape[-1]))
        X_train_scaled = X_train_scaled.reshape(X_train_seq.shape)
        self.scalers['lstm'] = scaler
        
        if X_val_seq is not None:
            X_val_scaled = scaler.transform(X_val_seq.reshape(-1, X_val_seq.shape[-1]))
            X_val_scaled = X_val_scaled.reshape(X_val_seq.shape)
        else:
            X_val_scaled = None
        
        # Build LSTM model
        model = Sequential([
            LSTM(lstm_config['units'], 
                 return_sequences=True if lstm_config['layers'] > 1 else False,
                 input_shape=(sequence_length, X_train_seq.shape[2])),
            Dropout(lstm_config['dropout']),
        ])
        
        # Add additional LSTM layers if specified
        for i in range(1, lstm_config['layers']):
            return_sequences = i < lstm_config['layers'] - 1
            model.add(LSTM(lstm_config['units'], return_sequences=return_sequences))
            model.add(Dropout(lstm_config['dropout']))
        
        # Output layer
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(patience=5, factor=0.5)
        ]
        
        # Train model
        validation_data = (X_val_scaled, y_val_seq) if X_val_scaled is not None else None
        
        history = model.fit(
            X_train_scaled, y_train_seq,
            epochs=lstm_config['epochs'],
            batch_size=lstm_config['batch_size'],
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        self.models['lstm'] = model
        logger.info("LSTM training completed")
        
        return model
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray, 
                         sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training
        
        Args:
            X: Features array of shape (samples, features)
            y: Target array of shape (samples,)
            sequence_length: Length of each sequence (timesteps)
            
        Returns:
            Tuple of (X_seq, y_seq) where:
                X_seq: shape (samples, timesteps, features)
                y_seq: shape (samples,)
        """
        X_seq, y_seq = [], []
        
        # Ensure we have enough samples for the sequence
        if len(X) <= sequence_length:
            logger.warning(f"Not enough data for sequence_length={sequence_length}. Need at least {sequence_length+1} samples.")
            return np.array([]), np.array([])
        
        for i in range(sequence_length, len(X)):
            # Get sequence of length 'sequence_length' ending at position i-1
            X_seq.append(X[i-sequence_length:i])
            # Target is the value at position i
            y_seq.append(y[i])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        logger.info(f"Created {len(X_seq)} sequences from {len(X)} samples")
        logger.info(f"  X_seq shape: {X_seq.shape} (samples, timesteps, features)")
        logger.info(f"  y_seq shape: {y_seq.shape}")
        
        return X_seq, y_seq
    
    def evaluate_model(self, model_name: str, X_test: np.ndarray, 
                      y_test: np.ndarray, sequence_length: int = 24) -> Dict[str, float]:
        """
        Evaluate a trained model
        
        Args:
            model_name: Name of the model to evaluate
            X_test: Test features
            y_test: Test target
            sequence_length: Sequence length for LSTM
            
        Returns:
            Dictionary with evaluation metrics
        """
        
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return {}
        
        model = self.models[model_name]
        
        # Prepare test data based on model type
        if model_name == 'lstm':
            if not TENSORFLOW_AVAILABLE:
                return {}
            X_test_prep, y_test_prep = self._create_sequences(X_test, y_test, sequence_length)
            scaler = self.scalers.get('lstm')
            if scaler:
                X_test_prep = scaler.transform(X_test_prep.reshape(-1, X_test_prep.shape[-1]))
                X_test_prep = X_test_prep.reshape(X_test_prep.shape[0], sequence_length, -1)
        elif model_name == 'svm':
            scaler = self.scalers.get('svm')
            X_test_prep = scaler.transform(X_test) if scaler else X_test
            y_test_prep = y_test
        else:  # random_forest
            X_test_prep = X_test
            y_test_prep = y_test
        
        # Make predictions
        if model_name == 'lstm':
            y_pred_proba = model.predict(X_test_prep).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred = model.predict(X_test_prep)
            y_pred_proba = model.predict_proba(X_test_prep)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test_prep, y_pred),
            'precision': precision_score(y_test_prep, y_pred, zero_division=0),
            'recall': recall_score(y_test_prep, y_pred, zero_division=0),
            'f1_score': f1_score(y_test_prep, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test_prep, y_pred_proba) if len(np.unique(y_test_prep)) > 1 else 0
        }
        
        logger.info(f"{model_name} evaluation metrics: {metrics}")
        return metrics
    
    def train_all_models(self, df: pd.DataFrame, 
                        target_column: str = 'cloud_burst_event') -> Dict[str, Dict[str, float]]:
        """
        Train all baseline models and evaluate them
        
        Args:
            df: DataFrame with features and target
            target_column: Name of target column
            
        Returns:
            Dictionary with evaluation metrics for all models
        """
        
        logger.info("Starting training of all baseline models")
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(df, target_column)
        
        results = {}
        
        # Train Random Forest
        try:
            self.train_random_forest(X_train, y_train, X_val, y_val)
            results['random_forest'] = self.evaluate_model('random_forest', X_test, y_test)
        except Exception as e:
            logger.error(f"Error training Random Forest: {e}")
            results['random_forest'] = {}
        
        # Train SVM
        try:
            self.train_svm(X_train, y_train, X_val, y_val)
            results['svm'] = self.evaluate_model('svm', X_test, y_test)
        except Exception as e:
            logger.error(f"Error training SVM: {e}")
            results['svm'] = {}
        
        # Train LSTM
        try:
            self.train_lstm(X_train, y_train, X_val, y_val)
            results['lstm'] = self.evaluate_model('lstm', X_test, y_test)
        except Exception as e:
            logger.error(f"Error training LSTM: {e}")
            results['lstm'] = {}
        
        # Save models
        self.save_models()
        
        logger.info("All models trained and evaluated")
        return results
    
    def save_models(self) -> None:
        """Save trained models to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_name, model in self.models.items():
            try:
                if model_name == 'lstm' and TENSORFLOW_AVAILABLE:
                    # Save Keras model
                    model_path = self.models_path / f"{model_name}_model_{timestamp}.h5"
                    model.save(model_path)
                else:
                    # Save scikit-learn model
                    model_path = self.models_path / f"{model_name}_model_{timestamp}.joblib"
                    joblib.dump(model, model_path)
                
                logger.info(f"Saved {model_name} model to {model_path}")
                
            except Exception as e:
                logger.error(f"Error saving {model_name} model: {e}")
        
        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            try:
                scaler_path = self.models_path / f"{scaler_name}_scaler_{timestamp}.joblib"
                joblib.dump(scaler, scaler_path)
                logger.info(f"Saved {scaler_name} scaler to {scaler_path}")
            except Exception as e:
                logger.error(f"Error saving {scaler_name} scaler: {e}")
        
        # Save feature names
        try:
            feature_path = self.models_path / f"feature_names_{timestamp}.joblib"
            joblib.dump(self.feature_names, feature_path)
            logger.info(f"Saved feature names to {feature_path}")
        except Exception as e:
            logger.error(f"Error saving feature names: {e}")
    
    def generate_model_report(self, results: Dict[str, Dict[str, float]]) -> str:
        """
        Generate a comprehensive model evaluation report
        
        Args:
            results: Dictionary with model evaluation results
            
        Returns:
            String with formatted report
        """
        
        report = "=" * 60 + "\n"
        report += "CLOUD BURST PREDICTION - BASELINE MODELS REPORT\n"
        report += "=" * 60 + "\n\n"
        
        # Model comparison table
        report += "Model Performance Comparison:\n"
        report += "-" * 60 + "\n"
        report += f"{'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10}\n"
        report += "-" * 60 + "\n"
        
        for model_name, metrics in results.items():
            if metrics:
                report += f"{model_name:<15} "
                report += f"{metrics.get('accuracy', 0):<10.4f} "
                report += f"{metrics.get('precision', 0):<10.4f} "
                report += f"{metrics.get('recall', 0):<10.4f} "
                report += f"{metrics.get('f1_score', 0):<10.4f} "
                report += f"{metrics.get('roc_auc', 0):<10.4f}\n"
        
        report += "-" * 60 + "\n\n"
        
        # Achievement of acceptance criteria
        report += "Acceptance Criteria Assessment:\n"
        report += "-" * 40 + "\n"
        threshold = self.config['thresholds']['model_f1_score']
        
        for model_name, metrics in results.items():
            if metrics:
                f1 = metrics.get('f1_score', 0)
                status = "✓ PASS" if f1 >= threshold else "✗ FAIL"
                report += f"{model_name}: F1-Score {f1:.4f} (Target: {threshold}) - {status}\n"
        
        report += "\n"
        
        # Recommendations
        report += "Recommendations:\n"
        report += "-" * 20 + "\n"
        
        best_model = max(results.keys(), 
                        key=lambda k: results[k].get('f1_score', 0) if results[k] else 0,
                        default='none')
        
        if best_model != 'none':
            best_f1 = results[best_model].get('f1_score', 0)
            report += f"• Best performing model: {best_model} (F1-Score: {best_f1:.4f})\n"
            
            if best_f1 >= threshold:
                report += "• Target F1-score achieved. Model ready for deployment.\n"
            else:
                report += "• Target F1-score not achieved. Consider:\n"
                report += "  - Feature engineering improvements\n"
                report += "  - Hyperparameter tuning\n"
                report += "  - Ensemble methods\n"
                report += "  - More training data\n"
        
        report += "\n" + "=" * 60 + "\n"
        
        return report


def main():
    """Main function for running baseline model training"""
    trainer = BaselineModels()
    
    # Generate sample data for demonstration
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-03-01', freq='H')
    
    sample_data = pd.DataFrame({
        'datetime': dates,
        'temperature_2m': np.random.normal(25, 5, len(dates)),
        'relative_humidity_2m': np.random.uniform(40, 95, len(dates)),
        'pressure_msl': np.random.normal(1013, 15, len(dates)),
        'wind_speed_10m': np.random.exponential(3, len(dates)),
        'cloud_cover': np.random.uniform(0, 100, len(dates)),
        'precipitation': np.random.exponential(0.5, len(dates)),
        'cape': np.random.exponential(1000, len(dates)),
        'lifted_index': np.random.normal(2, 3, len(dates))
    })
    
    # Train all models
    results = trainer.train_all_models(sample_data)
    
    # Generate and print report
    report = trainer.generate_model_report(results)
    print(report)
    
    # Save report to file
    report_path = trainer.models_path / f"model_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()