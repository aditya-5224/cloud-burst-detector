"""
Sprint 3: Model Training & Evaluation

Train and evaluate machine learning models for cloud burst prediction:
- Random Forest with hyperparameter tuning
- SVM with feature scaling
- LSTM with sequence preparation
- Handle class imbalance with SMOTE
- Comprehensive evaluation metrics
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.database import DatabaseManager
from src.models.baseline_models import BaselineModels

# Scikit-learn imports
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelTrainingPipeline:
    """Complete model training and evaluation pipeline for Sprint 3"""
    
    def __init__(self, data_path: str = "./reports/sprint2/engineered_features.csv",
                 output_dir: str = "./reports/sprint3",
                 models_dir: str = "./models"):
        """
        Initialize the model training pipeline
        
        Args:
            data_path: Path to engineered features CSV
            output_dir: Directory for reports and outputs
            models_dir: Directory to save trained models
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.models_dir = Path(models_dir)
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.db = DatabaseManager()
    
    def load_data(self) -> tuple:
        """Load engineered features from CSV"""
        logger.info("="*80)
        logger.info("LOADING ENGINEERED FEATURES")
        logger.info("="*80)
        
        if not self.data_path.exists():
            logger.error(f"Data file not found: {self.data_path}")
            return None, None
        
        df = pd.read_csv(self.data_path)
        
        logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
        
        # Separate features and target
        X = df.drop(columns=['datetime', 'cloud_burst'], errors='ignore')
        y = df['cloud_burst']
        
        # Handle infinite and NaN values
        logger.info("Cleaning data...")
        inf_count = np.isinf(X.values).sum()
        nan_count = np.isnan(X.values).sum()
        
        if inf_count > 0:
            logger.warning(f"Found {inf_count} infinite values, replacing with NaN")
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values")
        
        # Replace infinities with NaN, then fill with 0
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        logger.info(f"Positive class: {y.sum()} ({(y.sum()/len(y)*100):.2f}%)")
        
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series,
                   test_size: float = 0.2,
                   val_size: float = 0.1) -> tuple:
        """Split data into train/val/test sets"""
        logger.info("\n" + "="*80)
        logger.info("SPLITTING DATA")
        logger.info("="*80)
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Second split: separate validation set from remaining data
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
        )
        
        logger.info(f"Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        logger.info(f"  Positive: {y_train.sum()} ({(y_train.sum()/len(y_train)*100):.2f}%)")
        logger.info(f"Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        logger.info(f"  Positive: {y_val.sum()} ({(y_val.sum()/len(y_val)*100):.2f}%)")
        logger.info(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        logger.info(f"  Positive: {y_test.sum()} ({(y_test.sum()/len(y_test)*100):.2f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def handle_imbalance(self, X_train: pd.DataFrame, y_train: pd.Series,
                        method: str = 'smote') -> tuple:
        """Handle class imbalance"""
        logger.info("\n" + "="*80)
        logger.info("HANDLING CLASS IMBALANCE")
        logger.info("="*80)
        
        logger.info(f"Original class distribution: {y_train.value_counts().to_dict()}")
        
        if method == 'smote':
            # SMOTE with undersampling
            over_sampler = SMOTE(sampling_strategy=0.5, random_state=42)
            under_sampler = RandomUnderSampler(sampling_strategy=0.8, random_state=42)
            
            logger.info("Applying SMOTE + Random Undersampling...")
            X_resampled, y_resampled = over_sampler.fit_resample(X_train, y_train)
            X_resampled, y_resampled = under_sampler.fit_resample(X_resampled, y_resampled)
            
        elif method == 'weights':
            logger.info("Using class weights (no resampling)")
            X_resampled, y_resampled = X_train, y_train
        
        else:
            logger.info("No resampling applied")
            X_resampled, y_resampled = X_train, y_train
        
        logger.info(f"Resampled class distribution: {pd.Series(y_resampled).value_counts().to_dict()}")
        logger.info(f"Resampled size: {len(X_resampled)} samples")
        
        return X_resampled, y_resampled
    
    def train_random_forest(self, X_train, y_train, X_val, y_val,
                           tune_hyperparameters: bool = True) -> dict:
        """Train Random Forest with optional hyperparameter tuning"""
        logger.info("\n" + "="*80)
        logger.info("TRAINING RANDOM FOREST")
        logger.info("="*80)
        
        from sklearn.ensemble import RandomForestClassifier
        
        if tune_hyperparameters:
            logger.info("Performing hyperparameter tuning...")
            
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced', 'balanced_subsample']
            }
            
            rf = RandomForestClassifier(random_state=42, n_jobs=-1)
            
            grid_search = GridSearchCV(
                rf, param_grid, cv=5, scoring='f1',
                n_jobs=-1, verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best CV F1-score: {grid_search.best_score_:.4f}")
            
            model = grid_search.best_estimator_
        else:
            logger.info("Training with default parameters + class weights...")
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)[:, 1]
        
        results = self._calculate_metrics(y_val, y_val_pred, y_val_proba, "Random Forest")
        
        self.models['random_forest'] = model
        self.results['random_forest'] = results
        
        # Save model
        joblib.dump(model, self.models_dir / 'random_forest_model.pkl')
        logger.info(f"Model saved to {self.models_dir / 'random_forest_model.pkl'}")
        
        return results
    
    def train_svm(self, X_train, y_train, X_val, y_val,
                  tune_hyperparameters: bool = True) -> dict:
        """Train SVM with feature scaling"""
        logger.info("\n" + "="*80)
        logger.info("TRAINING SVM")
        logger.info("="*80)
        
        from sklearn.svm import SVC
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        self.scalers['svm'] = scaler
        
        if tune_hyperparameters:
            logger.info("Performing hyperparameter tuning...")
            
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'kernel': ['rbf', 'linear'],
                'class_weight': ['balanced']
            }
            
            svm = SVC(probability=True, random_state=42)
            
            grid_search = GridSearchCV(
                svm, param_grid, cv=3, scoring='f1',
                n_jobs=-1, verbose=1
            )
            
            grid_search.fit(X_train_scaled, y_train)
            
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best CV F1-score: {grid_search.best_score_:.4f}")
            
            model = grid_search.best_estimator_
        else:
            logger.info("Training with default parameters + class weights...")
            model = SVC(
                C=10,
                gamma='scale',
                kernel='rbf',
                class_weight='balanced',
                probability=True,
                random_state=42
            )
            model.fit(X_train_scaled, y_train)
        
        # Evaluate on validation set
        y_val_pred = model.predict(X_val_scaled)
        y_val_proba = model.predict_proba(X_val_scaled)[:, 1]
        
        results = self._calculate_metrics(y_val, y_val_pred, y_val_proba, "SVM")
        
        self.models['svm'] = model
        self.results['svm'] = results
        
        # Save model and scaler
        joblib.dump(model, self.models_dir / 'svm_model.pkl')
        joblib.dump(scaler, self.models_dir / 'svm_scaler.pkl')
        logger.info(f"Model saved to {self.models_dir / 'svm_model.pkl'}")
        
        return results
    
    def train_lstm(self, X_train, y_train, X_val, y_val,
                   sequence_length: int = 24) -> dict:
        """Train LSTM with sequence preparation"""
        logger.info("\n" + "="*80)
        logger.info("TRAINING LSTM")
        logger.info("="*80)
        
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        except ImportError:
            logger.error("TensorFlow not available. Skipping LSTM training.")
            return {}
        
        # Prepare sequences
        from src.features.timeseries_features import create_sequences_for_lstm
        
        # Convert to arrays
        X_train_arr = X_train.values if hasattr(X_train, 'values') else X_train
        X_val_arr = X_val.values if hasattr(X_val, 'values') else X_val
        y_train_arr = y_train.values if hasattr(y_train, 'values') else y_train
        y_val_arr = y_val.values if hasattr(y_val, 'values') else y_val
        
        # Create sequences
        logger.info(f"Creating sequences with length={sequence_length}...")
        
        # Simple sequence creation (use last sequence_length samples)
        if len(X_train_arr) < sequence_length:
            logger.error(f"Not enough data for sequences. Need at least {sequence_length} samples.")
            return {}
        
        # Create sequences manually
        X_train_seq, y_train_seq = [], []
        for i in range(sequence_length, len(X_train_arr)):
            X_train_seq.append(X_train_arr[i-sequence_length:i])
            y_train_seq.append(y_train_arr[i])
        
        X_val_seq, y_val_seq = [], []
        for i in range(sequence_length, len(X_val_arr)):
            X_val_seq.append(X_val_arr[i-sequence_length:i])
            y_val_seq.append(y_val_arr[i])
        
        X_train_seq = np.array(X_train_seq)
        y_train_seq = np.array(y_train_seq)
        X_val_seq = np.array(X_val_seq)
        y_val_seq = np.array(y_val_seq)
        
        logger.info(f"Training sequences: {X_train_seq.shape}")
        logger.info(f"Validation sequences: {X_val_seq.shape}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_seq.reshape(-1, X_train_seq.shape[-1]))
        X_train_scaled = X_train_scaled.reshape(X_train_seq.shape)
        
        X_val_scaled = scaler.transform(X_val_seq.reshape(-1, X_val_seq.shape[-1]))
        X_val_scaled = X_val_scaled.reshape(X_val_seq.shape)
        
        self.scalers['lstm'] = scaler
        
        # Calculate class weights
        class_weight = {
            0: 1.0,
            1: len(y_train_seq) / (2 * y_train_seq.sum()) if y_train_seq.sum() > 0 else 1.0
        }
        
        logger.info(f"Class weights: {class_weight}")
        
        # Build model
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(sequence_length, X_train_seq.shape[2])),
            Dropout(0.3),
            BatchNormalization(),
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            BatchNormalization(),
            Dense(16, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        logger.info("\nModel architecture:")
        model.summary(print_fn=logger.info)
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
            ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6, monitor='val_loss')
        ]
        
        # Train
        logger.info("\nTraining LSTM...")
        history = model.fit(
            X_train_scaled, y_train_seq,
            validation_data=(X_val_scaled, y_val_seq),
            epochs=50,
            batch_size=32,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        y_val_pred_proba = model.predict(X_val_scaled)
        y_val_pred = (y_val_pred_proba > 0.5).astype(int).flatten()
        
        results = self._calculate_metrics(y_val_seq, y_val_pred, y_val_pred_proba.flatten(), "LSTM")
        results['history'] = {
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']]
        }
        
        self.models['lstm'] = model
        self.results['lstm'] = results
        
        # Save model
        model.save(self.models_dir / 'lstm_model.h5')
        joblib.dump(scaler, self.models_dir / 'lstm_scaler.pkl')
        logger.info(f"Model saved to {self.models_dir / 'lstm_model.h5'}")
        
        return results
    
    def _calculate_metrics(self, y_true, y_pred, y_proba, model_name: str) -> dict:
        """Calculate comprehensive evaluation metrics"""
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0,
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        logger.info(f"\n{model_name} - Validation Results:")
        logger.info(f"  Accuracy:  {results['accuracy']:.4f}")
        logger.info(f"  Precision: {results['precision']:.4f}")
        logger.info(f"  Recall:    {results['recall']:.4f}")
        logger.info(f"  F1-Score:  {results['f1_score']:.4f}")
        logger.info(f"  ROC-AUC:   {results['roc_auc']:.4f}")
        logger.info(f"  Confusion Matrix:\n{np.array(results['confusion_matrix'])}")
        
        return results
    
    def evaluate_on_test_set(self, X_test, y_test) -> dict:
        """Evaluate all models on test set"""
        logger.info("\n" + "="*80)
        logger.info("FINAL TEST SET EVALUATION")
        logger.info("="*80)
        
        test_results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"\nEvaluating {model_name}...")
            
            if model_name == 'svm':
                X_test_processed = self.scalers['svm'].transform(X_test)
                y_pred = model.predict(X_test_processed)
                y_proba = model.predict_proba(X_test_processed)[:, 1]
            
            elif model_name == 'lstm':
                # Create sequences
                sequence_length = 24
                X_test_arr = X_test.values if hasattr(X_test, 'values') else X_test
                y_test_arr = y_test.values if hasattr(y_test, 'values') else y_test
                
                X_test_seq, y_test_seq = [], []
                for i in range(sequence_length, len(X_test_arr)):
                    X_test_seq.append(X_test_arr[i-sequence_length:i])
                    y_test_seq.append(y_test_arr[i])
                
                X_test_seq = np.array(X_test_seq)
                y_test_seq = np.array(y_test_seq)
                
                X_test_scaled = self.scalers['lstm'].transform(X_test_seq.reshape(-1, X_test_seq.shape[-1]))
                X_test_scaled = X_test_scaled.reshape(X_test_seq.shape)
                
                y_proba = model.predict(X_test_scaled).flatten()
                y_pred = (y_proba > 0.5).astype(int)
                y_test = y_test_seq
            
            else:  # Random Forest
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
            
            results = self._calculate_metrics(y_test, y_pred, y_proba, f"{model_name} (Test)")
            test_results[model_name] = results
        
        self.results['test'] = test_results
        
        return test_results
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations"""
        logger.info("\n" + "="*80)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("="*80)
        
        # 1. Model comparison
        self._plot_model_comparison()
        
        # 2. Confusion matrices
        self._plot_confusion_matrices()
        
        # 3. Feature importance (Random Forest)
        self._plot_feature_importance()
        
        # 4. LSTM training history
        if 'lstm' in self.results and 'history' in self.results['lstm']:
            self._plot_lstm_history()
        
        logger.info(f"All visualizations saved to {self.output_dir}")
    
    def _plot_model_comparison(self):
        """Plot model performance comparison"""
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        model_names = list(self.models.keys())
        
        data = {metric: [] for metric in metrics}
        
        for model_name in model_names:
            if model_name in self.results:
                for metric in metrics:
                    data[metric].append(self.results[model_name].get(metric, 0))
        
        # Create subplot
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Bar chart
        x = np.arange(len(model_names))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            axes[0].bar(x + i*width, data[metric], width, label=metric.replace('_', ' ').title())
        
        axes[0].set_xlabel('Model')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Model Performance Comparison (Validation Set)')
        axes[0].set_xticks(x + width * 2)
        axes[0].set_xticklabels([m.replace('_', ' ').title() for m in model_names])
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # Heatmap
        heatmap_data = []
        for model_name in model_names:
            if model_name in self.results:
                heatmap_data.append([self.results[model_name].get(m, 0) for m in metrics])
        
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlGnBu',
                   xticklabels=[m.replace('_', ' ').title() for m in metrics],
                   yticklabels=[m.replace('_', ' ').title() for m in model_names],
                   ax=axes[1])
        axes[1].set_title('Performance Heatmap')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Saved model_comparison.png")
    
    def _plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        
        n_models = len(self.models)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, results) in enumerate(self.results.items()):
            if model_name == 'test':
                continue
            
            cm = np.array(results['confusion_matrix'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Predicted 0', 'Predicted 1'],
                       yticklabels=['Actual 0', 'Actual 1'],
                       ax=axes[idx])
            axes[idx].set_title(f'{model_name.replace("_", " ").title()}\nConfusion Matrix')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Saved confusion_matrices.png")
    
    def _plot_feature_importance(self):
        """Plot feature importance from Random Forest"""
        
        if 'random_forest' not in self.models:
            return
        
        model = self.models['random_forest']
        importances = model.feature_importances_
        feature_names = model.feature_names_in_
        
        # Sort by importance
        indices = np.argsort(importances)[::-1][:30]  # Top 30
        
        plt.figure(figsize=(10, 12))
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.title('Top 30 Feature Importances (Random Forest)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Saved feature_importance.png")
    
    def _plot_lstm_history(self):
        """Plot LSTM training history"""
        
        history = self.results['lstm']['history']
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        axes[0].plot(history['loss'], label='Training Loss')
        axes[0].plot(history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('LSTM Training History - Loss')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Accuracy
        axes[1].plot(history['accuracy'], label='Training Accuracy')
        axes[1].plot(history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('LSTM Training History - Accuracy')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'lstm_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Saved lstm_history.png")
    
    def save_results(self):
        """Save all results to JSON"""
        
        # Convert numpy types to Python types
        results_serializable = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                results_serializable[key] = {}
                for k, v in value.items():
                    if isinstance(v, (np.integer, np.floating)):
                        results_serializable[key][k] = float(v)
                    elif isinstance(v, np.ndarray):
                        results_serializable[key][k] = v.tolist()
                    else:
                        results_serializable[key][k] = v
        
        with open(self.output_dir / 'training_results.json', 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        logger.info(f"Results saved to {self.output_dir / 'training_results.json'}")
    
    def run_pipeline(self, tune_hyperparameters: bool = False) -> dict:
        """Run complete training pipeline"""
        logger.info("\n" + "="*100)
        logger.info("SPRINT 3: MODEL TRAINING & EVALUATION PIPELINE")
        logger.info("="*100)
        logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load data
        X, y = self.load_data()
        if X is None:
            logger.error("Failed to load data")
            return {}
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # Handle imbalance
        X_train_balanced, y_train_balanced = self.handle_imbalance(X_train, y_train, method='smote')
        
        # Train Random Forest
        self.train_random_forest(X_train_balanced, y_train_balanced, X_val, y_val,
                                tune_hyperparameters=tune_hyperparameters)
        
        # Train SVM
        self.train_svm(X_train_balanced, y_train_balanced, X_val, y_val,
                      tune_hyperparameters=tune_hyperparameters)
        
        # Train LSTM
        self.train_lstm(X_train, y_train, X_val, y_val, sequence_length=24)
        
        # Evaluate on test set
        self.evaluate_on_test_set(X_test, y_test)
        
        # Generate visualizations
        self.generate_visualizations()
        
        # Save results
        self.save_results()
        
        # Save metrics to database
        self._save_metrics_to_db()
        
        logger.info("\n" + "="*100)
        logger.info("PIPELINE SUMMARY")
        logger.info("="*100)
        logger.info(f"Models trained: {len(self.models)}")
        logger.info(f"Best model: {self._get_best_model()}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Models directory: {self.models_dir}")
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*100)
        
        return self.results
    
    def _get_best_model(self) -> str:
        """Determine best model based on F1-score"""
        best_model = None
        best_f1 = 0
        
        for model_name, results in self.results.items():
            if model_name != 'test' and 'f1_score' in results:
                if results['f1_score'] > best_f1:
                    best_f1 = results['f1_score']
                    best_model = model_name
        
        return f"{best_model} (F1: {best_f1:.4f})" if best_model else "N/A"
    
    def _save_metrics_to_db(self):
        """Save model metrics to database"""
        logger.info("\nSaving metrics to database...")
        
        for model_name, results in self.results.items():
            if model_name == 'test':
                continue
            
            metrics = {
                'model_name': model_name,
                'model_type': model_name.replace('_', ' ').title(),
                'accuracy': results.get('accuracy', 0),
                'precision': results.get('precision', 0),
                'recall': results.get('recall', 0),
                'f1_score': results.get('f1_score', 0),
                'roc_auc': results.get('roc_auc', 0),
                'training_date': datetime.now()
            }
            
            self.db.insert_model_metrics(metrics)
        
        logger.info("Metrics saved to database")


def main():
    """Main execution"""
    pipeline = ModelTrainingPipeline()
    
    # Run pipeline with hyperparameter tuning (set to False for faster execution)
    results = pipeline.run_pipeline(tune_hyperparameters=False)
    
    print("\n" + "="*100)
    print("SPRINT 3 COMPLETE!")
    print("="*100)
    print(f"Successfully trained {len(pipeline.models)} models")
    print(f"Best model: {pipeline._get_best_model()}")
    print(f"Results saved to: {pipeline.output_dir}")
    print(f"Models saved to: {pipeline.models_dir}")
    print("="*100)


if __name__ == "__main__":
    main()
