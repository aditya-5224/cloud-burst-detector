"""
Prediction Service - Core prediction logic for cloud burst forecasting
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionService:
    """Service for making cloud burst predictions"""
    
    def __init__(self, model_dir: str = "./models"):
        """Initialize prediction service with trained models"""
        # Get absolute path relative to project root
        if not Path(model_dir).is_absolute():
            # Try to find project root
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent  # Go up to project root
            self.model_dir = project_root / model_dir.lstrip('./')
        else:
            self.model_dir = Path(model_dir)
        
        self.models = {}
        self.feature_names = None
        self.model_loaded = False
        self._load_primary_model()
    
    def _load_primary_model(self):
        """Load the primary Random Forest model"""
        try:
            logger.info(f"Looking for model in: {self.model_dir}")
            
            # Try trained subdirectory first
            model_path = self.model_dir / 'trained' / 'random_forest_model.pkl'
            logger.info(f"Trying path: {model_path}")
            
            # Fallback to root models directory
            if not model_path.exists():
                model_path = self.model_dir / 'random_forest_model.pkl'
                logger.info(f"Trying fallback path: {model_path}")
            
            if not model_path.exists():
                logger.error(f"Model not found at: {model_path}")
                logger.error(f"Current working directory: {Path.cwd()}")
                logger.error(f"Model directory contents: {list(self.model_dir.iterdir()) if self.model_dir.exists() else 'Directory does not exist'}")
                return
            
            self.models['random_forest'] = joblib.load(model_path)
            # Ensure feature_names is a Python list, not numpy array
            self.feature_names = list(self.models['random_forest'].feature_names_in_)
            self.model_loaded = True
            
            logger.info(f"Loaded Random Forest model from {model_path}")
            logger.info(f"Expected features: {len(self.feature_names)}")
            logger.info(f"Feature names type: {type(self.feature_names)}")
            
        except Exception as e:
            logger.error(f"Failed to load primary model: {e}")
            self.model_loaded = False
    
    def validate_features(self, features: pd.DataFrame) -> Tuple[bool, str]:
        """Validate input features"""
        if not self.model_loaded:
            return False, "Model not loaded"
        
        if len(features.columns) != len(self.feature_names):
            return False, f"Expected {len(self.feature_names)} features, got {len(features.columns)}"
        
        missing_features = set(self.feature_names) - set(features.columns)
        if missing_features:
            return False, f"Missing features: {missing_features}"
        
        if features.isnull().any().any():
            return False, "Features contain null values"
        
        if np.isinf(features.values).any():
            return False, "Features contain infinite values"
        
        return True, ""
    
    def predict(self, features: pd.DataFrame, model_name: str = 'random_forest') -> Dict:
        """Make prediction using specified model"""
        logger.info(f"Received features DataFrame with columns: {list(features.columns)}")
        logger.info(f"Expected feature names: {self.feature_names}")
        logger.info(f"Columns match: {list(features.columns) == self.feature_names}")
        
        is_valid, error_msg = self.validate_features(features)
        if not is_valid:
            return {
                'success': False,
                'error': error_msg,
                'prediction': None,
                'probability': None
            }
        
        # Features are already in correct order, no need to reorder
        # Just ensure they match exactly
        logger.info(f"Features validated successfully")
        
        try:
            prediction = self.models[model_name].predict(features)
            probability = self.models[model_name].predict_proba(features)[:, 1]
            risk_level = self._interpret_probability(probability[0])
            
            return {
                'success': True,
                'prediction': int(prediction[0]),
                'probability': float(probability[0]),
                'risk_level': risk_level,
                'model': model_name,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'success': False,
                'error': str(e),
                'prediction': None,
                'probability': None
            }
    
    def _interpret_probability(self, probability: float) -> str:
        """Interpret probability score as risk level"""
        if probability >= 0.8:
            return "EXTREME"
        elif probability >= 0.6:
            return "HIGH"
        elif probability >= 0.4:
            return "MODERATE"
        elif probability >= 0.2:
            return "LOW"
        else:
            return "MINIMAL"
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """Get feature importance from Random Forest model"""
        if 'random_forest' not in self.models:
            return {}
        
        model = self.models['random_forest']
        importances = model.feature_importances_
        
        feature_importance = dict(zip(self.feature_names, importances))
        sorted_features = sorted(feature_importance.items(), 
                                key=lambda x: x[1], reverse=True)
        
        return dict(sorted_features[:top_n])
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        return {
            'models_loaded': list(self.models.keys()),
            'primary_model': 'random_forest',
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'model_ready': self.model_loaded
        }


# Singleton instance
_prediction_service = None

def get_prediction_service() -> PredictionService:
    """Get singleton prediction service instance"""
    global _prediction_service
    
    if _prediction_service is None:
        _prediction_service = PredictionService()
    
    return _prediction_service


if __name__ == "__main__":
    print("="*80)
    print("PREDICTION SERVICE TEST")
    print("="*80)
    
    service = PredictionService()
    info = service.get_model_info()
    print(f"\nModel Info: {info}")
    
    if service.model_loaded and service.feature_names:
        print(f"\nExpected features: {len(service.feature_names)}")
        print(f"First 5 features: {service.feature_names[:5]}")
        
        # Create dummy features for testing
        dummy_features = pd.DataFrame({
            feature: [0.0] for feature in service.feature_names
        })
        
        print("\nTesting prediction with dummy data...")
        result = service.predict(dummy_features)
        print(f"Result: {result}")
        
        print("\nTop 10 Important Features:")
        importance = service.get_feature_importance(top_n=10)
        for i, (feature, score) in enumerate(importance.items(), 1):
            print(f"  {i}. {feature}: {score:.4f}")
    
    print("\n" + "="*80)
