"""
A/B Testing Framework for Model Comparison

Enables controlled experiments comparing different model versions in production.
"""
import logging
import json
import random
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from enum import Enum
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrafficSplitStrategy(Enum):
    """Traffic splitting strategies"""
    RANDOM = "random"
    PERCENTAGE = "percentage"
    USER_HASH = "user_hash"
    GRADUAL_ROLLOUT = "gradual_rollout"


@dataclass
class ModelVariant:
    """Model variant configuration"""
    variant_id: str
    model_path: str
    model_type: str
    version: str
    traffic_percentage: float
    description: str = ""
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ExperimentResult:
    """Single experiment result"""
    experiment_id: str
    variant_id: str
    timestamp: datetime
    latitude: float
    longitude: float
    prediction: int
    probability: float
    actual_outcome: Optional[int] = None
    response_time_ms: float = 0.0
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ABTestingFramework:
    """A/B testing framework for model comparison"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.experiments_dir = Path("models/experiments")
        self.results_dir = Path("models/experiment_results")
        
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Active experiments
        self.active_experiments: Dict[str, Dict] = {}
        self.experiment_results: Dict[str, List[ExperimentResult]] = defaultdict(list)
        
        # Load existing experiments
        self._load_experiments()
    
    def create_experiment(
        self,
        experiment_id: str,
        variants: List[ModelVariant],
        strategy: TrafficSplitStrategy = TrafficSplitStrategy.PERCENTAGE,
        description: str = ""
    ) -> Dict:
        """
        Create new A/B test experiment
        
        Args:
            experiment_id: Unique experiment identifier
            variants: List of model variants to test
            strategy: Traffic splitting strategy
            description: Experiment description
            
        Returns:
            Experiment configuration dictionary
        """
        # Validate traffic percentages
        total_traffic = sum(v.traffic_percentage for v in variants)
        if not np.isclose(total_traffic, 100.0, atol=0.01):
            raise ValueError(f"Traffic percentages must sum to 100, got {total_traffic}")
        
        # Create experiment config
        experiment = {
            'experiment_id': experiment_id,
            'strategy': strategy.value,
            'description': description,
            'variants': [asdict(v) for v in variants],
            'created_at': datetime.now().isoformat(),
            'status': 'active',
            'total_requests': 0,
            'variant_counts': {v.variant_id: 0 for v in variants}
        }
        
        # Save experiment
        self.active_experiments[experiment_id] = experiment
        self._save_experiment(experiment)
        
        logger.info(f"✅ Created experiment: {experiment_id} with {len(variants)} variants")
        
        return experiment
    
    def select_variant(
        self,
        experiment_id: str,
        user_id: Optional[str] = None
    ) -> Optional[ModelVariant]:
        """
        Select model variant based on traffic splitting strategy
        
        Args:
            experiment_id: Experiment identifier
            user_id: User identifier for consistent variant assignment
            
        Returns:
            Selected ModelVariant or None if experiment not found
        """
        if experiment_id not in self.active_experiments:
            logger.warning(f"Experiment not found: {experiment_id}")
            return None
        
        experiment = self.active_experiments[experiment_id]
        strategy = TrafficSplitStrategy(experiment['strategy'])
        variants = [ModelVariant(**v) for v in experiment['variants']]
        
        # Select based on strategy
        if strategy == TrafficSplitStrategy.RANDOM:
            selected = self._random_selection(variants)
        
        elif strategy == TrafficSplitStrategy.PERCENTAGE:
            selected = self._percentage_selection(variants)
        
        elif strategy == TrafficSplitStrategy.USER_HASH:
            if user_id is None:
                user_id = str(random.random())
            selected = self._hash_based_selection(variants, user_id)
        
        elif strategy == TrafficSplitStrategy.GRADUAL_ROLLOUT:
            selected = self._gradual_rollout_selection(variants, experiment)
        
        else:
            selected = variants[0]  # Default to first variant
        
        # Update counters
        experiment['total_requests'] += 1
        experiment['variant_counts'][selected.variant_id] += 1
        
        return selected
    
    def _random_selection(self, variants: List[ModelVariant]) -> ModelVariant:
        """Random variant selection"""
        return random.choice(variants)
    
    def _percentage_selection(self, variants: List[ModelVariant]) -> ModelVariant:
        """Weighted random selection based on traffic percentages"""
        rand = random.uniform(0, 100)
        cumulative = 0
        
        for variant in variants:
            cumulative += variant.traffic_percentage
            if rand <= cumulative:
                return variant
        
        return variants[-1]  # Fallback
    
    def _hash_based_selection(self, variants: List[ModelVariant], user_id: str) -> ModelVariant:
        """Consistent hash-based selection for same user"""
        import hashlib
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        bucket = hash_value % 100
        
        cumulative = 0
        for variant in variants:
            cumulative += variant.traffic_percentage
            if bucket < cumulative:
                return variant
        
        return variants[-1]
    
    def _gradual_rollout_selection(
        self,
        variants: List[ModelVariant],
        experiment: Dict
    ) -> ModelVariant:
        """Gradual rollout - increase traffic to new variant over time"""
        total_requests = experiment['total_requests']
        
        # Adjust traffic based on request count
        if total_requests < 100:
            # First 100 requests: 10% new variant
            adjusted_variants = variants.copy()
            adjusted_variants[1].traffic_percentage = 10
            adjusted_variants[0].traffic_percentage = 90
        elif total_requests < 500:
            # Next 400: 30% new variant
            adjusted_variants = variants.copy()
            adjusted_variants[1].traffic_percentage = 30
            adjusted_variants[0].traffic_percentage = 70
        elif total_requests < 1000:
            # Next 500: 50% new variant
            adjusted_variants = variants.copy()
            adjusted_variants[1].traffic_percentage = 50
            adjusted_variants[0].traffic_percentage = 50
        else:
            # After 1000: use original percentages
            adjusted_variants = variants
        
        return self._percentage_selection(adjusted_variants)
    
    def record_result(self, result: ExperimentResult):
        """
        Record experiment result
        
        Args:
            result: ExperimentResult object
        """
        self.experiment_results[result.experiment_id].append(result)
        
        # Periodically save results
        if len(self.experiment_results[result.experiment_id]) % 100 == 0:
            self._save_results(result.experiment_id)
    
    def update_actual_outcome(
        self,
        experiment_id: str,
        timestamp: datetime,
        actual_outcome: int
    ):
        """
        Update actual outcome for a prediction (for metric calculation)
        
        Args:
            experiment_id: Experiment identifier
            timestamp: Timestamp of original prediction
            actual_outcome: Actual cloud burst occurrence (0 or 1)
        """
        if experiment_id not in self.experiment_results:
            logger.warning(f"No results for experiment: {experiment_id}")
            return
        
        # Find matching result and update
        for result in self.experiment_results[experiment_id]:
            if result.timestamp == timestamp:
                result.actual_outcome = actual_outcome
                logger.info(f"Updated actual outcome for {experiment_id} at {timestamp}")
                break
    
    def analyze_experiment(self, experiment_id: str) -> Dict:
        """
        Analyze experiment results and compare variants
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Dictionary with analysis results
        """
        if experiment_id not in self.experiment_results:
            return {'error': f'No results for experiment: {experiment_id}'}
        
        results = self.experiment_results[experiment_id]
        
        if not results:
            return {'error': 'No results collected yet'}
        
        # Convert to DataFrame
        df = pd.DataFrame([asdict(r) for r in results])
        
        # Analyze by variant
        variant_analysis = {}
        
        for variant_id in df['variant_id'].unique():
            variant_df = df[df['variant_id'] == variant_id]
            
            analysis = {
                'total_requests': len(variant_df),
                'avg_probability': variant_df['probability'].mean(),
                'prediction_rate': variant_df['prediction'].mean(),
                'avg_response_time_ms': variant_df['response_time_ms'].mean()
            }
            
            # If actual outcomes available, calculate accuracy
            if variant_df['actual_outcome'].notna().any():
                valid_df = variant_df[variant_df['actual_outcome'].notna()]
                analysis['accuracy'] = (
                    valid_df['prediction'] == valid_df['actual_outcome']
                ).mean()
                analysis['precision'] = (
                    valid_df[valid_df['prediction'] == 1]['actual_outcome'] == 1
                ).sum() / max((valid_df['prediction'] == 1).sum(), 1)
                analysis['recall'] = (
                    valid_df[valid_df['actual_outcome'] == 1]['prediction'] == 1
                ).sum() / max((valid_df['actual_outcome'] == 1).sum(), 1)
                analysis['validated_samples'] = len(valid_df)
            
            variant_analysis[variant_id] = analysis
        
        # Statistical comparison
        winner = self._determine_winner(variant_analysis)
        
        return {
            'experiment_id': experiment_id,
            'total_requests': len(df),
            'variants': variant_analysis,
            'winner': winner,
            'analyzed_at': datetime.now().isoformat()
        }
    
    def _determine_winner(self, variant_analysis: Dict) -> Dict:
        """Determine winning variant based on performance metrics"""
        if len(variant_analysis) < 2:
            return {'message': 'Need at least 2 variants to compare'}
        
        # Score each variant
        scores = {}
        for variant_id, metrics in variant_analysis.items():
            score = 0
            
            # Weight different metrics
            if 'accuracy' in metrics:
                score += metrics['accuracy'] * 0.5
            if 'precision' in metrics:
                score += metrics['precision'] * 0.25
            if 'recall' in metrics:
                score += metrics['recall'] * 0.25
            
            # Response time penalty (normalize to 0-1 scale)
            if 'avg_response_time_ms' in metrics:
                max_time = max(m.get('avg_response_time_ms', 0) 
                             for m in variant_analysis.values())
                if max_time > 0:
                    time_score = 1 - (metrics['avg_response_time_ms'] / max_time)
                    score += time_score * 0.1
            
            scores[variant_id] = score
        
        winner_id = max(scores, key=scores.get)
        
        return {
            'winner_variant': winner_id,
            'winner_score': scores[winner_id],
            'all_scores': scores,
            'confidence': 'high' if scores[winner_id] > 0.8 else 'medium' if scores[winner_id] > 0.6 else 'low'
        }
    
    def stop_experiment(self, experiment_id: str):
        """Stop an active experiment"""
        if experiment_id in self.active_experiments:
            self.active_experiments[experiment_id]['status'] = 'stopped'
            self.active_experiments[experiment_id]['stopped_at'] = datetime.now().isoformat()
            self._save_experiment(self.active_experiments[experiment_id])
            self._save_results(experiment_id)
            logger.info(f"Stopped experiment: {experiment_id}")
    
    def _save_experiment(self, experiment: Dict):
        """Save experiment configuration"""
        filepath = self.experiments_dir / f"{experiment['experiment_id']}.json"
        with open(filepath, 'w') as f:
            json.dump(experiment, f, indent=2)
    
    def _save_results(self, experiment_id: str):
        """Save experiment results"""
        if experiment_id not in self.experiment_results:
            return
        
        results = self.experiment_results[experiment_id]
        filepath = self.results_dir / f"{experiment_id}_results.json"
        
        results_data = [asdict(r) for r in results]
        # Convert datetime to string
        for r in results_data:
            r['timestamp'] = r['timestamp'].isoformat() if isinstance(r['timestamp'], datetime) else r['timestamp']
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logger.info(f"Saved {len(results)} results for {experiment_id}")
    
    def _load_experiments(self):
        """Load existing experiments from disk"""
        if not self.experiments_dir.exists():
            return
        
        for filepath in self.experiments_dir.glob("*.json"):
            try:
                with open(filepath, 'r') as f:
                    experiment = json.load(f)
                
                if experiment.get('status') == 'active':
                    self.active_experiments[experiment['experiment_id']] = experiment
                    logger.info(f"Loaded active experiment: {experiment['experiment_id']}")
            except Exception as e:
                logger.error(f"Error loading experiment {filepath}: {e}")
    
    def list_experiments(self, status: str = None) -> List[Dict]:
        """
        List all experiments
        
        Args:
            status: Filter by status ('active', 'stopped', or None for all)
            
        Returns:
            List of experiment summaries
        """
        experiments = []
        
        for filepath in self.experiments_dir.glob("*.json"):
            with open(filepath, 'r') as f:
                exp = json.load(f)
            
            if status is None or exp.get('status') == status:
                experiments.append({
                    'experiment_id': exp['experiment_id'],
                    'status': exp['status'],
                    'variants': len(exp['variants']),
                    'total_requests': exp.get('total_requests', 0),
                    'created_at': exp['created_at']
                })
        
        return sorted(experiments, key=lambda x: x['created_at'], reverse=True)


if __name__ == "__main__":
    # Example usage
    framework = ABTestingFramework()
    
    # Create experiment
    variants = [
        ModelVariant(
            variant_id='control',
            model_path='models/trained/random_forest_model.pkl',
            model_type='random_forest',
            version='v1',
            traffic_percentage=50.0,
            description='Current production model'
        ),
        ModelVariant(
            variant_id='treatment',
            model_path='models/versions/random_forest_v20250101_120000.pkl',
            model_type='random_forest',
            version='v2',
            traffic_percentage=50.0,
            description='New improved model'
        )
    ]
    
    experiment = framework.create_experiment(
        experiment_id='rf_v1_vs_v2',
        variants=variants,
        strategy=TrafficSplitStrategy.PERCENTAGE,
        description='Compare original RF with retrained version'
    )
    
    print(json.dumps(experiment, indent=2))
