"""
Feature Validation and Selection

Advanced feature engineering validation including:
- Feature importance analysis
- Correlation analysis
- Statistical testing
- Feature selection algorithms
- Data quality checks
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Scikit-learn imports
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, SelectFromModel
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, pearsonr
from scipy.cluster import hierarchy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureValidator:
    """Validate and analyze feature quality"""
    
    def __init__(self, output_dir: str = "./reports"):
        """
        Initialize feature validator
        
        Args:
            output_dir: Directory to save reports and plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate data quality
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with quality metrics
        """
        logger.info("="*80)
        logger.info("DATA QUALITY VALIDATION")
        logger.info("="*80)
        
        quality = {}
        
        # Basic statistics
        quality['total_rows'] = len(df)
        quality['total_columns'] = len(df.columns)
        
        # Missing values
        missing = df.isnull().sum()
        quality['missing_values'] = missing[missing > 0].to_dict()
        quality['total_missing'] = df.isnull().sum().sum()
        quality['missing_percentage'] = (quality['total_missing'] / (len(df) * len(df.columns))) * 100
        
        # Infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_counts = {}
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                inf_counts[col] = inf_count
        quality['infinite_values'] = inf_counts
        
        # Duplicate rows
        quality['duplicate_rows'] = df.duplicated().sum()
        
        # Constant columns (no variance)
        constant_cols = []
        for col in numeric_cols:
            if df[col].nunique() == 1:
                constant_cols.append(col)
        quality['constant_columns'] = constant_cols
        
        # Data types
        quality['dtypes'] = df.dtypes.value_counts().to_dict()
        
        # Print summary
        logger.info(f"\n📊 Data Shape: {quality['total_rows']} rows × {quality['total_columns']} columns")
        logger.info(f"❌ Missing Values: {quality['total_missing']} ({quality['missing_percentage']:.2f}%)")
        logger.info(f"∞ Infinite Values: {sum(inf_counts.values())} in {len(inf_counts)} columns")
        logger.info(f"🔁 Duplicate Rows: {quality['duplicate_rows']}")
        logger.info(f"📏 Constant Columns: {len(constant_cols)}")
        
        if quality['missing_percentage'] > 5:
            logger.warning("⚠️ High percentage of missing values detected!")
        
        if quality['duplicate_rows'] > 0:
            logger.warning(f"⚠️ {quality['duplicate_rows']} duplicate rows found!")
        
        logger.info("="*80)
        
        return quality
    
    def analyze_feature_importance(self, X: pd.DataFrame, y: pd.Series,
                                   method: str = 'random_forest',
                                   top_n: int = 20) -> pd.DataFrame:
        """
        Calculate feature importance
        
        Args:
            X: Feature DataFrame
            y: Target variable
            method: Method to use ('random_forest', 'extra_trees', 'mutual_info')
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importances
        """
        logger.info(f"Calculating feature importance using {method}...")
        
        if method == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X, y)
            importances = model.feature_importances_
        
        elif method == 'extra_trees':
            model = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X, y)
            importances = model.feature_importances_
        
        elif method == 'mutual_info':
            importances = mutual_info_classif(X, y, random_state=42)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Plot top N features
        plt.figure(figsize=(10, 8))
        top_features = importance_df.head(top_n)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Features - {method.replace("_", " ").title()}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(self.output_dir / f'feature_importance_{method}.png', dpi=300)
        plt.close()
        
        logger.info(f"✓ Top features saved to {self.output_dir / f'feature_importance_{method}.png'}")
        
        return importance_df
    
    def analyze_correlations(self, df: pd.DataFrame, 
                            threshold: float = 0.8,
                            method: str = 'pearson') -> Tuple[pd.DataFrame, List[str]]:
        """
        Analyze feature correlations
        
        Args:
            df: DataFrame with features
            threshold: Correlation threshold for identifying redundant features
            method: Correlation method ('pearson' or 'spearman')
            
        Returns:
            Tuple of (correlation matrix, list of redundant features)
        """
        logger.info(f"Analyzing correlations (method={method}, threshold={threshold})...")
        
        # Remove constant columns before correlation analysis
        numeric_df = df.select_dtypes(include=[np.number])
        non_constant_cols = [col for col in numeric_df.columns 
                            if numeric_df[col].nunique() > 1]
        df_filtered = numeric_df[non_constant_cols]
        
        if len(df_filtered.columns) == 0:
            logger.warning("No non-constant columns found for correlation analysis")
            return pd.DataFrame(), []
        
        # Calculate correlation matrix
        if method == 'pearson':
            corr_matrix = df_filtered.corr()
        elif method == 'spearman':
            corr_matrix = df_filtered.corr(method='spearman')
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Replace NaN values with 0
        corr_matrix = corr_matrix.fillna(0)
        
        # Find highly correlated features
        redundant_features = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    colname = corr_matrix.columns[j]
                    redundant_features.add(colname)
        
        redundant_features = list(redundant_features)
        
        # Plot correlation heatmap
        plt.figure(figsize=(20, 16))
        
        # Cluster features by correlation if enough features
        if len(df_filtered.columns) > 2:
            try:
                # Convert correlation to distance matrix
                distance_matrix = 1 - np.abs(corr_matrix)
                # Replace any remaining NaN/inf values
                distance_matrix = distance_matrix.fillna(1).replace([np.inf, -np.inf], 1)
                
                # Create linkage matrix
                from scipy.spatial.distance import squareform
                condensed_dist = squareform(distance_matrix)
                corr_linkage = hierarchy.linkage(condensed_dist, method='average')
                dendro = hierarchy.dendrogram(corr_linkage, labels=corr_matrix.columns, 
                                             no_plot=True)
                
                # Reorder matrix based on clustering
                idx = dendro['leaves']
                corr_matrix_ordered = corr_matrix.iloc[idx, idx]
            except Exception as e:
                logger.warning(f"Clustering failed: {e}. Using original order.")
                corr_matrix_ordered = corr_matrix
        else:
            corr_matrix_ordered = corr_matrix
        
        # Plot heatmap
        sns.heatmap(corr_matrix_ordered, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                   vmin=-1, vmax=1, annot=False)
        plt.title(f'Feature Correlation Matrix ({method.title()})')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'correlation_matrix_{method}.png', dpi=300)
        plt.close()
        
        logger.info(f"✓ Correlation matrix saved to {self.output_dir / f'correlation_matrix_{method}.png'}")
        logger.info(f"✓ Found {len(redundant_features)} highly correlated features (>{threshold})")
        
        return corr_matrix, redundant_features


class FeatureSelector:
    """Select best features using multiple methods"""
    
    def __init__(self):
        """Initialize feature selector"""
        self.selected_features = {}
    
    def select_k_best(self, X: pd.DataFrame, y: pd.Series, 
                     k: int = 50, score_func=f_classif) -> List[str]:
        """
        Select k best features using statistical tests
        
        Args:
            X: Feature DataFrame
            y: Target variable
            k: Number of features to select
            score_func: Scoring function (f_classif or mutual_info_classif)
            
        Returns:
            List of selected feature names
        """
        logger.info(f"Selecting {k} best features using {score_func.__name__}...")
        
        selector = SelectKBest(score_func=score_func, k=min(k, X.shape[1]))
        selector.fit(X, y)
        
        # Get selected features
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        
        # Get scores
        scores = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_
        }).sort_values('score', ascending=False)
        
        self.selected_features['k_best'] = selected_features
        
        logger.info(f"✓ Selected {len(selected_features)} features")
        
        return selected_features
    
    def select_from_model(self, X: pd.DataFrame, y: pd.Series,
                         threshold: str = 'median') -> List[str]:
        """
        Select features based on importance from tree-based model
        
        Args:
            X: Feature DataFrame
            y: Target variable
            threshold: Threshold for feature selection ('mean', 'median', or float)
            
        Returns:
            List of selected feature names
        """
        logger.info(f"Selecting features using model importance (threshold={threshold})...")
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)
        
        # Select features
        selector = SelectFromModel(model, threshold=threshold, prefit=True)
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        
        self.selected_features['from_model'] = selected_features
        
        logger.info(f"✓ Selected {len(selected_features)} features")
        
        return selected_features
    
    def select_rfe(self, X: pd.DataFrame, y: pd.Series,
                   n_features: int = 50) -> List[str]:
        """
        Recursive Feature Elimination
        
        Args:
            X: Feature DataFrame
            y: Target variable
            n_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        logger.info(f"Performing RFE to select {n_features} features...")
        
        # Use Random Forest for RFE (faster than SVM)
        estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        selector = RFE(estimator, n_features_to_select=min(n_features, X.shape[1]), step=5)
        selector.fit(X, y)
        
        # Get selected features
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        
        self.selected_features['rfe'] = selected_features
        
        logger.info(f"✓ Selected {len(selected_features)} features")
        
        return selected_features
    
    def select_by_variance(self, X: pd.DataFrame, threshold: float = 0.01) -> List[str]:
        """
        Select features with variance above threshold
        
        Args:
            X: Feature DataFrame
            threshold: Minimum variance threshold
            
        Returns:
            List of selected feature names
        """
        logger.info(f"Selecting features with variance > {threshold}...")
        
        # Calculate variance for each feature
        variances = X.var()
        selected_features = variances[variances > threshold].index.tolist()
        
        removed_count = len(X.columns) - len(selected_features)
        
        self.selected_features['by_variance'] = selected_features
        
        logger.info(f"✓ Removed {removed_count} low-variance features")
        logger.info(f"✓ {len(selected_features)} features remaining")
        
        return selected_features
    
    def get_consensus_features(self, min_votes: int = 2) -> List[str]:
        """
        Get features selected by multiple methods (voting)
        
        Args:
            min_votes: Minimum number of methods that must select a feature
            
        Returns:
            List of consensus feature names
        """
        logger.info(f"Finding consensus features (min_votes={min_votes})...")
        
        if not self.selected_features:
            logger.warning("No feature selection methods have been run yet")
            return []
        
        # Count votes for each feature
        all_features = set()
        for features in self.selected_features.values():
            all_features.update(features)
        
        feature_votes = {}
        for feature in all_features:
            votes = sum(1 for features in self.selected_features.values() 
                       if feature in features)
            feature_votes[feature] = votes
        
        # Select features with enough votes
        consensus_features = [f for f, votes in feature_votes.items() 
                             if votes >= min_votes]
        
        # Sort by votes
        consensus_features.sort(key=lambda f: feature_votes[f], reverse=True)
        
        logger.info(f"✓ Found {len(consensus_features)} consensus features")
        logger.info(f"  Methods used: {list(self.selected_features.keys())}")
        
        return consensus_features


if __name__ == "__main__":
    print("="*80)
    print("FEATURE VALIDATION & SELECTION TEST")
    print("="*80)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Create features with different characteristics
    df_test = pd.DataFrame({
        'important_1': np.random.randn(n_samples) * 2 + 5,
        'important_2': np.random.randn(n_samples) * 1.5 + 3,
        'correlated_1': np.random.randn(n_samples),
        'noise_1': np.random.randn(n_samples) * 0.1,
        'noise_2': np.random.randn(n_samples) * 0.1,
        'constant': np.ones(n_samples),
    })
    
    # Add correlated feature
    df_test['correlated_2'] = df_test['correlated_1'] * 0.95 + np.random.randn(n_samples) * 0.1
    
    # Create target based on important features
    df_test['target'] = ((df_test['important_1'] > 5) & 
                         (df_test['important_2'] > 3)).astype(int)
    
    print(f"\nSample data: {df_test.shape}")
    print(f"Target distribution: {df_test['target'].value_counts().to_dict()}")
    
    # Test validation
    validator = FeatureValidator(output_dir="./reports/test")
    
    X = df_test.drop('target', axis=1)
    y = df_test['target']
    
    quality = validator.validate_data_quality(df_test)
    
    # Test feature importance
    importance_df = validator.analyze_feature_importance(X, y, method='random_forest', top_n=7)
    print(f"\nTop 3 features:")
    print(importance_df.head(3))
    
    # Test correlation analysis
    corr_matrix, redundant = validator.analyze_correlations(X, threshold=0.8)
    print(f"\nRedundant features: {redundant}")
    
    # Test feature selection
    selector = FeatureSelector()
    
    k_best = selector.select_k_best(X, y, k=5)
    print(f"\nK-best features: {k_best}")
    
    from_model = selector.select_from_model(X, y, threshold='mean')
    print(f"From model features: {from_model}")
    
    by_variance = selector.select_by_variance(X, threshold=0.01)
    print(f"By variance features: {by_variance}")
    
    consensus = selector.get_consensus_features(min_votes=2)
    print(f"\nConsensus features: {consensus}")
    
    print("\n" + "="*80)
